import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from time import perf_counter

from nanovllm.backends.tilelang.runtime import configure_tilelang_runtime
from nanovllm.config import Config
from nanovllm.engine.device import (
    dist_backend_name,
    is_rvv_device,
    torch_device_name,
    use_pin_memory,
)
from nanovllm.engine.rvv_planner import compile_engine_rvv_kernels
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import set_attn_backend
from nanovllm.layers.linear import set_linear_backend
from nanovllm.layers.layernorm import set_norm_backend
from nanovllm.layers.activation import set_act_backend
from nanovllm.layers.rotary_embedding import set_rope_backend
from nanovllm.layers.embed_head import set_embed_backend
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.backends.spike.report_context import (
    active_collector,
    begin_engine_step,
    install_collector,
    report_context,
    set_warmup_phase,
)
from nanovllm.backends.spike.generate_report import GenerateReportCollector


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        set_attn_backend(config.attn_backend)
        set_linear_backend(config.linear_backend)
        set_norm_backend(config.norm_backend)
        set_act_backend(config.act_backend)
        set_rope_backend(config.rope_backend)
        set_embed_backend(config.embed_backend)
        if config.rvv_generate_report:
            install_collector(
                GenerateReportCollector.open(
                    build_dir=config.rvv_build_dir,
                    config=config,
                )
            )
            collector = active_collector()
            assert collector is not None
            collector.mark_init_start()
        configure_tilelang_runtime(
            backend=config.tilelang_backend,
            nr_lanes=config.rvv_nr_lanes,
            build_dir=config.rvv_build_dir,
            compile_only=config.rvv_compile_only,
            execution_backend=config.rvv_execution_backend,
        )
        self.device_name = torch_device_name(config)
        self._pin_memory = use_pin_memory(config)
        dist.init_process_group(
            dist_backend_name(config),
            "tcp://localhost:2333",
            world_size=self.world_size,
            rank=rank,
        )
        if not is_rvv_device(config):
            torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device_name)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        if is_rvv_device(config) and config.rvv_compile_only:
            self.allocate_kv_cache()
            self.compile_rvv_kernels()
        else:
            self.warmup_model()
            self.allocate_kv_cache()
            if not self.enforce_eager and not is_rvv_device(config):
                self.capture_cudagraph()
        collector = active_collector()
        if collector is not None:
            collector.mark_init_end()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager and not is_rvv_device(self.config):
            del self.graphs, self.graph_pool
        if not is_rvv_device(self.config):
            torch.cuda.synchronize()
        dist.destroy_process_group()

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if is_rvv_device(self.config):
            return tensor
        return tensor.cuda(non_blocking=True)

    def _make_int_tensor(self, data, dtype: torch.dtype) -> torch.Tensor:
        return self._to_device(
            torch.tensor(data, dtype=dtype, pin_memory=self._pin_memory)
        )

    def compile_rvv_kernels(self) -> list[str]:
        """Lower the representative TileLang RVV kernel set for this engine config."""
        return compile_engine_rvv_kernels(self.config)

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        if not is_rvv_device(self.config):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        set_warmup_phase(True)
        try:
            self.run(seqs, True)
        finally:
            set_warmup_phase(False)
        if not is_rvv_device(self.config):
            torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        if is_rvv_device(config):
            if config.num_kvcache_blocks < 0:
                config.num_kvcache_blocks = config.rvv_num_kvcache_blocks
        else:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            device=self.device_name,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = self._make_int_tensor(block_tables, torch.int32)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = self._make_int_tensor(input_ids, torch.int64)
        positions = self._make_int_tensor(positions, torch.int64)
        cu_seqlens_q = self._make_int_tensor(cu_seqlens_q, torch.int32)
        cu_seqlens_k = self._make_int_tensor(cu_seqlens_k, torch.int32)
        slot_mapping = self._make_int_tensor(slot_mapping, torch.int32)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = self._make_int_tensor(input_ids, torch.int64)
        positions = self._make_int_tensor(positions, torch.int64)
        slot_mapping = self._make_int_tensor(slot_mapping, torch.int32)
        context_lens = self._make_int_tensor(context_lens, torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = self._to_device(
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=self._pin_memory)
        )
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        from nanovllm.backends.tilelang.runtime import (
            RvvExecutionUnavailableError,
            rvv_model_execution_enabled,
        )

        if is_rvv_device(self.config) and not rvv_model_execution_enabled(self.config):
            raise RvvExecutionUnavailableError(
                "device='rvv' without an execution backend: kernels were lowered during "
                f"init under {self.config.rvv_build_dir!r}. Set rvv_execution_backend='spike' "
                "or rvv_compile_only=False on an RVV host to run generate()."
            )
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512 or is_rvv_device(self.config):
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        from nanovllm.backends.spike.report_context import current_step_id

        collector = active_collector()
        phase = "prefill" if is_prefill else "decode"
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs)
        step_started_at = None
        step_id = -1
        if collector is not None and not report_context().in_warmup:
            begin_engine_step(phase=phase)
            step_id = current_step_id()
            step_started_at = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        if collector is not None and step_started_at is not None:
            sampled = token_ids[0] if token_ids else None
            collector.end_engine_step(
                phase=phase,
                step_id=step_id,
                num_tokens=num_tokens,
                started_at=step_started_at,
                logits=logits,
                sampled_token_id=sampled,
            )
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
