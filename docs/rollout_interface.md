# RLHF / Rollout 接口

## 1. 上游现状（只读引用自 `RLHF` 仓库）

| 项 | 证据 | 内容 |
| --- | --- | --- |
| RolloutServer | `rlhf/rollout_server.h:10-22` | `serve(prompts)` / `update_weights` / `load_checkpoint` |
| GPUActor | `rlhf/models/gpu_policy_model.h` + `rlhf/rollout/*` | `rollout(prompts)`，默认 `max_seq_len=2048` |
| KV Cache | `rlhf/rollout/kv_cache.h:9-30` | CPU dense `std::vector`，布局 `[num_layers, 2, batch, heads, seq, dim]` |
| 构建缺陷 | `RLHF/CMakeLists.txt:12` | `CMAKE_CUDA_ARCHITECTURES "80;86;89;90"` 不含 120 |
| 构建缺陷 | `RLHF/CMakeLists.txt:49-52` | `gpu_ops.cu` 未加入 `RLHF_CUDA_SOURCES` |
| 缺口 | `rlhf/models/gpu_policy_model.cpp` | GPU 路径没有 attention 实现 |

## 2. 两条集成路径

```text
路径 1（服务化）：RLHF → vLLM(C++) serve → SciCompute-Attention
  优点：复用调度/批处理/服务化；缺点：需先把注意力原型换成 SCA 后端
路径 2（直连）  ：RLHF → sca::flash_attention / decode_attention + PagedKVCache
  优点：最短路径、便于实验/教学；缺点：需要自己写批处理与调度
```

两条路径共用本项目的稳定 C++ API（`include/scicompute_attention/*`）。

## 3. 迁移路径（KV 布局）

```text
rlhf::KVCache（CPU dense [L,2,B,H,S,D]）
   → 逐层上传并写入 PagedKVCache（block-major [L][blocks][block_size][H_kv][D]）
   → page table 由 BlockManager 生成；slot = physical*block_size + offset
（可选）vllm::KVCache（每 layer 连续）↔ sca::KVCache：同构，可直接指针映射
```

## 4. 指标定义（与 benchmark 一致）

```text
tokens/s            = 生成 token 总数 / 端到端时间（含 prefill 与 decode）
prefill_latency     = 单次 prefill 的 P50/P99
decode_step_latency = 单步 decode 的 P50/P99
kv_peak_bytes       = KV Cache 峰值占用
batch_utilization   = 实际参与计算的序列数 / max_num_seqs
```

## 5. Stub 实测（`examples/rollout_engine_stub.cpp`）

```bash
$ ./build/examples/rollout_engine_stub --n 4 --max-new 8
[rollout] device=NVIDIA GeForce RTX 5070 Laptop GPU free=7602 MiB, KV plan: 262 blocks = 589.5 MiB
[rollout] requests=4 prompt_len=[32..47] max_new=8
[rollout] prefill  P50=0.245 ms P99=1.000 ms (flash, causal)
[rollout] decode   P50=0.020 ms P99=1.043 ms (paged, split-K, 4 序列批处理)
[rollout] tokens/s=11204.23 generated=32 kv_peak=589.5 MiB used_blocks=20/262
```

口径说明：`tokens/s` 为 stub 的**合成** decode 步吞吐（真实值取决于模型前向），
但 prefill/decode 的 kernel 时间是真实测量值。stub 用真实 kernel（flash + paged），
不含 PPO/GRPO 训练逻辑。

## 6. 复现

```bash
bash scripts/configure.sh --build-type Release && bash scripts/build.sh -j"$(nproc)"
./build/examples/rollout_engine_stub --n 8 --max-new 32 --json benchmarks/results/rollout.json
```

