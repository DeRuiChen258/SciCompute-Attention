# 真实模型对接：Qwen3-4B-Thinking-2507-Q8

> 数据文件：`docs/results/qwen3_4b_shapes.json`（由 `tools/model_probe.py` 生成）
> 读取器：`tools/gguf_reader.py`（纯 Python GGUF v3 实现，本机未安装 `gguf`/`llama_cpp`）

## 1. 模型事实（本机实测复核）

| 项 | 值 | 来源 |
| --- | --- | --- |
| 目录 / 权重 | `$SCA_MODEL_DIR/Qwen3-4B-Thinking-2507-Q8_0.gguf`（4,280,404,960 B） | 目录清单 |
| GGUF 版本 / 张量数 | 3 / 398 | `tools/gguf_reader.py` |
| 架构 / 量化 | `qwen3` / file_type 7（Q8_0） | GGUF 元数据 |
| 参数量 | 4,022,468,096 | GGUF `general.parameter_count` |
| 层数 | 36 | `qwen3.block_count` |
| 注意力头 | H_q=32，H_kv=8（**GQA group = 4**） | `qwen3.attention.head_count(_kv)` |
| head_dim | key/value length = 128 | `qwen3.attention.{key,value}_length` |
| hidden size | 2560 | `qwen3.embedding_length` |
| 上下文 | 262,144（本机 Ollama 运行时 4096） | `qwen3.context_length` |
| RoPE / RMS eps | 5e6 / 1e-6 | `qwen3.rope.freq_base` / `...rms_epsilon` |
| 注意力相关权重 | 144 个张量，956.2 MiB（q/k/v/o 全部 36 层） | `docs/results/qwen3_4b_shapes.json` |

推导出的注意力配置：`B=1, H_q=32, H_kv=8, D=128, causal=True`——恰好落在本项目的主力
head_dim（128）与 GQA 主路径（group=4）上。

## 2. GGUF 读取与反量化

```bash
python tools/gguf_reader.py <gguf> --meta            # 元数据
python tools/gguf_reader.py <gguf> --list-tensors    # 张量表（名称/维度/类型/字节）
python tools/gguf_reader.py <gguf> --dump-tensor blk.0.attn_q.weight
python tools/model_probe.py                          # → docs/results/qwen3_4b_shapes.json
```

支持类型：`F32`、`F16`、`Q8_0`（block = 2 B fp16 scale + 32×int8，34 B/32 权重）。
遇到其他类型抛 `UnsupportedGgmlType` 并打印类型号（禁止静默跳过）。

## 3. KV Cache 显存预算（本项目布局，fp16）

```text
每 token 字节 = 2(K,V) × 36(layers) × 8(kv_heads) × 128(head_dim) × 2(fp16) = 147,456 B = 144 KiB
每 16-token block = 2.25 MiB
```

| 上下文 | KV 大小 | 8 GB 显存可行性 |
| --- | --- | --- |
| 512 / 1024 / 2048 | 72 / 144 / 288 MiB | 可行 |
| 4096 | 576 MiB | 可行（需先停 Ollama） |
| 8192 | 1.13 GiB | 需停服务 + batch=1 |
| 16384 | 2.25 GiB | 需停服务 + 严格预算 |
| 32768 | 4.5 GiB | 与 4B 权重同机不可行 |
| 262144 | 36 GiB | 仅作设计上限 |

权重侧约束：Q8_0 全量反量化为 fp16 约 8.04 GB > 本机 7.8 GiB，因此**禁止**一次性反量化；
单层 attention 权重（≈52 MB fp16）可逐层流式处理，36 层全量约 1.9 GB。

## 4. 真实 shape 的 benchmark 档位

```text
prefill : B=1, Hq=32, Hkv=8, D=128, causal, S ∈ {512,1024,2048,4096}
decode  : B=1, Hq=32, Hkv=8, D=128, causal, S_q=1, S_kv ∈ {1024,4096,8192}
```

已落盘数据（`docs/results/benchmark_report.md` §4）：

| 场景 | 实现 | P50 | TFLOPS |
| --- | --- | --- | --- |
| S=2048 非 causal | flash | 4.32 ms | 15.9 |
| S=2048 非 causal | SDPA(flash) | 2.40 ms | 28.6 |
| S_q=1, S_kv=4096 | decode | 0.246 ms | 8.7 |

## 5. 量化边界（必须遵守）

```text
1) Q8_0 是权重量化，不是激活/KV 量化；本项目的注意力计算仍在 fp16/bf16 下进行。
2) Q8_0 反量化本身引入误差（block=32 + fp16 scale）；parity 的参考基准必须是
   「同一份反量化后的 fp16/bf16 张量」用 FP64 计算的结果，而不是原始 GGUF 字节流。
3) 禁止把 Q8_0 原始字节直接送入 fp16 kernel；禁止宣称「已支持 Q8_0 注意力」。
4) GPU 侧 dequant-fused attention 属于 Roadmap，不属于 v1 交付。
5) Ollama 的数字只能作为完整模型服务基线，不得与本项目 kernel 数据混为一谈。
```

## 6. 显存共存与实验顺序（8 GB 硬约束）

```text
1) SCA 侧实验前：确认 Ollama 未运行（curl -s http://127.0.0.1:11435/api/ps 应失败）
2) Ollama 基线采集：cd 模型目录 && ./serve.sh
3) 禁止两者同时压满显存后再声明数据有效
```

本次 benchmark 运行时显存占用 65 MiB（Ollama 未运行），结果 JSON 记录了 GPU 与显存状态。

## 7. 当前状态

| 项 | 状态 | 说明 |
| --- | --- | --- |
| GGUF 元数据解析 | 完成 | 与提示词 §23.1 完全一致（36/32/8/128/2560/5e6/262144/398 tensors） |
| KV 预算表 | 完成 | 147,456 B/token，与 §23.2 一致 |
| 真实 shape benchmark | 完成 | prefill S=2048 / decode S_kv=4096（见 §4） |
| 真实权重 Q/K/V parity | 未完成 | 需要逐层反量化 + RoPE 前向；`tools/dump_qwen3_qkv.py` 未实现，属 Roadmap；复现路径已在 F-005 记录 |
| Ollama 服务基线 | **已完成** | `tools/ollama_baseline.py` 实测：53.34 tok/s（128 token × 2 次），服务 VRAM 4952 MiB；落盘 `docs/results/ollama_baseline.json` |

## 8. 本地服务连通与基线（实测）

```bash
# 启动（模型库指向模型目录，端口默认 11435）
SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 \
SCA_OLLAMA_MODEL=qwen3:4b-thinking-2507-q8_0 bash scripts/serve_model.sh

# 连通性检查 / 基线采集
python tools/ollama_baseline.py --check
python tools/ollama_baseline.py --max-tokens 128 --runs 2 --json docs/results/ollama_baseline.json
```

| 指标 | 实测值 | 证据 |
| --- | --- | --- |
| 服务版本 / 模型 tag | Ollama 0.32.6 / `qwen3:4b-thinking-2507-q8_0` | `/api/version`、`/api/tags` |
| 首次加载 | ≈ 2.88 s | `load_duration` |
| prompt eval（21 token） | 38.1 ms | `prompt_eval_duration` |
| decode 吞吐 | 53.25 / 53.44 tok/s（两次） | `eval_count / eval_duration` |
| 服务 VRAM | 4,952 MiB（模型 4,731 MiB） | `nvidia-smi` + `/api/ps` |

**口径边界（再次强调）**：以上是完整模型服务指标；本项目 attention kernel 的实测见
`docs/results/benchmark_report.md`（flash 在 S=4096 非 causal 下 3.95 ms / 17.4 TFLOPS）。
两者不可混用，也不可相除得出“加速比”。

> 系统级 Ollama（`/usr/local/bin/ollama`，端口 11434）常见为 **CPU-only**（缺少 runner），
> 因此本文档统一使用用户级、带 CUDA runner 的安装，并通过 `SCA_OLLAMA_BIN` 指向它。
