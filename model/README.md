# Model Card — Qwen3-4B-Thinking-2507-Q8 (GGUF Q8_0)

> 这个目录是**模型卡 + 复现脚本**，**不包含权重**：权重体积 4.28 GB，超过 GitHub 单文件 100 MiB 的
> 硬限制（见 §4）。权重通过 `download_model.sh` 从官方源获取，并用下表 sha256 校验。

## 1. 模型事实（本机实测解析 GGUF 得到）

| 项 | 值 |
| --- | --- |
| 基础模型 | Qwen3-4B-Thinking-2507（Apache-2.0） |
| 权重文件 | `Qwen3-4B-Thinking-2507-Q8_0.gguf`，4,280,404,960 B（3.99 GiB） |
| sha256 | `012aa2736c32b7b74c3ca7b2da181b9e1d24a3973abf5510dee5590b27445440` |
| 容器 / 量化 | GGUF v3，file_type 7（Q8_0：block = 2 B fp16 scale + 32 × int8） |
| 张量数 / 参数量 | 398 / 4,022,468,096 |
| 架构 | `qwen3`：36 层，H_q=32，H_kv=8（GQA group 4），D=128，hidden 2560，context 262144 |
| RoPE / RMS eps | 5e6 / 1e-6 |
| 推理服务 | Ollama（tag `qwen3:4b-thinking-2507-q8_0`），本机实测 100% GPU，VRAM ≈ 4.7 GiB |

## 2. 获取权重

```bash
# 方式 A（推荐）：ollama 直接拉取到本目录的独立模型库
SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 bash model/download_model.sh

# 方式 B：从 HuggingFace 拿 GGUF（Qwen 官方仓库），再用 sha256 校验
#   https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507-GGUF
sha256sum Qwen3-4B-Thinking-2507-Q8_0.gguf
#   期望：012aa2736c32b7b74c3ca7b2da181b9e1d24a3973abf5510dee5590b27445440
```

## 3. 启动服务并验证连通

```bash
# 启动一个带 CUDA runner 的 Ollama，模型库指向本目录（默认 127.0.0.1:11435）
SCA_MODEL_DIR=/path/to/Qwen3-4B-Thinking-2507-Q8 \
SCA_OLLAMA_MODEL=qwen3:4b-thinking-2507-q8_0 python tools/ollama_baseline.py --check

# 采集基线（TTFT / tokens/s / VRAM），结果落盘 JSON
python tools/ollama_baseline.py --json docs/results/ollama_baseline.json
```

本机实测基线（RTX 5070 Laptop, 8 GB, context 4096）：

| 指标 | 值 |
| --- | --- |
| 加载时间（首次） | ≈ 2.9 s |
| prompt eval（21 token） | ≈ 38 ms |
| decode 速度 | **≈ 53–55 tok/s** |
| 服务 VRAM | 4731 MiB（`/api/ps` 的 `size_vram`） |

> 口径（重要）：这些数字是**完整模型服务**（分词 + 权重 + 采样 + KV Cache）的指标，
> 与本项目 attention kernel 的 benchmark 不是同一件事，禁止混在同一张表里比较。

## 4. 为什么权重没有进 Git

| 限制 | 数值 | 结论 |
| --- | --- | --- |
| GitHub 单文件硬限制（普通 git） | 100 MiB | 4.28 GB 文件直接拒绝 |
| GitHub Release 单资产上限（Free） | 2 GiB | 仍然超限 |
| Git LFS 免费配额 | 1 GB 存储 / 1 GB 月流量 | 单个 4.28 GB 对象即超配额 |
| 仓库体积建议上限 | ≤ 1 GB（超出会收到警告/被限制） | 即使切成 <100 MB 分片也会严重超标 |

因此本目录采用「模型卡 + 下载脚本 + sha256 校验」的方式交付。若你拥有 Git LFS 数据包（data pack）
或 GitHub Pro/Team，执行：

```bash
git lfs install
git lfs track "model/*.gguf"          # 写入 .gitattributes
cp /path/to/Qwen3-4B-Thinking-2507-Q8_0.gguf model/
git add .gitattributes model/Qwen3-4B-Thinking-2507-Q8_0.gguf
git commit -m "chore: add Qwen3-4B Q8_0 weights via Git LFS"
git push
```

## 5. 许可证

权重与基础模型遵循 **Apache-2.0**（GGUF 元数据 `general.license`）；本目录的模型卡与脚本同样以
Apache-2.0 提供。商用与再分发请遵守上游许可证与 Qwen 的模型使用条款。

