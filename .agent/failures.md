# 失败尝试与解决路径

> 规则：禁止删除历史记录；每条记录写清「现象 / 已尝试 / 结论 / 下一步」。

### F-001（Phase 0）CMake 版本与提示词基线不一致
- 现象：提示词 §1.1 记录 CMake 4.4.0-rc1，实测 `cmake --version` = 3.31.6。
- 已尝试：`which -a cmake` 确认无多版本共存。
- 结论：非阻塞差异；本项目 `cmake_minimum_required(VERSION 3.24)` 在 3.31.6 下满足。
- 下一步：在 `docs/env_report.md` 的「与提示词 §1 的差异」中登记。

### F-002（Phase 2）naive 输出全 0：块级归约没有广播
- 现象：S=1、Q=K=1 的最小用例输出全 0（期望等于 V）；`scores` 缓冲读回全是 0。
- 定位：`BlockReduceMax` 只在 warp 0 内做第二级归约，其它 warp 仍持有各自的部分值；softmax 里
  `all_masked = !IsFinite(m)` 对非 0 号 warp 恒为真，最终 `inv_sum = 0` 把整行乘成 0。
- 修复：第二级归约后把结果写回 `scratch[0]` 并让所有线程读取（`src/cuda_common/numerics.cuh`），
  同时在注释里写清「块级归约必须广播」的约束。
- 教训：块级归约的正确判据不是"warp0 拿到了正确值"，而是"每个线程都拿到同一个值"。

### F-003（Phase 5）flash 的 PV 结果错位：ldmatrix.x4.trans 寄存器→片段映射推导错误
- 现象：均匀注意力用例中**奇数编号的 8 列输出块**全为 0（列 8-15、24-31、…），偶数块正确。
- 定位：我按「矩阵 0/1 → n-tile 0，矩阵 2/3 → n-tile 1」推导 B 片段配对，实际 `.trans` 的语义
  是 `reg0=(M[2c][g],M[2c+1][g])`、`reg1=(M[8+2c][g],M[9+2c][g])`、`reg2/reg3` 为 +8 列块，
  即 (b0,b1) 是 (reg0,reg1) 与 (reg2,reg3)。原实现把 `{vb[0],vb[2]}` 当作 b0/b1，导致奇数 n-tile
  使用了错行数据；在该用例中 V 的第 8-15 行是零填充，于是表现为「整块为 0」而不是"数值略偏"。
- 修复：配对改为 `{vb[0],vb[1]}` 与 `{vb[2],vb[3]}`。
- 教训：**指令级语义必须实测**（`/tmp/sca_dbg/ldtrans.cu` 一类的 walk-through 探针），
  推导只能用来缩小验证范围，不能作为实现依据。该结论已写回 `src/backends/flash/flash_fwd_impl.cuh`
  与 `docs/flash_attention.md` §3。

### F-004（Phase 2）benchmark 进程退出时 SIGSEGV
- 现象：`benchmark_naive` 结果正确但进程以 139 退出，gdb 栈指向
  `sci::BufferHandle::deallocate()` ← 静态 `unordered_map` 析构。
- 定位：scratch pool 的静态缓存持有 `sci::Tensor`，静态析构发生在 CUDA runtime 拆除之后，
  `cudaFree` 访问已失效的上下文。
- 修复：缓存改为进程级故意泄漏（`new std::unordered_map`），`ReleaseScratch()` 仍可显式释放。
- 教训：持有设备内存的静态对象不要依赖析构顺序，必须在退出前显式释放或有意泄漏。
