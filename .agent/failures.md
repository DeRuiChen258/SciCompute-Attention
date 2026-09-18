# 失败尝试与解决路径

> 规则：禁止删除历史记录；每条记录写清「现象 / 已尝试 / 结论 / 下一步」。

### F-001（Phase 0）CMake 版本与提示词基线不一致
- 现象：提示词 §1.1 记录 CMake 4.4.0-rc1，实测 `cmake --version` = 3.31.6。
- 已尝试：`which -a cmake` 确认无多版本共存。
- 结论：非阻塞差异；本项目 `cmake_minimum_required(VERSION 3.24)` 在 3.31.6 下满足。
- 下一步：在 `docs/env_report.md` 的「与提示词 §1 的差异」中登记。

