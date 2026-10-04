# 补充 RTX 5090 原生构建资格

2026-10-05。此窗口用于发现原生编译问题。Thor / RTX 5080 的单卡 E2E、高 batch / 高请求并发、模型正确性与速度矩阵仍未运行。

| 尝试 | 固定源码 | 自然 SSH / make exit | GPU 测试进程 | 实际失败与修复 |
| --- | --- | --- | ---: | --- |
| r1 | `cf70e266` | 1 / 2 | 0 | NVCC 拒绝普通 `.so.1` 输入；改用 `-Xlinker=` 链接已有运行库 |
| r2 | `cf70e266` | 1 / 2 | 0 | 官方 NCCL 的可选 GDAKI 缺少 DOCA 开发头；单卡测试统一使用 `NCCL_GIN_GDAKI_ENABLE=0` |
| r3 | `cf70e266` | 1 / 2 | 0 | fixture 常量与 transport 的 `kIterations` 宏冲突；源码 `cc203297` 将常量改名为 `kSignalTestGenerations` |

完整记录：[attempts.json](attempts.json)。原始构建输出：[r1](r1-native-build.log)、[r2](r2-native-build.log)、[r3](r3-native-build.log)。实际命令和进程终态：[r1](r1-commands.jsonl)、[r2](r2-commands.jsonl)、[r3](r3-commands.jsonl)。没有 GPU 正确性、模型或速度通过记录；未生成加速比。

三次失败均在归还窗口前收集。独立交接检查核对同一 boot ID、旧控制器 PID 已消失，并同时取得 GPU0 / heavy-IO 锁后立即释放。其他任务的模型、环境和进程没有修改。

下一轮固定源码为 `cc203297a3c5cc0ed2666cb85fa940127b57d5bb`，源码包 SHA256 为 `df158569251ba19f675113a1dd1897144a7969395d2a2af5984ae5b597b086b3`。需要新的有限窗口，旧 grant 不可复用。原生 HT 数值 gate 执行真实 scan / dispatch / combine；完整 Granite 模型 gate 单独验证全部 24 层的 GIN 输入与输出，两者分别保留验收证据。
