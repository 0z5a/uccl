# 补充 RTX 5090 原生资格

2026-10-05。r5 在固定源码 `3e9ea407bae317abec1c2e36f0ac6281ec23b344` 上自然退出 0，完成实际 vendored scan / dispatch / combine、production FIFO signal 和 tensor 正确性资格。机器为 westd RTX5090 GPU0 / SM120，CUDA 13.0，现有 Torch 2.12.1+cu130 环境只读。

Thor / RTX 5080 的单卡 E2E、高 batch / 高请求并发、完整模型正确性与速度矩阵仍未运行。原生 HT 的单 rank 路径命令数为 0；这些结果不提供 GIN 加速比、NIC 或完整 Granite 推理证据。

| r5 case | B | 实际 pending C | Tokens / batch | 每 backend 轮数 | 全值 oracle / 两 backend hash |
| --- | ---: | ---: | ---: | ---: | --- |
| 常规 | 8 | 32 | 1024 | 7 | 通过 / 一致 |
| 常规 | 16 | 64 | 2048 | 7 | 通过 / 一致 |
| 常规 | 32 | 128 | 4096 | 7 | 通过 / 一致 |
| 常规 | 64 | 128 | 8192 | 7 | 通过 / 一致 |
| 最后请求 tail4 | 8 | 32 | 1020 | 7 | 通过 / 一致 |

两 backend 共 10 个独立 HT 测试进程、70 轮；每进程前2轮标记 warmup。每轮实际排队 C 个请求、按 B 完成，完整 routing/count/BF16/probability/flag oracle 通过。另有 24 native signal cases、7 不均匀 tensor cases、队列 gate 通过，14 个测试进程均自然退出 0。全部六个二进制/库 hash 已保存，CTA signal 按预期编译拒绝。

原始证据：[独立汇总](r5/validation-summary.json)、[HT/GIN 测试命令与终态](r5/ht-test/commands.jsonl)、[构建命令与终态](r5/ht-build/commands.jsonl)、[实际构建输出](r5/ht-build/native-build.log)、[CTA 拒绝输出](r5/ht-build/unsupported_signal_coop.log)、[二进制 hash](r5/ht-build/binary-hashes.json)、[机器准入记录](r5/ht-test/admission.json)、[signal](r5/ht-test/adapter-signal.log)、[tensor](r5/ht-test/native-tensor.log)。各 case 原始7轮在 [r5/ht-test](r5/ht-test)。保留了容器不允许 NUMA mempolicy 的原始警告；没有修改运行环境。

控制器 PID7488、SSH 和 make 均实际自然退出 0。[23,229-byte 收集收据](r5/receipt.json)的 archive SHA256 为 `95bf71197bbf40057e37af4ea8db10d20a5855d7ce16f923dd90cc1177904d88`；先收集后[独立交接](r5/handoff.json)，核对同 boot/PID消失并同时取得 GPU0 / heavy-IO 锁后释放。首次交接探测非零，后续实际成功，记录在 [inspection](r5/initial-handoff-inspection.json)。本任务在该机没有模型权重。

## 已保留的构建失败

| 尝试 | 固定源码 | 自然 SSH / make exit | GPU 测试进程 | 实际失败与修复 |
| --- | --- | --- | ---: | --- |
| r1 | `cf70e266` | 1 / 2 | 0 | NVCC 拒绝普通 `.so.1` 输入；改用 `-Xlinker=` 链接已有运行库 |
| r2 | `cf70e266` | 1 / 2 | 0 | 官方 NCCL 的可选 GDAKI 缺少 DOCA 开发头；单卡测试统一使用 `NCCL_GIN_GDAKI_ENABLE=0` |
| r3 | `cf70e266` | 1 / 2 | 0 | fixture 常量与 transport 的 `kIterations` 宏冲突；源码 `cc203297` 将常量改名为 `kSignalTestGenerations` |
| r4 | `cc203297` | 1 / 2 | 0 | 原生 HT TU 找不到 vendored `common.hpp`；测试 HTFLAGS 补上 repo root 与 `nccl-ep/include` |

四次失败记录：[attempts.json](attempts.json)。原始构建输出：[r1](r1-native-build.log)、[r2](r2-native-build.log)、[r3](r3-native-build.log)、[r4](r4-native-build.log)。实际命令和进程终态：[r1](r1-commands.jsonl)、[r2](r2-commands.jsonl)、[r3](r3-commands.jsonl)、[r4](r4-commands.jsonl)。这四次尝试没有启动 GPU 测试。

四次失败均在归还窗口前收集。独立交接检查核对同一 boot ID、旧控制器 PID 已消失，并同时取得 GPU0 / heavy-IO 锁后立即释放。其他任务的模型、环境和进程没有修改。

r4 固定源码为 `cc203297a3c5cc0ed2666cb85fa940127b57d5bb`，源码包 SHA256 为 `df158569251ba19f675113a1dd1897144a7969395d2a2af5984ae5b597b086b3`。r5 使用独立新窗口和 `3e9ea407` / archive `4e47235cbd079356c56af8da50ce74006cbf9faecc1f21815df96e171b756f5e`，完成 include 修复的原生验证。原生 HT 数值 gate 与完整 Granite 模型 gate 分别保留验收证据；已归还的 grant 不可复用。
