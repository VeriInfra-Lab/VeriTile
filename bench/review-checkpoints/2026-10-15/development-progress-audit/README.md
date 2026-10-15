# 开发进度全量核对（2026-10-15）

**184 题中，183 题已有原源码对应的证明或合同；1 题仅有修复候选。58 是软件位值证明这一类的数量，不能作为总体完成题数，也不能将另外 126 题统称为未开发。**

随后进行了[本地提交清理](cleanup.md)：重复历史快照移入本地归档，当前源码、必要证据和实验工具保留，并在精简副本中重新验证。本页此前的构建和检查记录仍描述当时的实际执行。

| 当前已登记成果（各题自己的声明范围） | 题数 |
|---|---:|
| 软件位值证明 | 58 |
| 程序相对等价证明（含 3 题精确数值绑定待补） | 109 |
| 后端参数化值合同 | 16 |
| 仅修复候选：fp4_to_bf16_conversion | 1 |

当前共有 **893 条公开声明、3840 个注册模块**，其中 3409 个属于题目目录。178 题具有数学正确性声明；这一数量与上表类别重叠，不能相加。

原来重点修复的 41 题中，35 题达到软件位值证明层，6 题仍保留后端算子。这 6 题已有相应证明工作，并非空白任务。

[184 题逐项表](tasks-184.tsv)列出每题的数学、相对等价、抽象舍入、后端值及位值声明数量，原接受范围、dtype、剩余范围和源码入口。[机器可读核对结果](summary.json)保留实际命令、退出码及源文件哈希。

## 本次实际检查

| 检查 | 实际结果 |
|---|---|
| [baseline-prepare](baseline-prepare.log) | 通过 |
| [catalog-integrity](catalog-integrity.log) | 通过 |
| [catalog-reproduction-host](catalog-reproduction-host.log) | 通过 |
| [catalog-reproduction](catalog-reproduction.log) | 未通过（退出码 -15）；保留原始日志 |
| [full-build-host](full-build-host.log) | 通过 |
| [full-build](full-build.log) | 沙箱中工具链定位失败；主机构建单独记录 |
| [legacy-manifest-status](legacy-manifest-status.log) | 未通过（退出码 1）；保留原始日志 |
| [metadata-regressions](metadata-regressions.log) | 通过 |
| [numerical-queue](numerical-queue.log) | 通过 |
| [source-regressions-fixed-rerun](source-regressions-fixed-rerun.log) | 未通过（退出码 1）；保留原始日志 |
| [source-regressions-host](source-regressions-host.log) | 未通过（退出码 -15）；保留原始日志 |
| [source-regressions-slowfs-rerun](source-regressions-slowfs-rerun.log) | 通过 |
| [source-regressions](source-regressions.log) | 未通过（退出码 1）；保留原始日志 |
| [source-replay-host](source-replay-host.log) | 通过 |
| [source-replay](source-replay.log) | 未通过（退出码 -15）；保留原始日志 |
| [value-publication-inputs](value-publication-inputs.log) | 通过 |
| [value-registry](value-registry.log) | 通过 |

核对输入 4154 个；构建与检查期间检测到的源码变化：0 个。Lake 增量构建复验使用 `--rehash`；成功时表示所有请求目标通过，不表示每个目标都从零重新编译。

本次修复了 dtype 审计硬编码扫描整个仓库的问题：现在遵守 `PORTS_ROOT`，隔离测试不再受其他题目辅助文件影响。首轮 340 项回归有 4 个相关断言失败及 8 个子进程超时；修复后的原时限重跑只剩超时。最终针对这 12 项使用 300 秒子进程时限，未修改输入、断言或源代码判定标准。原始失败记录与最终重跑分别保留。可在仓库根目录运行 `python3 -B bench/review-checkpoints/2026-10-15/development-progress-audit/reproduce_slowfs_regressions.py`。

## 此前的证明没有漏掉

`ksoftmax_triton` 的 backward 检查点已证明完整数学 IO 和 typed 浮点等价，覆盖 216 种独立存储类型组合及全部相关开关。后续 forward 证明已经接入；当前公开入口为 4 条数学正确性声明、2 条浮点等价声明。它属于上表的程序相对等价类，不能因没有计入 58 而算作没有做完证明开发。

最新 Retention DQ 和 DQ/barrier/DK/DV 组合又增加 2 条公开合同、6 个模块；相应源执行和内存保持证明已经登记。它们保留数值后端参数，不增加软件位值闭合题数。

## 尚未关闭的事项

- 原 41 题余下的 6 项是 logsumexp_fwd、mixed_sparse_attention、parallel_retention_attention、rbe_triton_transform、token_softmax_bloom、token_softmax_llama；主要涉及 exp/log/pow/sin/cos、dot 图和归约顺序的具体对应。
- 5 个精确数值绑定 profile 仍待补：typed fp32 乘法及 BGMV 的 4 个加法 profile。它们影响 3 题；已被反例推翻的全域 round=id 另列，不能靠实验变成真命题。
- 原始 scaled FP4 程序仍存在编译问题，修复候选不等于原始程序已经通过。
- 旧 CI 入口仍需接入当前 catalog/验收适配器。本次实际运行旧 manifest 检查，在 destindex_copy/RawBitCorrect.lean 处报 Unexpected split-port entries；旧全目录 dtype 扫描也会扫到新 typed helper。当前目录/显式源码适配器通过，不等于现有整条 CI 已通过。
- 官方 comparator 已有冻结 F3 library 的 118 目标／8 模块通过记录；保存的 corpus 维护快照为 9/365 文件、13/872 声明，不是实时全量通过结果。其后 21 条新声明不在该批次内。
- GPU/native 完整正确性没有由本次检查建立；77 题已有具体数值实验配方，37 项扩展实验工作和 Retention 的 7 项具体义务保留在统一待办中。

源码、数学模型、原始/候选等价、具体软件位值、独立重放和 GPU 对应是不同进度轴。183 表示已有原源码合同，58 表示其中的软件位值类别；二者都不是最终端到端验收题数。旧 README 的 173/178 与历史验收账本的 59/125 仅描述旧快照。

本次完成全量目录、现有逐题范围记录、源码哈希和机械检查的核对；没有把它称作重新独立审阅所有定理语义。

## 推送内容的独立目录核对

从 Git 暂存区导出完整待提交文件后，在该目录中重新执行 catalog 完整性、值证明登记和数值待办检查，三项均通过。4154 个已审计输入的 SHA-256 全部一致；指定的 backward 检查点 35 个文件全部保留且内容一致。上述检查不依赖原工作区未提交的文件。详见[发布核对记录](publication-validation.json)。源码没有变化，因此未重复执行 Lean 全量构建。

暂存源码的空白检查另记录了已有行尾空格及文件末尾空行；为保持已核对的源码/证据哈希，本次原样保留，未声称 `git diff --check` 全部通过。
