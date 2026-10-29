# 软件位语义补充验收：2026-10-15

本包审查当前工作区的新增证明、登记和逐项统计。**这些增量检查通过，不代表 184 项硬件正确性已全部验收。** [逐项表](tasks-184.tsv)覆盖 184 个任务；[复现命令](../../../current_source_acceptance.md)使用本包的新元数据基线。

| 当前声明范围 | 任务数 |
|---|---:|
| 具体软件浮点位语义 | 51 |
| 依赖后端算子 | 132 |
| 仅修复候选 | 1 |

最初 41 项仍为 **30/11**。本次只将 `layer_norm_ops` 原接受的 fp32 九阶段升级为具体软件位语义：维度 N 经精确有理数转换再舍入到 fp32，算术和存储采用具体位运算。未新增数值回调、数值输入限制或输出相等前提；未扩大为完整 forward producer、全部调用 dtype 或多行 backward。见[独立语义复核](independent-semantic-review/receipt.json)。

本次增加 9 条公开声明，总数 **881 = 原 747 + 新 134**；新 134 条包括 130 条数值声明、2 条内存帧声明、1 条恒等式和 1 条协议声明。原 41 项的 118 条声明已包含在原 747 中。另新增的 FP4/int4 辅助定理和固定见证不计入公开声明。

| 实际检查 | 结果与证据 |
|---|---|
| 元数据生成与独立复现 | 57 项测试通过；root 与候选生成的安装字节一致。[生成回执](generation-receipt.json)、[测试日志](root-tests.log)。 |
| 工作区与登记 | 实际目录清单和重复检查通过，全部 191 个 Python 输入保留，3,792 个模块全部登记。[安装后检查](installed-check/receipt.json)。 |
| 历史源码对照 | 当前清单下 178 个历史视图、23 段原检查通过，保留完整实际日志。[源码回执](installed-check/legacy-source/receipt.json)。 |
| 编译证据 | 重新核验原 3,783 个模块，组合 9 个新模块的实际独立编译，覆盖 3,792 个模块；没有声称本轮全量重编。[组合回执](incremental-build/actual/receipt.json)、[root 独立 API 复验](incremental-build/root-review/receipt.json)。 |
| 公理检查 | 原 872 条公开声明继续引用旧 v5；新增 39 个命名检查均通过，含全部 9 条新公开声明。[逐模块编译和公理证据](incremental-build/actual/new-module-evidence.json)。 |
| 原声明与分类 | 原 747 保留；184 项仅 LayerNorm 一行改变，原 41 分类不变。[逐项表生成检查](table-generation.json)。 |

本轮其他结果：

- [scaled FP4 算法与整数格式 oracle](../../2026-10-14/next-value-repair/fp4-scaled-algorithm-oracle/README.md)覆盖 256 个 scale 指数及 4,096 个 nibble/scale 组合，并保留 FTZ 等反例。原始 `pow(fp32,int16)` 编译失败仍在，显式 int32 修复仍属候选。
- [int4 完整 packed tile 反例](../../2026-10-14/next-value-repair/int4-dotacc-gap/README.md)反驳独立 dot 后相加与 carry-seeded FMA 图的普适等价；两个软件图分别得到 half `0x3c00` 和 `0x0000`。这没有证明 native WGMMA 的输出。int4 保持后端分类，不能把已反驳的等式列为等待实验通过的假设。
- [无缩放 FP4 工具 v2](../../2026-10-14/next-value-repair/fp4-unscaled-experiment-runner-v2/README.md)已交付具体 N21 配置：26 项测试、2 个 SM90 离线编译通过；GPU 为 `NOT_RUN`、0 次启动。仍缺物理别名和内部操作数等覆盖，未将 N21 或 37 个完整工作项标为完成。
- [官方 comparator 虚拟机预检](../../2026-10-14/next-value-repair/comparator-vm-preflight/README.md)及[源文件写入拒绝控制](../../2026-10-14/next-value-repair/comparator-vm-source-write-control/README.md)通过。正式项目 replay 正在单独运行；本包没有完整 library/corpus 通过结论，更没有用旧 872 条范围代替新 881 条范围。

尚未验收的项目包括原始 scaled FP4 源码合同、全量正式 comparator，以及软件/后端语义与实际 GPU 执行的对应。有限数值实验可以反驳关系或支持记录的配置，不构成通用硬件正确性证明。旧“129 项无额外数值律”和“43 项已有实验”是历史义务轴，不能据此跳过这些绑定。

[元数据安装清单](installation.json)记录修改前后哈希；[目录清单](current-source-catalog.json)、[补充声明登记](catalog-supplement.json)和[新基线](metadata-baseline/receipt.json)相互绑定。[旧统一验收包](../../2026-10-14/next-value-repair/final-acceptance/SUMMARY.zh-CN.md)保持冻结。原 `delivery-index.json` 新增一项已复核源绑定后即固定，其他增量通过独立的 [supplements-index.json](../../2026-10-14/next-value-repair/supplements-index.json)索引。
