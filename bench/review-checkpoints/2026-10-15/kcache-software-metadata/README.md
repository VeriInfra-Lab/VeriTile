# 当前增量验收：kcache 原始位存储闭合

当前 184 项为 **52 项软件位值证明、131 项依赖后端、1 项修复候选**。原来那 41 项中为 **31 项软件位值证明、10 项依赖后端**。[逐项验收表](tasks-184.tsv)仅 `kcache_copy_triton` 一行发生本次升级。

新增 [kcache 五格式证明](../kcache-raw-bit-closure/README.md)保留原 n_tokens=1/splitX 完整参数范围，证明实际源执行成功，整个最终内存等于独立初态散射的编码，并保留 NaN 载荷、正负零、别名、零长度和完整 frame。旧任意载荷合同未改，历史 fp64 注解不冒称 native64。

**4 个新增模块独立编译通过，29 条具名公理检查通过；新增公开 specification 只有 1 条。** 当前共 882 条公开声明、3,796 个注册模块。构建证据组合为复验先前 3,783+9 记录，再加入本次 4 个实际编译记录；没有声称重新全量构建。

元数据生成器的 31 项测试已独立重跑。独立生成的 11 个安装文件、输入清单和归档成员字节一致；ZIP 时间戳使两个归档及快照回执哈希不同，已在[复现比较](independent-generation/comparison.json)明确记录。安装前遗漏的两个生成文件 before 哈希由上版冻结快照补验，首次守卫停止记录保留于 `installation-history`。

安装后实际工作区检查、最终源码漂移与逐项统计见[根代理回执](root-review.json)。[构建记录](incremental-build/receipt.json)、[表格生成](table-generation.json)、[新证据索引](evidence-package-index.json)、[具体复现命令](../../../current_source_acceptance.md)可单独核对。现有递归 Lake 登记覆盖新增文件，lakefile 未改。

[正式 comparator 首批检查](../comparator-initial-samples/README.md)已通过 canonical library、VectorAdd 和 F3 任务集内 1 条 FP4 声明；完整任务 corpus 及后来新增声明仍待检查。本包所有软件/backend 统计均不表示 GPU 正确性。原始 scaled FP4 缺口、机器地址和算子到硬件的绑定继续保留。

[上版 881 声明验收](../software-closure-metadata/README.md)、旧 41 项 registry、旧索引和旧回执均保持原字节。[原 41 项剩余缺口分析](../original41-remaining-analysis/README.md)按其生成时的 30/11 历史状态冻结，本包另行闭合其中 kcache 的软件表示缺口。
