# 当前增量验收：int4_matmul 完整软件位语义

当前 **184 项：55 项软件位值证明、128 项依赖后端、1 项修复候选**。最初那 41 项为 **34／7**。本次仅 int4_matmul 升级，见[184 项表](tasks-184.tsv)及[剩余 7 项清单](remaining-original41.tsv)。这里的“软件”指具体舍入和字表示已有完整合同，仍不代表 GPU 已验收。

[证明包](../int4-source-dotacc-software-closure/README.md)新增两条公开入口：原源码的独立 dot 后加 carry，以及观察到的 live-carry 编译图。两者保持原 SPLIT_K=1、完整 K-block 循环、packed 输入和原地址条件；没有新增有限数、单块或输出值前提。初态内存参考、成功执行、完整内存和 frame 均保留。原辅助定理保持原样，新增 facade 将它正式纳入主源码合同；compiled 入口单独登记。

**9 个新增模块独立复编、32 条具名公理检查通过；39 项元数据测试及独立重生成通过。** 当前 886 条公开声明、3,811 个注册模块；构建证据由已核验的 3,802 条旧记录与 9 条新实际编译组成，不声称新跑了一遍全库。16 个安装文件与输入清单、ZIP 成员字节匹配，ZIP 时间戳差异单列。见[生成比较](independent-generation/comparison.json)、[组合证据](incremental-build/receipt.json)、[安装后检查](actual-workspace-gates/receipt.json)、[根审查](root-review.json)。

原源码 half 输出 1、正序 carry-FMA 输出 0、逆序输出 2 的同输入完整循环反例说明两图不能合并。FMAPlan 是每个 K lane 恰出现一次的排列数据，不是假设硬件 MMA 可以串行化。[GPU 采集包](../int4-native-characterization/README.md)已独立通过 26 项工具测试和 4 个离线编译配置，实际 GPU 状态为 NOT_RUN。

所有 191 份 Python 输入、旧 884 条公开记录、35 条扩展任务和 58 个证据包逐项保留；新增证据包后索引为 59，包数不是任务完成数。[证据索引](evidence-package-index.json)、[复现命令](../../../current_source_acceptance.md)、[上一版冻结验收](../masked-select-software-metadata/README.md)。官方完整比较器继续检查冻结 F3 的 365 文件／872 条声明；本版后来新增的 14 条声明另行记录，不包含在 F3 批次中。

剩余 7 项为 matmul_dequantize_int4、token_softmax_bloom、token_softmax_llama、logsumexp_fwd、rbe_triton_transform、parallel_retention_attention、mixed_sparse_attention。尚未安装的候选证明不提前计数。机器地址、字节 ABI、MMA/SFU 语义与设备运行仍分别待验证；已被反例推翻的普遍等式不交给实验批准。
