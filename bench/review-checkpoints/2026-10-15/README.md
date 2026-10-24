# 2026-10-15 验收入口

当前安装：**184 项，58 项软件位值／125 项依赖后端／1 项修复候选**；原41项 **35／6**。共 **893 条公开声明、3,840 个注册模块**。软件证明、正式comparator与GPU结果分别记录，完整硬件验收未完成。

- [Retention DQ and backward helper proofs](retention-query-source-proof/README.md): two additional backend contracts, six modules and 20 named axiom checks; numerical work is deferred in the [shared queue](../../tritonbench_g/deferred_numerical_experiments.md).
- [P7验收及逐项表](raw-copy-software-metadata/README.md)：只新增两项原完整域位值合同；原剩余六项分类不变。
- [destindex_copy](destindex-copy-raw-bit-closure/README.md)与[matrix_transpose](matrix-transpose-raw-bit-closure/README.md)：共8模块、44条具名检查、2条公开合同。
- [上版dequant验收](dequant-int4-source-compiled-software-metadata/README.md)、[完整证明](dequant-int4-source-compiled-software-closure/README.md)与[采集工具](dequant-int4-native-characterization/README.md)：原source与compiled图分别记录；GPU未运行。
- [Bloom](token-softmax-bloom-exp-lowering-gap/README.md)与[Llama](token-softmax-llama-exp-lowering-gap/README.md)：内部指数反例和来源桥；不推断最终softmax输出反例，不增加完成数。
- [LSE图与实验交接](logsumexp-exp-fusion-gap/README.md)：保留原完整范围；exp/log/标量转换仍开放，GPU未运行。
- [RBE原核采集v2](rbe-original-characterization-v2/README.md)：修复环境/架构记录检查，保留原pow/sin/cos及双mask；没有新增公开合同或GPU数值通过。
- [Retention原核采集v2](parallel-retention-native-characterization-v2/README.md)：24个离线配置与工具验收通过；GPU为NOT_RUN、0次launch，不新增完成任务或公开声明。
- [int4验收](int4-source-dotacc-software-metadata/README.md)与[采集工具](int4-native-characterization/README.md)：不同软件图与native顺序边界分别说明。
- [masked_select](masked-select-software-metadata/README.md)、[KV-cache](kv-cache-software-metadata/README.md)、[K-cache](kcache-software-metadata/README.md)：先前冻结验收。
- [官方canonical library](comparator-canonical-library/README.md)、[样例](comparator-initial-samples/README.md)和[存储维护](comparator-storage-maintenance/README.md)：仅引用各自冻结结果/进度快照，不表示365文件全量通过。完整批次目标是F3的365文件／872声明，后续新增21声明另列。
- [反量化早期离线图](dequant-int4-compiled-graph/README.md)、[原41历史缺口](original41-remaining-analysis/README.md)与[早期软件验收](software-closure-metadata/README.md)保留历史范围，当前分类以P7表为准。

旧“129无额外数值律”和“43实验记录”不代表硬件完整正确。[复现指南](../../current_source_acceptance.md)。
