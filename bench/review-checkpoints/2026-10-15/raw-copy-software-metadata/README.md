# 当前增量验收：两个完整搬运内核的位值合同

当前 **184 项：58 项软件位值证明、125 项依赖后端、1 项修复候选**。最初那 41 项保持 **35／6**。本次只升级 destindex_copy 和 matrix_transpose；两项均不属于原 41 项。见[184 项表](tasks-184.tsv)和[剩余 6 项](remaining-original41.tsv)。软件位合同不等于 GPU 验收。

[destindex_copy 证明](../destindex-copy-raw-bit-closure/README.md)覆盖九组独立 fp32/fp16/bf16 存储组合、所有原参数、signed Int 目标索引、完整无 mask padding、两个初态 tile 读取、顺序双 store、别名和完整内存/frame。fp32 使用明确的 raw BitVec32 载荷；`.real` 存储标签不是实数算术。输出逐点结论仅保留旧单射和输出区域分离前提。

[matrix_transpose 证明](../matrix-transpose-raw-bit-closure/README.md)直接执行原 fp16 load/transpose/store，同时覆盖既有索引交换候选。任意原尺寸与四个步长、零尺寸、初态和输入输出别名均保留；每个输出按初态定义，NaN payload、负零和其他全部 16 位载荷原样保存。唯一前提为原输出地址单射。不可达浮点回调由参数化 backend-independence 定理消除，不假设浮点转换恒等。

**8 个新增模块独立复编、44 条具名标准公理检查、52 项元数据测试通过。** 当前 **891 条公开声明、3,834 个注册模块**；构建证据为 3,826 条原记录重新核验加 8 个新实际编译，不称整库重新编译。17 个元数据安装目标和 4,187 个生成器输入已绑定，含全部 24 个历史任务文件的前后检查。见[组合回执](incremental-build/receipt.json)、[独立生成比较](independent-generation/comparison.json)、[安装后检查](actual-workspace-gates/receipt.json)与[根审查](root-review.json)。

191 份 Python、旧 889 条公开记录、37 条扩展任务和 60 个证据包保持不变；现为 39 个扩展、[62 个证据包](evidence-package-index.json)。包数不是完成任务数。**891 = 747 历史 + 144 新声明；144 = 140 值合同 + 2 frame + 1 恒等式 + 1 协议。** 固定 witness 不计为参数化完整合同。

原单程序 Nat/Int 地址模型和软件 typed-memory 仍需与实际固定宽度地址、物理 byte ABI、合法分配/grid、线程布局与别名执行顺序对应；本次没有 GPU 执行。[Retention v2采集包](../parallel-retention-native-characterization-v2/README.md)已发布，GPU仍为NOT_RUN、0次launch，不增加任务或公开声明数。原剩余六项仍依赖后端。剩余表只纠正 Llama 的范围说明为原 ∀FloatDType 完整 row，fp32 单列为实验默认配置；它不改变分类或旧证明范围，P6 冻结表不改。

[上版 dequant 验收](../dequant-int4-source-compiled-software-metadata/README.md)及[任务说明](../../../tritonbench_g/matmul_dequantize_int4/SOFTWARE-ACCEPTANCE.md)保留；[dequant 采集工具](../dequant-int4-native-characterization/README.md)已交付，实际 GPU 仍为 NOT_RUN。[两个任务的新说明](../../../tritonbench_g/destindex_copy/SOFTWARE-ACCEPTANCE.md)、[矩阵说明](../../../tritonbench_g/matrix_transpose/SOFTWARE-ACCEPTANCE.md)另写，原 README 不改。

[官方 library 结果](../comparator-canonical-library/README.md)、[样例](../comparator-initial-samples/README.md)与[存储维护快照](../comparator-storage-maintenance/README.md)各保留自身范围。全量批次目标固定为 **F3 的 365 文件／872 声明**，这里不声称全量已通过；P7 相对 F3 新增的 **19 条声明**单列。本页不把历史快照当作实时进度。[复现命令](../../../current_source_acceptance.md)、[当天索引](../README.md)。
