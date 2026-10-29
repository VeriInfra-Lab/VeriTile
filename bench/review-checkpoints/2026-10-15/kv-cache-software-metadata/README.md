# 当前增量验收：KV cache 原始位存储闭合

当前 **184 项：53 项软件位值证明、130 项依赖后端、1 项修复候选**。原来那 41 项为 **32／9**。[逐项表](tasks-184.tsv)本次仅 kv_cache_copy 升级；原 41 项历史 registry 保持原字节。

新增 [KV cache 混合格式证明](../kv-cache-raw-bit-closure/README.md)覆盖旧接受的完整 masked x-block，K/V 各选五格式，共 25 个组合。两个 payload 和 metadata 均从初态读取，K stores 后接 V stores；整个最终内存等于独立原始字散射参考。全内存公式不需要 cache 分离或地址注入，双读回只沿用旧条件；别名、零宽度、零 stride 单 lane、masked 尾部及完整 frame 保留。

**新增 3 模块独立编译、20 条具名公理检查通过，新增公开 specification 为 1 条。** 现在共 883 条公开声明、3,799 个注册模块；构建证据复验旧 3,796 条并加入本次 3 条实际编译记录，未声称重新全量构建。见[构建回执](incremental-build/receipt.json)。

根代理独立重跑 41 项元数据测试与完整生成，10 个待安装文件、输入清单和 ZIP 成员字节相同；ZIP 时间戳造成快照回执哈希不同，见[复现比较](independent-generation/comparison.json)。安装后目录、逐项统计、历史 source adapter 和最终源码漂移检查见[根代理回执](root-review.json)。[复现命令](../../../current_source_acceptance.md)与[证据索引](evidence-package-index.json)可直接使用。

五格式是软件字存储证明，机器地址、字节 ABI、完整 Python static loop、GPU 行为仍未闭合。旧 arbitrary-R/36 注解合同未改，legacy fp64 不计为 native64。[上一版 K-cache 验收](../kcache-software-metadata/README.md)保持冻结于 882 条声明；[正式 comparator 样例](../comparator-initial-samples/README.md)仍只覆盖冻结 F3 的 1 条 FP4 任务声明，完整任务 corpus 与新声明待检查。
