# 原 41 项剩余 11 项：实际语义缺口

[逐项报告](analysis/REPORT.zh-CN.md)检查了全部 11 项的实际声明和实现，区分存储表示、缺失位算法和编译图不一致。此包是源码分析与后续选题，没有新编译、GPU 实测或分类升级；当前仍为原 41 项 30/11、全库 51/132/1。

3 个纯复制任务已经有泛型逻辑复制证明；下一步需补明确存储宽度和独立 bit-memory 的表示桥，不能把选择 `bitsBackend` 重算作新证明。其余 8 项涉及 dot、exp/log、pow/sin/cos 或归约图。缺失函数不能填常数、实数 oracle 或任意回调再标为闭合。

[kcache 独立设计复核](copy-representation-review/receipt.json)明确要求保留初态 snapshot、alias、全部内存 frame 和原参数范围，并区分具名机器格式、legacy `.real` 与真正 native fp64。实现候选另行审查，本包不预先批准其分类升级。

报告的“matmul_dequantize_int4 尚需检查编译图”描述的是报告生成时的状态；随后取得的离线编译与舍入反例将作为独立补充保留，不回改本分析。已知 int4 图等价反例不属于等待实验通过的假设。所有有限实验仍只验证记录的配置，不能证明全域等价。

117 个实际源/证据文件均重新核对 SHA，并打包为 [reviewed-inputs.zip](reviewed-inputs.zip)，对应[清单](reviewed-inputs.json)。当前已验收的软件增量见[软件位语义补充](../software-closure-metadata/README.md)。
