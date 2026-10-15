# int4_matmul：原源码与 carry-dot 两个软件位语义合同

本包新增 **9 个模块、2 条公开合同**，根代理实际复编 9/9，通过 32 条具名标准公理检查。旧 14 个源码文件保持原样，完整旧参数、循环次数、group/stride/program 参数、初态和既有输入/地址条件均保留。没有增加有限数、正数、单块或结果等式假设。

原源码入口 SourceSoftwareCorrect 将已有完整辅助定理公开：half 反量化、fp32 逐项乘加及块间单独加法、最后 half 舍入均为具体位运算。新声明除名称外与旧辅助定理的合同字节相同。它是 original_source_translation 主入口，覆盖完整接受阶段。

DotAccCorrect 是另一个 observed_compiled_graph 入口：唯一 dot 赋值改为 live carry；全循环执行、独立初始内存参考、整个最终内存和 frame 均有证明。每个 K 块的顺序是经验证的有限排列数据，执行 IEEE fp32 FMA。两个合同的数值计算图不同；同一个完整两块见证分别给出 half 1、正序 carry 0、逆序 carry 2。不存在两图普遍等价的结论。

8 个原核离线编译配置覆盖 SM80/SM90、融合开关及两个原 autotune 配置。384 条捕获的 PTX 反量化线性链通过附加寄存器重定义/控制流检查；这不是一般 PTX 验证器，也没有证明 MMA 内部可表示为某个串行 FMA 排列。候选顶层清单漏列 native/receipt.json 的问题由 additional-native-binding 显式绑定其精确字节解决，冻结候选保持原样。

GPU 实际运行次数为 0。MMA 内部精度/分组、FTZ、NaN、物理地址/字节 ABI、编译器到硬件的数值对应关系仍待验证。half 的软件位语义闭合不等于 GPU 已通过。采集工具另行交付。

[根编译记录](build-records.json)、[具名公理检查](axiom-review.json)、[DotAcc 独立审查](independent-dotacc-review/review.json)、[原源码入口独立审查](independent-source-facade-review/review.json)。第一次 facade 编译因不完整首个对象搜索根失败，修正为完整已审查对象根后通过；首次记录保留于 root-facade-history。

本包先冻结证明与复核；工作区安装、统计及正式比较器范围由后续 metadata 包单独记录。
