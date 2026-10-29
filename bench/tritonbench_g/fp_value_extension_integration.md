# 值合同扩展：第三版集成边界

本候选冻结46个已复核证据包和37项尚未执行GPU的实验任务。184项同表记录，32项扩展增加125条公开specification；原41项登记和747条历史公开声明保持原文。

新增记录包括121条参数化值合同（其中6条独立SC模型）、2条单独frame合同、1条递推恒等式和1条SC互斥协议合同。固定见证、协议、frame、不同编译图与独立模型均不能补齐原源码的值阶段。

当前按原接受的输入/配置/阶段范围：50项有闭合的软件浮点模型，133项保留后端/数值原语参数，1项仅有修复候选。50项包括原修复中的30项和本轮完整声明配置中的20项；不代表所有Python dtype、完整host程序或GPU实现已证明。

LayerNorm、StreamK和spinning使用各自的typed-AST到SC模型桥，与共享DSL分开。LayerNorm后向/最终归约软件闭合，前向sqrt仍是回调；StreamK两条软件host合同只覆盖ordered_host_launches，不能代替两波中间状态的闭合；SC互斥不计作数值合同。triton_attention全部三阶段有值证明，预处理三种存储格式软件闭合，前后向保留hostLiteral/exp/dot回调。

源码分步运算、编译出的FMA/dotAcc、修改后的候选源码分别登记；不假定它们全域等价。原23项中仅2项覆盖原wrapper枚举的全部dtype，其余21项覆盖明确选定的profile。已知反例要求修改图/模型，有限实验不能把反例关系变成普遍定理。

两个Retention任务已修复signed-low32索引参考公式，原24条公开声明保持；不增加BT上界。该语义修复使用独立receipt和源码SHA，不能靠header未变来放行。

本冻结包通过33项拒错测试及独立目录重放；统一3783模块构建、canonical源码/验收适配和CI补充检查另行提供同源证据。此候选本身不宣称这些后续步骤已通过，也不覆盖GPU正确性。旧41与历史总表未回写。

重放：解压inputs.zip后运行 `python3 bench/fp_value_extensions.py --check` 及 `python3 -m unittest discover -s bench/tests -p test_fp_value_extensions.py -v`。
