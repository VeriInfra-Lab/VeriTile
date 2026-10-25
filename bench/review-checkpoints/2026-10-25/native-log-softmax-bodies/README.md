# log_softmax 原始函数体、元组语法与有符号证明

原Python及wrapper不改；正/反向原程序与既有地址候选共4段完整函数体，
指针/运行参数/constexpr信息体外。保留PID/arange与M/N/K元组INT64转换、
有符号size-1 clamp、mask、padding、全部FP32 cast和浮点/归约/写回顺序。
候选Python明确标为从已有Lean候选补写，不冒充历史原始源码。

共享前端保留括号元组源码，只适配为既有独立赋值语法；需并行寄存器捕获
的依赖元组明确拒绝，避免旧顺序赋值错误。真实三处整数转换执行、完整
程序相同、错误次序/arity拒绝均验证。严格整数地址连接与零尺寸mask分开
重证，不增加尺寸前提；完整状态/失败等价、Real前向/VJP公式及原有
FP/grid/safety/Tensor/native/host/frame全部范围保留，无新数值恒等式。

343个前端/控制/近期题级根9086jobs通过。2698根及全部依赖warm、联合导入
并从空项目产物独立重建；Git源码闭包和实际jobs见review.json。88项针对性
Python通过，全量333项仍仅旧Structural数值凭据错误，不虚假重钉。
官方比较器所需隔离不可用，CPU缺Torch；实际失败保留不计通过。
无GPU或新数值实验。

累计75/183tasks、19/19examples；235对中230raw、5旧AST，其余108题继续。
草稿不合入，等待用户审阅。
