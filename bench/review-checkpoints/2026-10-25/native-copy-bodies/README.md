# Destination copy 与多 token K-cache 的函数体

四题实际原始/已有候选函数体逐句粘贴，运行实参与 constexpr、指针 dtype 在外。
原始 Python 不变；destination 两个新写 Python 仅对照既有 Lean 索引候选。
Copy 与 Kv1 真正相同的原始函数体明确共享，两个 Python 原始文件分别核对。
Kv1 的两维 mask、Kv2 原本的仅 head mask 和全部旧宽度前提保留。
删除旧 Lean 多出的同 FP32 store cast，在既有 raw-copy 解释下证明完整新旧
执行相等；完整新程序的 FP 等价、Real、输入/执行/安全/输出/frame、
word/byte/host/Tensor 结果直接重证，不对任意浮点代数假设 cast 恒等。

K-cache 保留所有 signed64 cast、负数 token shift、INT32 元数据、完整地址
及 return。整个原始/候选函数族与旧 typed source 由 rfl 核对。未用实参
仍在签名中，并证明对任意实际值整个程序不变；原有调用域没有缩小。
末尾空 return 是函数自然退出；嵌套及非末尾 return 保持拒绝。整数嵌套
读取根据实际指针存储判断，绝不因目标 cast 是整数就放过浮点读取；
地址/mask 子表达式也检查。完整程序和反向回归均纳入联合构建。

详细本地/联合/独立无项目缓存构建、Git 源码闭包、45 项针对性 Python、
全量 Python 已知旧 Structural.lean 数值凭据错误、实际 comparator 和
CPU 不可用结果见 review.json。没有新数值规则、全程序前提或 GPU 实验。
任务 41/183，examples 19/19；153 对中 148 对为逐句函数体，5 对旧 AST。
139 个未覆盖任务、1 个旧 AST 和 2 个间接入口仍待处理；保持草稿不合入。
