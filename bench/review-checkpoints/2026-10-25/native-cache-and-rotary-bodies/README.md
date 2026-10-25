# 单 token K/V cache 与两题旋转编码的原始函数体

三个任务的实际原始/已有候选函数体逐句粘贴，签名和调用实参在体外。
原始 Python 不变；两个旋转任务新写 Python 仅对照既有 Lean 索引候选。
K/V cache 保留原本的 static_range 商和带非零起点 arange，整个原始/
候选语句体与旧完整 typed source 及历史生成源由 rfl 核对。保留全部
18 个运行参数、两个 constexpr 及无效迭代前仍发生的 metadata 读取；
对未使用的 stride_kcd 实参证明整个程序独立。

旋转编码保留每个 signed64 cast、GQA/提前 return、独立 sine stride、
全部 mask、metadata 及 Q/K/cache 写回顺序。对四份 nopad 原始/候选及
两份普通旋转程序，分别在任意浮点原语和初始状态下证明完整执行的
Option.map 全内存相等，包含成功/失败和每个内存单元。临时索引寄存器
的 Nat/Int 投影差异不被冒充成 AST 或完整寄存器文件相等。实际新代码
的完整原始/候选等价、安全、独立数学公式、native 图、输入/输出/frame、
word/byte/host/Tensor 结果均另直接重证，维持原有完整范围。

共享前端支持不可变 Nat constexpr 商/静态循环、原始 shifted-arange，
及整数张量上下文的弱整数字面量 +,-,*,比较；不改写浮点字面量、
寄存器、显式浮点 cast 或除号的类型。静态循环拒绝运行/可变边界；
调用方 Lean term 内的名字不被重新绑定成 Triton 表达式。

详细本地/联合/独立无项目缓存构建、Git 源码闭包、针对性 Python、
全量 Python 已知旧 Structural.lean 凭据错误、实际 comparator 与 CPU
不可用结果见 review.json。没有 GPU 实验、新数值规则或全程序前提。
任务 44/183，examples 19/19；161 对中 156 对为逐句函数体、5 对旧 AST。
136 个未覆盖任务、1 个旧 AST 和 2 个间接入口仍待处理；保持草稿不合入。
