# 原生 OnlineSoftmax 与实际 FP32 观察

源码提交见 review.json 的 code_commit。新 Python 明确从独立 softmax
数学规格建立，不声称历史原版。两个 JIT 函数体与 Lean 原文逐句对应；
完整 online 源码包含两次输入读取和全部输出 store。libdevice.exp 不变。
FP32 scalar full 初始化与原生 maximum 保留实际循环精度。

两个完整源码的 Real erasure 均与明确的数学证明程序相等，原有数学公式、
成功执行、安全、输出和 frame 均保留。实际 FP32 程序对重新证明，规则集和
公开条件不变；数值条件重接到真正 FP32-loaded 输入。只读观察证明也保留：
其执行的是整段源码直到第二次实际读取的前缀，观察已读入的 x 寄存器，
仍证明成功读取整行、正常结束及全部内存不变。前缀由完整程序等式连接，
没有声称它们是另一个历史原版或独立 native JIT。

573 个显式模块、9,409 jobs 本地和独立干净检出均通过，联合导入通过。
干净检出开始无项目 build 或 .veritile-cache，仅复用固定外部 packages
用于编译；Git 对象只复用源码对象。source-closure.json 验证全部本地 Lean
依赖受 Git 跟踪并与干净检出逐字相同。70 个源码对照中 65 个为函数体模式，
5 个仍为旧 AST。33 项针对性 Python 检查通过；全量 278 项保留 1 项旧
数值 receipt 的 Structural.lean 哈希错误，不将全量 suite 记为通过。
官方 comparator 实际调用仍因未配置不可用；无新的 GPU/native 实验或数值假设。

当前 tasks 为 19/183 完成函数体对应，另有 161 个未完成、1 个旧 AST、
2 个间接入口；examples 为 14/19 覆盖，余 5 个待迁移。全量任务未完成；
草稿保持未合入，等待用户对完成结果的最终验收。
