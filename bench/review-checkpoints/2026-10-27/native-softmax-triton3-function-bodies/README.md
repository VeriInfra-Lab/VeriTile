# softmax_triton3 原始函数体与证明

两份原始 Python 函数体完整保留 int64 转换、None 判断、FP32 load cast、finite classifier、1.0 常量和快速/回退分支。调用者在函数体外提供 dtype、标量值和可选指针是否存在；未受非 None 分支保护的可选指针访问会被拒绝。原 Python、wrapper、公开数学/FP/IO 规格和输入域未改。

旧手写 Nat 索引 AST 的相等证明已移除。RealExecution 直接证明当前完整函数的执行、输出、访问安全和 frame；FP 执行、原有 div_mul_rcp 条件规则、native 模型及 Tensor/host 连接分别重验。

初版新增的 is 全局关键字导致 9 个既有证明文件解析失败，warm/cold 失败日志保留。修复为 Triton 内部的上下文关键字后，这 9 个模块和新增普通 Lean 变量名检查通过。

34 个题目模块、5 组前端检查、33 项假设审计通过。累计 3402 个入口与联合导入在 warm 和独立空项目构建中通过。127 项针对性 Python 检查通过；372 项全量检查保留已知旧 Structural 数值凭据错误。15 次 comparator 因宿主隔离不可用，CPU checker 缺 Torch；未运行 GPU 或新数值实验。

累计 tasks 91/183、examples 19/19，仍有 92 个任务待完成。保持草稿，等待人工审阅。
