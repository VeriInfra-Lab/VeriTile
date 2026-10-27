# BMM 原始函数体与证明

原始和已有整数余数改写候选均保留完整 Python 函数体，包括所有 constexpr 分支、None mask、tuple 赋值、原始 `_` 迭代变量、dot 参数与访存。调用类型及尺寸绑定放在函数体外。候选只表达原先已选择的余数改写；原 Python、wrapper、公开数学规格、FP 接口和输入范围不变。

通用前端按调用方 Boolean constexpr 枚举全部分支，None 只作为编译期缺省 mask。运行时无法合并的 None 分支、重写 constexpr、None 算术和循环后未确定的 mask 都拒绝编译。零次循环不会凭空建立 None 事实。实际执行、安全和等价证明使用原始 `_` 与 tuple 生成的代码；旧 `_i` 源码只保留在证明视图中，不声称完整寄存器状态相同。

37 个模块、8 组前端检查、72 项实际源码假设审计及 3 项新前端 footprint 检查通过。累计 3471 个入口与联合导入在 warm 和独立空项目构建中通过。136 项针对性 Python 检查及原 wrapper 的 11 个轴边界控制通过；381 项全量检查保留已知旧 Structural 数值凭据错误。40 次 comparator 因宿主隔离不可用，CPU interpreter 缺 Torch；未运行 GPU 或新数值实验。

累计 tasks 93/183、examples 19/19，仍有 90 个任务待完成。保持草稿，等待人工审阅。
