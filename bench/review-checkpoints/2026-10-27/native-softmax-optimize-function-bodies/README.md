# softmax_optimize 原始函数体与证明

原辅助函数、四循环 kernel 和已有候选函数体直接保留在 Kernels.lean；Python、wrapper、公开规格未改。dtype 和调用信息位于函数体外，保留所有 cast、NaN/无穷值 guard、归约、循环、mask 与 store。候选仍只交换四处整数地址乘法。

证明现在跟随实际 N 参数绑定，移除旧转写虚构的 cols 初始化，并证明实际 cdiv 表达式与原数学值的连接。完整执行、失败、安全、Real、具体 FP、frame、host 和 Tensor 证明重放通过。

54 个本题入口及其完整依赖从空项目构建目录独立通过；3303 个累计入口和联合导入通过 warm 检查。本批没有共享前端改动，未宣称重新对全部累计入口冷构建。21 项假设审计通过，公开 FP assumptions 为 none。

119 项针对性 Python 检查通过；364 项全量检查只保留已知旧 Structural 数值凭据错误。22 次 comparator 因宿主隔离不可用，CPU checker 缺 Torch，均保留实际日志。未运行 GPU 或新数值实验。

累计 tasks 89/183、examples 19/19，仍有 94 个任务待完成。保持草稿，等待人工审阅。
