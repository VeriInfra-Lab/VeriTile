# TMA matrix product 原始函数体与证明

完整原始函数与历史转换候选两份源码，保留 block pointer、load、默认 dot 参数、条件 FP16 cast 和 store。原始 Python 没有 FP32 cast；候选明确保留旧形式的冗余输出指针类型转换，不再把它归到原始源码。调用信息放在函数体外。

前端修复 typed Bool constexpr 分支后的类型环境。完整实际执行、访存安全、独立矩阵乘法、原始／候选 FP 等价性和 native/Tensor 结果重证。选定 native 模型明确定义 FP16 存储恒等，保留已有 FP32 恒等与 FP16 编码；不假定任意后端 cast 恒等。新旧 AST 不同，复用旧物理证明有完整具体模型执行等式连接。原始 Python、Real 规格及公开输入域不变。

18 个本题模块、109 个前端入口和累计 4209 个入口及联合导入通过，含独立空项目重建。27 项本题定理及 4 项前端证明仅用标准公理基。196 项针对性 Python 检查通过；441 项全量检查保留旧 Structural 数值凭据错误。20 次官方 comparator 尝试因主机隔离不可用，不计为通过。本题无 CPU contract checker，未运行 GPU 或新数值实验。

累计 tasks 102/183、examples 19/19，共 324 个选中对应项；剩余 81 个任务待修。保持草稿，等待人工审阅后再合入。
