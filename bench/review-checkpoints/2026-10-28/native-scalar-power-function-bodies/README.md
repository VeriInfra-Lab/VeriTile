# Scalar power 原始函数体与证明

原始 helper、完整原始 writer 和已有整数索引交换候选共三份源码对应项。保留两个分支、block pointer、boundary check、所有显式 cast、runtime val0 和原有 decorator。调用类型、参数与主 CUDA import 绑定在函数体外；fallback 路径记录为来源，不假定 intrinsic 相同。

旧展开遗漏的输入 FP32-to-FP32 cast 已保留，借助明确的证明适配器连接旧内存/执行引理；Real 与既有 word 解释分别验证，不假定任意 cast 恒等。独立数学规格、完整原始／候选 FP 等价性、访存安全和所有本题入口重新验证，公开输入域不变。

27 个本题模块、108 个前端入口和累计 4190 个入口及联合导入通过，含独立空项目重建。55 项本题定理及 1 项前端 AST 证明仅用标准公理基。189 项针对性 Python 检查通过；434 项全量检查保留旧 Structural 数值凭据错误。官方 comparator 隔离和 Torch 不可用，不计为通过。未运行 GPU 或新数值实验。

累计 tasks 101/183、examples 19/19，共 322 个选中对应项；剩余 82 个任务待修。保持草稿，等待人工审阅后再合入。
