# 矩阵乘法、BGMV shrink 与 multinomial sampling 原始函数体

矩阵乘法 4 个、BGMV shrink 8 个和 multinomial sampling 5 个新函数体/ABI 对应项保留原始 cast、分支、提前 return、循环与运算次序。已有候选仅作分组宽度移位、最后乘法换序或 token 循环展开。矩阵/BGMV 完整 Python 模块、multinomial 完整 JIT 对已核对。原程序、调用/契约和公开 Real/FP 文件不变。

矩阵证明从原始 K 推导实际循环次数，保留完整 grouped grid、零轴和越界 id 无写入行为。BGMV 保留全部 batch × 64 split 程序、每个地址的 atomic 次序、实际输出转换和 sentinel 行；完整执行、安全、数学结果与 frame 重新连接。严格 FP 乘法证书沿用，无新增数值假设。

矩阵 215 入口的开发验证使用其当时的前端；BGMV 62 入口随后验证了 atomic 隐式转换修复；multinomial 的 54 个入口也使用该最终前端。三题及所有前端回归在同一最终源码的 5524 个累计模块与联合导入中再次验证，包含独立空项目缓存重建。6962 个本地 Lean 源文件与 Git 一致；1807 个定义模块的 36595 个定理通过标准公理依赖检查。

335 项针对性 Python 测试通过；580 项全量测试保留一条旧 Structural 数值凭据错误。331 次官方 comparator 因主机隔离不可用；Torch/Triton 缺依赖。未运行 GPU 或新数值实验。失败草稿和各阶段源码记录保留。

累计 tasks 116/183、examples 19/19、409 个对应项；67 个任务待迁移。保持草稿，等待人工审查再决定合入。
