# masked_select 原始函数体与 prefix 数学通道

既有 Python 七条语句完整粘贴，Bool cast、prefix 减一、mask 和写回
顺序都保留。类型与调用信息在体外；对照仍是同一真实 kernel 的
两个 autotune 配置，全部二十个配置保留，没有新建无关候选。

完整新旧 ComputeKernel 对任意调用参数由 rfl 证明相等，包含端口、
全部语句和转换。现有自然数 prefix 数学通道明确记录为 int64 存储
的投影；元信息只允许已定类型 Region.nat，不推断非负或 native
界限，仍由既有 prefix/IO/native 证明承担。前端完整显式 AST 检查
和三个错误用法守卫通过；默认 int64 通道不能冒充该自然数投影的
实际拒绝日志也保留。

十九个题级模块、九十五个前端回归和此前全部已迁移题目、十九个
例子及历史/可选范围完整重放。精确根、jobs、Git 源闭包和空项目
构建目录的独立重建见 review.json。七十二项针对性 Python 通过；
全量 317 项仅既有 Structural 数值凭据哈希错误。官方 comparator
因主机隔离功能不可用，实际 CPU 契约程序缺 torch，均未计作通过。
没有 GPU 或新数值实验。

累计 60/183 tasks、19/19 examples；205 对中 200 raw、5 旧 AST。
剩余 120+1+2 题继续，不合入，等待用户审阅。
