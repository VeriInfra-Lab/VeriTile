# 原生 fused SiLU / SwiGLU 完整流水线

源码提交见 review.json；两个例子的新 Python 明确从独立数学规格建立，
不声称历史原版。七段 JIT 的函数体与 Lean 原文对应。实数公式、成功
执行、安全、输出和完整 frame 重新证明；统一 IO 数据保存在各自 IO.lean。

FusedSiLU 的实际 FP32 临时缓冲区会被下一阶段重新读取，公开 FP 结论
明确使用已有 scalarFP32 软件字模型，覆盖所有配置和所有 FP32 位模式；
sigmoid 和 BF16 转换保留相同 opaque 回调。不声称旧的无类型任意代数
结论适用于新类型源码。回调空间非空由 Lean 证明，没有新数值假设。
FusedSwiglu 仍对任意数值代数证明完整 masked 两阶段等价，包含中间
BF16 舍入、转回 FP32、所有空尺寸和尾块。旧 boundary-rounding 证明
明确连接归档的旧 Lean 程序，不与当前 native 证明混用。

590 模块、9,426 jobs 本地及独立干净项目构建通过；联合导入通过。
33 项针对性 Python 测试通过。全量 278 项保留一个已复现的旧 numerical
receipt 哈希错误，不虚报全量通过。官方 comparator 实际调用仍未配置；
无 GPU/native 实验。source-closure.json 对照源码提交与干净检出，项目
build/cache 不复用，只有固定外部 packages 共用。全部目标尚未完成：
tasks 19/183；examples 16/19。草稿仍不合入，等待用户最终审阅。
