# 两道 token softmax 的原始函数体与完整证明

Llama、Bloom 的两份原版及两份既有 Python 候选完全不变；独立清单
分别核对共享原始引用。所有十一条顶层语句、五处显式 int64 cast、
padding、归约、除法、mask 和写回顺序保留，没有新建 Python 候选。

元信息保留实际 INT32 元数据存储、自然数逻辑通道、INT64 runtime
strides 和 block constexpr。加载先读 Nat 通道，再按源码转成 Int；
完整显式 AST 守卫在旧版本实际失败、修复后通过。dtype 参数保留原有
六种类型范围，不把通用 API 固定为 FP32；Python wrapper 原有 FP32
要求不变。初次全量重放发现宽泛整数推断改动影响十个旧模块，已删除该宽泛
改动，仅保留自然数加载后显式加宽；162 个修复回归通过，失败日志
及最终完整重建留存。任意两种输入/输出 dtype 的完整内核恒等及三种错误用法
守卫通过；SourceDType 及其旧有限特化保持不变。

实际源码执行内存、完整候选程序和失败状态、访问安全、独立 Real
公式、公开 IO、全 grid 输出和 frame、Tensor/host/native 范围完整
重证。数值假设不变，整数投影不冒充机器范围证明。57 个题级模块、
95 个前端回归及此前全部已迁移任务与 19 个 examples 完整重放，
精确根、jobs、初始项目对象为空的独立重建及 Git 源闭包见 review.json。

76 项针对性 Python 通过；全量 321 项仅既有 Structural 数值凭据
哈希错误。官方 comparator 因主机隔离不可用，实际 CPU 契约缺 torch；
原始失败留存，不计通过。没有 GPU 或新数值实验。

累计 63/183 tasks、19/19 examples；210 对中 205 raw、5 旧 AST。
剩余 117+1+2 题继续，不合入，等待用户审阅。
