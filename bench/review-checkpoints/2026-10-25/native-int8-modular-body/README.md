# int8 matmul 原始函数体与完整证明重放

既有 Python 的 23 条顶层语句及全部内层语句完整粘贴，保留 cast、
mask、static_assert、循环、指针顺序与原始 add.u32 字符串和五项 ABI。
通用前端只接受已建模的纯双寄存器字加法；完整 AST 恒等、ABI 重排
及九个拒绝守卫验证转换，复用严格模32/有符号32原语，无 FP 新假设。
Python partial/prefix 原样可用，静态断言可直接执行；十五配置的断言
准入仍由既有证明承担，未假设任意断言为真。

实际 TileSource.source 直接等于 quotedSource，无题内语句投影。实际
循环、完整状态与失败、安全访问轨迹、输出和 frame 均重新证明。
最终 int32 存储 cast 保留，并由实际累积值的 signed32 规范性证明。
历史 AST 相等仅保留在明确标注的 Proofs 历史范围，不冒充新源码恒等。
两边仍是同一真实 kernel 的配置比较，十五配置及既有 Real/FP/native/
Tensor/host 范围完整保留。

43 个题级模块与 95 个前端回归通过；全部已迁移题目和 19 个 examples
在本地、联合导入及初始项目构建目录为空的独立 checkout 重放。
精确根、jobs 和 Git 源闭包见 review.json。74 项针对性 Python 通过；
全量 319 项仅既有 Structural 数值凭据哈希错误。官方 comparator
因主机隔离不可用，旧编译/交付缓存证据缺失，实际失败均留存，未计通过。
没有 GPU 或新数值实验。

累计 61/183 tasks、19/19 examples；206 对中 201 raw、5 旧 AST。
剩余 119+1+2 题继续，不合入，等待用户审阅。
