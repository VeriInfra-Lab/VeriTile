# Batched vecmat 原始函数体与直接证明

原程序及已有 block-first carry 候选各为完整 9 条顶层语句。
保留 N/K 和两个 PID 的 INT64 转换、signed K 循环、两处 unmasked load、
tuple broadcast、axis=2 归约、transpose、零 carry、原加法顺序及 store。
指针 dtype、运行参数绑定与 constexpr 在体外；原 Python/wrapper、
Correct/FPEquiv/IO/Contract 不改。候选 Python 明确由既有优化补写。

通用前端从调用上下文解析构造器 dtype 属性，并分别追踪 broadcast 的
结果类型，去除源码中不存在的同 FP32 store cast。显式窄化及真实输出
转换保留；不支持的同时赋值捕获会拒绝。相关精度回归一并重放。

SourceGraph.source 直接连接当前原始函数体，signed 执行、不变量、完整
内存/frame、安全、既有 full-grid FP 等价、独立 full-K Real、有限索引、
host/Tensor 及所有原输入/输出域重证。旧 stream contract 明确指向历史
cast-free 程序，未伪称当前 AST 相等。

2871 个原有/当前根 warm、joint、从空项目产物独立完整重建。
独立目录曾因磁盘空间不足中断，清理已完成的另一临时副本的可重建缓存后，
在同一个独立目录恢复并完成；没有借用其他 checkout 的项目产物。
源码 closure 与同一 Git source commit 一致；未使用 .veritile-cache。
102 项针对性 Python 通过；全量 347 项只有旧 Structural 数值凭据错误。
14 次官方 comparator/trust 因宿主隔离不可用，实际 CPU checker 缺 Torch，
均保留失败，不计通过；静态原 repair/wrapper 断言另行执行通过。
无 GPU 或新增数值实验。

探索性 82 个历史/当前根检查暴露一个未选用的旧 FPModel witness 缺口，
已在改动前的发布父提交复现；原文件、命题未改，未以公理或整体假设
掩盖。失败及范围见 legacy-failure.json。当前 41 个选用根及 3 个修改的
共享根全部通过，2871 个完整通过范围列在 modules.json。

累计 81/183 tasks、19/19 examples；251 对中 246 raw、5 旧 AST。
仍有 102 题待迁移；保持草稿，等待用户审阅后才能合入。
