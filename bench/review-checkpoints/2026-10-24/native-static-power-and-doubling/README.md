# 原始 constexpr 与四个翻倍函数体

源码提交、逐题语义核对和完整验证记录见 review.json。mul-exponent 原样保留
静态幂常量和两个 int64 cast。共享前端仅处理可精确表示的静态二进制幂，
保持 weak Python literal，拒绝任意浮点幂、runtime 指数、越界和重赋值。
有符号索引通过完整执行/内存证明连接旧地址公式；native 复用原有索引和
指针界限，不新增逐题位算术证明或数值假设。triton-mul2 的四个实际 Python
函数均原样粘贴，每个完整 typed Kernel 与旧表示的相等性由 rfl 验证。
独立 Real 公式、全部配置、完整 grid、执行、安全、输出和 frame 均保留。

743 roots / 9,583 jobs 本地和独立冷启动构建通过，联合导入通过；两题目录
全部 Lean 文件和旧回归均覆盖。1,336 个本地 Lean 依赖均由 Git 跟踪，冷检出
与代码提交逐字相同，无初始项目 build/cache，只复用固定外部 packages。
36 个针对性 Python 测试通过；全量 281 项仍有一个已复现的旧 numerical
receipt 错误。官方 comparator 缺 setup，实际失败日志保留。无 GPU 实验。

当前任务源码为 23/183，examples 为 19/19，共 101 对源码：96 对原始函数体、
5 对旧 AST。仍有 157 个未覆盖任务、1 个旧 AST、2 个间接/未解析入口。
这份记录不表示全部任务完成，也不授权合入。
