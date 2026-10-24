# Adam 与双 signed-mask ABI 原始函数体

源码提交和完整逐题语义检查见 review.json。Adam 保留 runtime 总长度/分块长度、
四个 FP32 参数、布尔乘法、两个原地输出、全部配置与任意合法 grid 次序。
mask-add 对 int32/int64 各自检查完整原始及候选函数体，删除 Python 没有的
alpha cast，并补全整数 mask 读端口。新的源码执行、安全、独立 Real 公式、
原有融合 FMA 图、完整内存等价和 frame 连接均重证；没有缩小尺寸或输入域。

688 roots / 9,512 jobs 集成和联合导入通过。独立冷启动分两组：Adam 与已有
回归为 659 roots / 9,479 jobs，mask-add 为其目录全部 29 roots / 3,534 jobs。
两个检出初始没有项目 build/cache；只复用固定外部 packages。每个依赖源码
均与最终代码提交逐字一致，完整 Git 闭包及分组哈希见 source-closure.json。

36 项针对测试通过；全量 281 项保留一个已知旧 numerical receipt hash 错误。
官方 comparator 缺 setup，额外 CPU contract 缺 torch，实际输出均保留，未冒充
通过；没有新 GPU/numerical 实验或数值假设。当前源码共 95 对：90 个原始函数体、
5 个旧 AST。tasks 为 21/183，examples 为 19/19；任务仍未结束，不合入。
