# 五题原生函数体与完整证明

mean reduction、L2 backward、两道 row softmax 和变长拷贝的实际 Python
逐句进入 Lean。相同前向/softmax 原始函数体明确共享；原始 Python 保留，
仅既有 Lean reciprocal 候选新建明确标注来源的 Python 对照。
运行/constexpr、signed cast、完整循环/分支/mask、原有维度与 IO 范围保留。
去除旧表示多出的 epsilon cast 和分支后同格式 FP32 store cast；
从实际程序重证完整执行、安全、全部输出/frame 与旧 native/host/Tensor 范围，
未引入 cast 恒等前提、新数值假设或整段程序前提。

964 roots / 9,828 jobs 本地与独立初始空项目构建通过，联合导入通过。
源码提交、Git/干净副本闭包和逐题语义审查见 review.json。仅复用固定外部
packages，不使用旧项目缓存。40 项针对性 Python 通过；全量 285 项保留
已复现的旧数值凭据错误，未虚假 repin。五题官方 comparator 均实际尝试；
补装 setup 缺 Go 1.24+，CPU contract 缺 torch，未计为通过；未跑 GPU。

当前 tasks 为 30/183，examples 为 19/19；119 对中 114 对为原始函数体、
5 对为旧 AST。150 个未覆盖任务、1 个旧 AST 与 2 个间接入口仍待处理。
继续保留草稿，等待人工查看与明确批准后才能合入。
