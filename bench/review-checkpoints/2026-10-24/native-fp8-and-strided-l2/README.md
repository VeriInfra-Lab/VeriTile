# FP8 六配置与独立 stride L2 backward

FP8 的六个已有原始/候选配置逐句粘贴，两次 decode store 及真实隐式转换保留；
完整新旧函数族相等由 Lean 核对。既有候选的 Python 对照明确标注为新写，
原始 Python 不变。L2 backward 保留所有 cast、独立 strides 和运行 epsilon；
七个运行 int64 参数在函数体外绑定，只有 BLOCK_N constexpr。
已有 Real/FP、完整执行/安全、全部输出/frame、所有尺寸与 stride、
native/byte/host/Tensor 范围均重放；无新增数值规则或全程序前提。

1044 roots / 9,917 jobs 本地与独立初始空项目构建通过，联合导入通过。
源码提交、完整 Git/干净副本闭包、逐题语义审查与原始日志见 review.json。
仅复用固定外部 packages。45 项针对性 Python 通过；全量 290 项保留
既有未改动 Structural.lean 的旧数值凭据错误，未虚假 repin。
官方 comparator 均实际尝试，setup 缺 Go 1.24+；CPU 检查缺 torch，
均未记为通过。FP8 CPU interpreter 检查明确不启动 GPU；未跑 GPU 实验。

任务 32/183，examples 19/19；133 对中 128 对为原始函数体、5 对旧 AST。
148 个未覆盖任务、1 个旧 AST 和 2 个间接入口仍待处理；保留草稿不合入。
