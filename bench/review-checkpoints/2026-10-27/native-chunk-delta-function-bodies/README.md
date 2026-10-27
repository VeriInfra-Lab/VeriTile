# chunk_delta_fwd 原始函数体与证明

共享 Kernels.lean 保留完整原始函数体及已有地址分组候选；选定题目明确引用它们。原 Python、wrapper、公开规格和静态 dot 条件未改。新 candidate_kernels.py 只把已有 Lean 候选写成 Python，完整 AST 比较确认只有三处整数地址乘法分组不同，签名和全部 autotune 配置保留。

tuple 程序索引、双层 range、初始/最终状态开关、所有 stride、block pointer、dtype 查询、cast、boundary check 和 dot 精度逐句保留。元数据位于函数体外，输出端口明确。证明通过既有循环语义定理连接实际 counted loop，保留成功、失败、安全、完整 grid、独立 Real 递推、原 native 模型及 frame；没有新增数值规则或缩小输入域，也未把前驱较宽输入域的缺陷标成修复。

133 个模块（64 个 chunk delta、69 个 RoPE）、4 组前端测试和 30 项假设审计通过。累计 3368 个入口与联合导入在 warm 和独立空项目构建中通过；公开 FP assumptions 为 none。121 项针对性 Python 检查通过；366 项全量检查保留已知旧 Structural 数值凭据错误。25 次 comparator 因宿主隔离不可用，CPU checker 缺 Torch；未运行 GPU 或新数值实验。

首次累计 warm/cold 构建暴露 RoPE 旧证明多出的两次 FP32 写回转换，失败源码和日志保留在 initial-failed-source。已按原函数体修正执行展开，保留显式输入转换并重新验证完整 Real/native/等价/安全证明；没有假定任意 cast 恒等，也未改变独立数学规格或公开域。

累计 tasks 90/183、examples 19/19，仍有 93 个任务待完成。保持草稿，等待人工审阅。
