# HGRN、SwiGLU backward 与 ReLU 原始函数体

三题原始 Python 不变，完整函数体直接粘贴，类型和调用绑定在体外。
HGRN 保留实际 tuple、_、do、elif、所有可选状态及 cast；重证前反向
全网格和独立梯度。SwiGLU backward 完整 AST 相等覆盖全部旧参数、
两种重算分支；candidate_kernels.py 明确为既有候选的新 native 对应。
ReLU 公开 IO 使用实际函数体，完整可失败状态与旧物理坐标模型相连。

50+26+18 题级模块和全部 95 前端回归均重放；1873 根本地、联合导入
及独立无项目缓存副本完整重建通过，精确 jobs 与 Git 闭包见 review.json。
59 项针对性 Python 通过；全量 304 项仅既有 Structural.lean 凭据哈希
错误。实际 comparator 因主机隔离不可用，CPU 缺 torch；未跑 GPU。

累计 54/183 tasks、19/19 examples，190 对中 185 为函数体、5 为旧 AST。
剩余 126+1+2 题继续修复，待用户审阅后再决定合入。
