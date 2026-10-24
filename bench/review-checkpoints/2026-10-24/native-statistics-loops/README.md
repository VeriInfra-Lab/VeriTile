# 原生 Welford 与 LayerNorm 循环

源码提交见 review.json 的 code_commit。两个例子均明确登记为从独立数学规格
建立的新 Python，不声称恢复历史原版。四个 JIT 函数体与 Lean 原文逐句对应。
原有 Real 数学公式、全部公开条件和 IO 范围保持；实际源码的 FP32 读取、
BF16 输出、两个统计量、LayerNorm 第二轮读取及完整执行/安全/frame 均重证。

原生 i.to(float32) 和 uint64 count tensor 的显式转换复用已有 fromNat 及
严格计数规则，保持原来的计数上界；没有新增数值假设。FP 条件重接到实际
FP32-loaded 输入。LayerNorm 空行仍在任意数值模型下成功且全部内存不变。
历史 LegacyRealEquiv 保留并重放，但明确指向冻结旧源码，不冒充当前 native 证明。

565 个显式模块、9,400 jobs 本地及独立干净检出均通过，联合导入通过。
独立检出从没有项目 build 或 .veritile-cache 的状态开始，只共享固定外部
packages；Git 对象存储只复用源码对象。全部仓库内 Lean 依赖均受 Git 跟踪且
源码逐字相同，见 source-closure.json。68 个对应关系中 63 个为函数体模式，
5 个仍为旧 AST。33 项针对性 Python 检查通过；全量 278 项仍有 1 项旧数值
凭据哈希错误，不记为全量通过。官方 comparator 实际调用仍因未配置不可用。
未运行新的 GPU/native 实验；没有新增数值规则或 axioms。

全量 tasks 仍为 19/183 完成对应，另有 161 个未完成、1 个旧 AST、2 个间接入口；
examples 为 13/19 覆盖，余 6 个待迁移。整个任务尚未完成，保持 draft、先不合入。
