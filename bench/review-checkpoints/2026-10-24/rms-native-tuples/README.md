# 原生 tuple 与 constexpr 的 RMSNorm 迁移

源码提交见 review.json 的 code_commit。本批迁移 rmsnorm_triton 和
rmsnorm_implementation 的完整选定 FP32 原版及已有地址交换候选；保留
Python 原文、两个循环、tuple zeros、转换、mask、运算顺序和 eps 的
constexpr 身份。两边各自均证明与冻结旧版完整 ComputeKernel 相等。

完整 2D-grid 与完整单行的原有数学范围分别保留，实际 FP 程序对、成功执行、
访存安全、输出和 frame 及两目录全部 61 个文件一起重新验证。旧有 4K tile
整数索引证明需要更大 Lean 栈空间，已在实际拥有该源码的 library 配置；
证明、索引范围和溢出反例源码没有变化。

550 个显式模块、9,381 jobs 本地及独立干净检出均通过，550 个联合导入通过。
独立检出开始没有项目 build 或 .veritile-cache，只复用固定外部 packages；
source-closure.json 核对全部仓库内 Lean 依赖均受 Git 跟踪且源码逐字相同。
64 个源码对照中 59 个为原始函数体模式，5 个仍是旧 argmax AST。
33 项针对性 Python 检查通过；全量 278 项仍有 1 项旧 numerical receipt
哈希错误，保留此前干净基线的复现，不将全量测试记作通过。
官方 comparator 实际调用仍因未配置而不可用；没有新增 GPU 实验或数值规则。

全量任务目前仅 19/183 完成函数体对应，另有 161 个待迁移、1 个旧 AST、
2 个间接入口；examples 仍为 11/19 已覆盖。全量完成检查应失败。
保持 draft，不合入，等待用户最终验收。
