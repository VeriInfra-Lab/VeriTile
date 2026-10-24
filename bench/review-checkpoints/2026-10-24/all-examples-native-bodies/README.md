# 19 个 examples 的原生函数体与完整证明

源码提交见 review.json。19/19 examples 已完成源码对应：两个历史 Python、
两个先前编写的 native counterpart、十五个明确标注的数学规格派生源程序。
全部 Kernel 主体保留原生 Python 语句；没有正文 dtype 指令、Lean antiquote、
Lean 循环或数值辅助函数。来源类别不混用，也不声称新来源是历史原版。

Adam 使用实际原始 Python，保留 runtime/constexpr 身份、布尔乘法、全部
mask 和两个原地 store；候选只交换最终 momentum 加法。完整数学投影、
独立 Real 公式、执行、安全、frame 和构造出的全 grid launch 结论均保留。
旧任务复用的 grid 接口经完整 kernel 投影连接，所有相关原配置范围重放。
两个 HyperConnections 新 Python 从独立规格建立；保留标量特例、完整矩阵、
两套 map/输出、全部维度和每次 Sinkhorn 迭代。tau 为 runtime FP32，dot
精度显式，既有候选和数值规则不变。实际源码的数学投影与完整证明程序
对应；FP32 除法改倒数乘法只复用原有局部规则及 finite/nonzero 条件。

前端保留 FP32 调用参数的首个精度作用域，覆盖 where、FP64 运算与 FP16
存储；执行型回归明确区分这些调用精度。广播和 FP32 dot 结果的 dtype
传递不发明多余转换，混合循环状态仍拒绝。无新的数值假设或硬件保证。

618 个模块、9,455 jobs 本地及独立干净项目构建通过，联合导入通过。
33 项针对性 Python 检查通过；全量 278 项仍保留一个已复现的旧数值 receipt
错误，未虚假 repin。官方 comparator 实际调用仍因未配置不可用。无 GPU/
native 实验。所有本地 Lean 依赖由 Git 跟踪，闭包与干净检出逐字一致；
项目 build/cache 不共享，只复用固定外部 packages。89 对源码中 84 对为
原始函数体模式、5 对仍为旧 AST。tasks 仍是 19/183，还有 161 个未覆盖、
1 个旧 AST、2 个间接入口；完整任务未结束。草稿保持未合入，等待最终审阅。
