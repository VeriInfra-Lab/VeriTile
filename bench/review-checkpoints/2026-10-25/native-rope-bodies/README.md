# 两道 RoPE 的真实函数体

原始 Python 不变，逐句保留 docstring、cast、方向分支、head 循环与两次写回。
调用 dtype、运行 strides/count 和 constexpr 在外部。新写 Python 仅对照
既有 Lean 索引候选，并明确标注来源。
Fast RoPE 的组大小赋值真实存在；普通 RoPE 的组大小是 constexpr 参数。
普通 RoPE 删除旧 Lean 多出的赋值，直接重证实际前缀和完整循环。
普通 RoPE 的 Real/native 公开 IO 绑定本题实际源码，重证完整执行与安全；
Fast RoPE 仍与旧 typed source 完整相等，但分别重证自己的前缀，
只通过完整循环体相等共享循环证明。全部旧 Real/FP、执行、安全、输出、
frame、native/byte/host/Tensor 范围保留，无新增数值规则或全程序前提。

本地、联合导入、独立初始空项目构建以及源码闭包见 review.json。
93 项已配置前端回归全部纳入联合 roots，包含此前源码复现的八项失败。
修正 FP8 分支穷尽性、位宽保持、真实 source dtype cast；旧测试改用
已证明的严格模型证书，保留错误精度/guard/mask 的负向检查。
抽象 Real 后端的 cumsum 完整执行直接证明，不再冒用严格软件理论模型。
45 项 Python 通过；全量 290 项保留旧 Structural.lean 数值凭据错误。
官方工具已安装，实际调用仍因主机隔离能力不足无法运行；CPU 检查缺
torch。未启动 GPU 或数值实验。

任务 37/183，examples 19/19；145 对中 140 对为逐句函数体、5 对旧 AST。
143 个未覆盖任务、1 个旧 AST 和 2 个间接入口仍待处理。保持草稿不合入。
