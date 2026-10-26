# Fused decoding rotary/cache 原始函数体

原 Python JIT 与 wrapper 不变；原程序和既有 Q 地址 CSE 候选两段完整函数体，
各 21 条顶层语句。9 个指针、27 个运行整数、4 个 constexpr 均在体外。
候选 Python 依据已有 Lean 候选补写并明确标注，运行 stride 不变成 constexpr。
保留 Q、条件 K-cache scatter、完整宽度 V、两种 cache 布局、全部 INT64 cast。

自然/整数半维度和 mask 连接重证，输入条件不增。32 个模块重放完整执行、安全、
独立 Real 公式、全状态/失败候选等价、全部 native words、grid/Tensor/host/frame。
独立声明审计补齐已移动的 native/Tensor imports 和当前调用名，全部通过。

2735 根 warm + 联合导入；32 题级根及全部依赖从空项目独立编译。
空项目初始 commit 为 2d697，最终仅修改非 Lean CPU 检查脚本；所有 Lean/构建
输入逐字相同，最终源码再次重放，具体 provenance 见 cold-final-source.json。
共享 VeriTile 源码和构建配置保持上次全 2703 根独立验证版本不变。
98 项针对性 Python 通过，全量 343 项仅旧 Structural 数值凭据错误。
官方比较器所需隔离不可用；CPU 在源码核对后因缺 Torch 停止，不计通过、
不重钉历史 CPU 凭据；无 GPU 或新数值实验。

累计 78/183 tasks、19/19 examples；245 对中 240 raw、5 旧 AST。
仍有 105 题待迁移。保留草稿，不合入，等待用户审阅。
