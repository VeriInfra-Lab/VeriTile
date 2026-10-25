# chunk_cumsum_vector 原始函数体与完整证明

原版和已有 sum-left 候选 Python JIT 均未修改，两份11条语句函数体
独立逐句核对。所有 cast、tuple、signed range、mask、padding、
TF32-disabled dot、输出转换和 post-store carry 顺序保留；元信息体外。
六种输入乘六种输出 dtype、13 constexpr、9配置及host检查不变。

实际 signed 初始化、除余与完整前向循环、单次及整条访存安全重证。
严格 Nat/Int 恒等连接独立 Real 前缀公式；FP保留有序运算、原 finite
carry 域和严格 ADD-COMMUTE 模型证书。完整 typed memory/frame、
公开IO及既有 Tensor/host/native 范围保留，全部55个题级模块通过。

完整联合 warm/导入重放2359个根；55个受影响根及完整依赖在项目
对象初始为空的独立 checkout 重建。无共享前端修改。精确jobs、Git
源码闭包和命令见 review.json。80项针对性Python通过；全量325项
仅既有Structural凭据哈希错误。官方比较器隔离不可用，CPU契约缺
Torch，实际失败留存，不计通过；无GPU或新数值实验。

累计67/183tasks、19/19examples；218对中213raw、5旧AST。其余116题
继续处理；草稿不合入，等待用户审阅。
