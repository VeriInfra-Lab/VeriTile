# apply_penalty 原始函数体与完整证明

原版和已有 reciprocal 候选 Python JIT 均未修改；两份17条语句函数体
独立逐句核对，保留全部 cast、元数据 load、mask、计数提升与写回。
类型、runtime stride 绑定及 constexpr 在体外；既有 INT32 存储、
非负逻辑通道、FP32 运算、公开 IO 和 host 契约不变。

前端 fresh dest=old+rhs 复用普通赋值推断，避免后续 Nat load 的预扫描
把已有 signed offset 改成 Nat。完整手写 AST 守卫在旧版本实际失败、
修复后通过；95前端加10旧回归共105根/3822 jobs通过。

完整 FP 执行、原有 DIV-MUL-RCP 域内完整 tagged memory 等价、成功和
frame 重证。独立 Real 数学公式通过严格 Nat/Int 内存及访存安全连接
接到原始源码，旧 AST 恒等不再冒充当前源码身份。Tensor/全grid 使用
同一 canonical raw source；所有26题级根通过3504 jobs。

共享前端变更后全部2304根在3cfa3fcb通过 warm、联合导入及项目产物
初始为空的独立重建。最终把未注解 stride 标为普通整数，保留原有
INT64值转换；只有本题元信息改变，共享源码字节一致。最终版本又
通过全部2304根 warm/联合导入和本题全部26根独立空产物重建，覆盖
依赖图中全部23个受影响根；两个源码快照及精确连接均留证。
精确jobs和Git源码闭包见review.json。79项针对性Python
通过；全量324项仅既有Structural凭据哈希错误。NumPy实际包装器/
函数体52项控制通过；该有界解释器不验证native signed64边界或FMA。
官方比较器缺主机隔离，实际失败留存；没有GPU或新数值实验。

累计66/183tasks、19/19examples；216对中211raw、5旧AST。剩余117题
继续；草稿不合入，等待用户审阅。
