# 浮点值结论

`FPCorrect.lean` 覆盖原先验收的 `quantize_global_scaled_store_slice`：读取初始 X 和 AbsmaxInv，按 fp32 先乘 `x * absmax_inv`，再乘精确常量 127；两次乘法分别舍入。值规格直接依赖初始内存，证明活跃位置的输出以及所有未写位置的保持。

这是已经约定的 llrint 之前阶段；CUDA llrint 和最终 int8 存储尚未纳入该阶段结论。fp32 数值由 `ScalarOps` 的位模型定义，可取指定舍入/下溢配置；结果不依赖补充 backend 的其它操作。

原始阶段的控制、地址和操作顺序对应由 `TypedShape.source_shape` 核验。硬件乘法、FTZ 配置、实际编译产物与该软件图的对应仍须实验/编译证据；不能将本结论登记成已验证 GPU。
