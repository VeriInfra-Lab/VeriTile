# 浮点值结论

新结论覆盖既有验收范围：`ScalePre` 已给定的量化前处理阶段，两个输出分别为尚未执行 bias/int8 转换的浮点值和保存的 scale。比例计算 `(preScale*x)/scale` 按 fp32 两次独立操作保留，不改为实数比例等价式。证明从初始内存取输入和 scale、两个输出的 readback、全部未写位置的 frame；两个输出区域沿用原先的不重叠条件。

`per_block_scaled_fp32` 对任意 backend 保留 Q/K 两个 preScale 的值图，常量转换也由 backend 显式给出。`per_block_scaled_k_software` 在原 K 配置 `preScale=1` 下给出每次乘除的具体二进制32舍入结果，保留乘1，不使用恒等化简来绕过 NaN、负零等行为。

Q 的宿主预计算系数对应、原 scale producer、bias/最终 int8 cast 和整核组合仍在此已约定阶段之外。硬件除法、舍入配置与编译产物对应尚需实验。软件结论不记为 GPU 验证通过。
