# aligned attention 原始 kernel 离线编译

[收据](receipt.json)记录 Triton 3.8.0、cuda sm90 目标及全部编译参数。原始 JIT AST 保持不变；fp16 配置编译成功，产物在 fp16/；bf16 配置被原生编译器拒绝，原因是该源码使用的 `tl.dot(out_dtype=bf16)` 不受支持，原始诊断见 [bf16/error.txt](bf16/error.txt)。

这份证据支持明确的 fp16 源码 profile，不能支持旧的全部 FloatDType 声明。没有执行 GPU kernel，也未证明 IEEE 数值精化或任意 launch/dtype 配置正确。全部 11 个归档产物的 SHA-256 已按收据逐项复核。
