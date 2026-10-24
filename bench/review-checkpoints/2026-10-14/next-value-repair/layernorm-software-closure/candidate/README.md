此包补齐 `layer_norm_ops` 原来九个接受阶段的显式软件位结果，供独立复核；现有 184 分类和原 41 registry 均保持冻结。

实质新增是消除旧 `FPStages.software rest cfg literal` 中任意的 `literal : ℝ → FP.Value .fp32` 参数。真实已接受源图只有 +0 和自然数维度 N 需要字面值解释。新 `dimensionLiteral` 将 N 按 Nat → 精确 Rat → fp32 最近偶数舍入编码，证明对所有 N 正确连接到源字面值；没有添加 N≤2^24、N>0、有限数据或最终结果相等的前提。`dimension_rounds` 证明 N=16777217 编码为 16777216，避免把维度转换默认为精确实数。

`values_closed` 对任意备用 `rest` 证明完整独立初始内存位公式：所有实际 add/sub/mul/div、fp32 读写转换，以及已有明确的“按 lane 升序、以 +0 为种子”的软件归约都已固定。九个公开定理没有数值回调参数：

- RMS 一行 backward 的 DX/DW；plain 一行 backward 的 DX/DW/DB；
- 完整 C1、C2 归约阶段；supplied C1/C2 的 DX 阶段；
- residual add 的双输出；supplied vector、mean、rstd 三种 store 阶段。

全部保留原 stride、mask、别名分离条件、初始内存参考和 memory frame，及任意 block/N 参数。原范围本来不含完整 forward 统计量生产或多行梯度聚合，此包没有新增这些范围。W 的 undef 尾部和 DY 的 +0 尾部仍参与全部 BLOCK_N 乘积；非空完整 C2 执行见证确实存出 NaN（Inf×0），没有删除 padding arithmetic。DW/DB 的 +0 加法也保留，带符号零反例证明它不能无条件删除。

软件参数的消除不等于原生绑定已完成。仍须单独核对 GPU 归约树、FMA/乘加融合、除法指令/精度、FTZ/DAZ、NaN 规则、字面值/constexpr 编译转换、真实地址宽度及合法 launch。这里只保留并闭合已有选定软件图，不声称其与任意硬件树等价；也不以九个定理或一个新候选自动增加已验收任务数。

模块、声明和编译命令分别见 `receipt.json`、`theorems.json`、`build-records.json`；输入/对象闭包见 `dependency-bindings.json`。所有新命名定理经 `#axiomsClean`，旧任务所有 Lean 文件保持 F3 原 SHA。复验：`python replay.py --out /tmp/layernorm-software-fresh-review`，使用不存在的新目录；不运行 Lake、不修改工作区或固定 R。
