此包新增完整两轮循环的位反例，不升级 `int4_matmul` 的完成分类。

三层边界分别如下：

- 原源码 AST：旧接受范围为 SPLIT_K=1、任意 K 分块循环、全部 half 解量化/加载/输出和 frame。新的 `source_profile_software_correctness` 展开已有 `∀D` 定理，给出从初始内存独立计算的逐位递推；这主要是显式实例化，不是新增的原语正确性证据。新 `all_memory_independent_of_unused` 证明输出与未使用回调无关。
- 编译图：原 Python 函数体 AST 原样离线编译，SM90、原 host 64×128×32 配置、fp16 数据/scale 和 int32 packed 数据。在 `enable_fp_fusion=False/True` 两种配置下均成功，TTIR 都将原累加器作为 `tt.dot(A,B,acc)` 第三实参，PTX 使用 WGMMA。`native/receipt.json` 绑定原文件、提取函数、全部编译产物。没有运行 GPU。
- 位模型：同一合法 64×128×32 tile、两轮 K、packed nibble 条件、原输入/输出布局下，完整源图成功存出 half `0x3c00`；保留 carry 依赖的顺序 FMA 软件图存出 half `0x0000`。第一轮得到 +1，第二轮非零乘积为 2^24、+1、−2^24。所有 32 项、半精度转换和最后存储都保留。

因此，不能把“从 +0 开始的 dot 再加 accumulator”与“dot 内携带 accumulator”当作对所有后端成立的等式，也不能把此等式列成等待实验通过的假设。这个反例没有绑定 WGMMA 的内部树、精度、NaN 或 FTZ 规则，**不证明实际 GPU 必然得到 0**。面向编译图的新合同必须显式保留 dotAcc/accumulator 依赖，并另行绑定真实指令语义。

可复核声明见 `theorems.json`，原范围前提见 `after/.../SoftwareCorrect.lean`。三个模块真实 direct Lean 编译通过，所有命名定理经 `#axiomsClean` 检查。旧失败日志保留在 `logs/`。`dependency-bindings.json` 绑定固定 R 的项目导入闭包、对象和直接外部依赖；`original-source-unchanged.json` 证明现有任务源文件仍等于 F3 冻结版本。

复验：`python replay.py --out /tmp/int4-dotacc-gap-fresh-review`。输出目录必须不存在。脚本先核验哈希，使用只读固定对象，再在新目录按拓扑 direct compile；不会修改工作区、固定 R 或统计。
