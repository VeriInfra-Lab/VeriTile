# matmul_dequantize_int4：原源码及两种编译图的完整软件位合同

本包新增 **15 模块、3 条公开合同**；根代理独立复编 15/15，通过 58 条具名标准公理检查。原源码主入口、关闭融合的 carry-dot、开启 half FMA 融合的 carry-dot 分别登记，没有把不同计算图等同。

原 source 完整合同固定逐 K 的 fp32 multiply/add 从 +0 开始，再把块结果单独 rounded add 到累加器；half 的两次乘法和减法逐次舍入。独立参考只读取初始内存，最后有序 scatter。两 NO_GROUPS、任意旧 Nat K/循环、stride/group 参数、原 PackedInputs 和 active-output 单射条件完整保留，没有新增有限值、K 整除、无 NaN 或区域分离条件。全内存定理无需单射，读回/frame 使用原条件。

两个 compiled 合同的 dot 均带 live carry，采用每个 K lane 恰出现一次的显式软件 FMA 排列。融合路径在共享 DSL 的实际 prefix 后读取当前 b/scales/zeros 寄存器，执行任务局部 half FMA 指令，再执行实际 dotAcc/pointer suffix；循环次数读取 prelude 的 num_pid_k，最后执行原 epilogue。该扩展不是共享 DSL 支持了 scalar FMA 的声明。

half FMA 软件原语是总定义，明确规定 NaN canonicalization、Inf/0、signed zero、subnormal 和一次直接 half 舍入。它不自动代表 PTX 特殊值/FTZ 行为。完整两循环软件见证分别区分融合与非融合；该小语义 tile 不是 GPU 配置。scalar weight=3、zero=2、scale=0x3c01 的分开舍入与融合值已有严格反例，不列为待实验通过的普遍假设。

原冻结捕获只有一个合法 autotune tile，SM80/90 × NO_GROUPS × fusion 共 8 个离线配置。独立 PTX 静态链检查的部分边跨控制标记，未证明 dominance 或机器级 refinement。MMA/WGMMA 的内部精度/顺序及可串行化、half NaN/FTZ、byte/address ABI、其它 16 个 tile 和 GPU 执行仍未证明。

[根编译记录](build-records.json)、[具名检查](axiom-review.json)、[独立审查](independent-review/review.json)、[编译图范围](compiled-candidate/scope.json)、[原源码范围](source-candidate/scope.json)。此处冻结证明和复核，后续 metadata 包才安装并统计；没有 GPU 运行。
