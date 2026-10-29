## 新增机器数入口（2026-10-14）

`FPCorrect.lean` 公开 `int8_dequant_matmul_fp_correctness`，覆盖原来接受的
`SPLIT_K=1 / EVEN_K=true` 全部 K 迭代和 `has_bias` 两分支。数据类型来自 Python
测试与输出分配：A/B 为 int8，state_x/state_w 为 fp32，bias/C 为 fp16。
其他存储 dtype 没有被这份具体 profile 自动覆盖。

规格从独立初态读取输入位值，执行以下机器运算：每个输入字节按有符号二补码解释；
每个 block 的 dot 和每次 accumulator 更新取低 32 位再作有符号解释；最终 int32
转换到 fp32，按 `w * (x * (acc * divfactor))` 顺序逐次 fp32 舍入；再直接舍入到
fp16；有 bias 时执行一次直接 fp16 加法。没有无溢出、有限数、结合律、恒等舍入或
预先假设输出的条件。软件模型使用 IEEE RNE、保留 subnormal，NaN 采用模型规定的
canonical 结果。

`Proofs/FPSource.lean` 保留原地址前缀、循环顺序、五条循环语句与最终存储/遮罩，
显式补回旧端口省略的 int8/int32 宽度。两处原 `.real` scale load 改为实际 `.fp32`
存储 load，再保留到计算寄存器的 cast。Python 的运行时 `divfactor` 用初态 `.fp32`
标量寄存器承载，不能把理想实数常量直接视为其机器值。`FPStorage.lean` 证明这些
具体 AST 形状、实际输入字节的解释与 dot 操作数范围；`FPReference.lean` 的独立
递推不是从最终状态反读结果。

公开结论同时给出成功执行、每个活动 lane 的输出读回、全部未写地址保持和输入位
桥接。正常输入/旧配置条件仍是 K 整块、pid1=0、运行时参数绑定、实际字节存储、
写地址不冲突；原内核的无遮罩 load 仍要求实际分配涵盖所访问地址。这些条件不列为
待补数值实验。

`int8_dequant_matmul_bit_success` 给出真实允许配置 BM=16、BN=32、BK=32、两轮 K
的非空见证，两条 bias 分支都保留。M=15、N=31；bias 缓冲提供 32 个可读元素，
因此原来的无遮罩 bias load 也有效。A 的字节 0x80 表示 -128，B 的 0x7f 表示127，
scale 为 fp32 1；divfactor=0x38820610。无 bias 输出半精度 0xd408；bias=0x3555
时输出 0xd403。输出偏移31（遮罩列）、495（遮罩行）与 weight[7] 保持，结果确实
区别于精确有理数计算。`divisor_bits` 另外检查 RNE_fp32(1/16129) 的编码；公开规格
允许任意实际传入的 fp32 除数位，不假定 Python/启动器已经传入该位模式。

仍待原生绑定：对具体 Triton/GPU 编译产物核对 int8 dot/int32 更新采用同一模 2^32
语义，int32→fp32、三次有序 fp32 mul、fp32→fp16 与 fp16 add 的舍入、FTZ、NaN
处理相符，并记录真实参数位及实际 operands。软件证明和非空软件位见证不等于已经
验证 GPU；现有实验框架需要该 profile 的 probe/source 绑定，不能直接声称自动支持。

下面保留旧数学入口的说明；其中 identity/placeholder 描述仅适用于旧数学模型，
不适用于上述新机器数入口。

---

# int8_dequant_matmul

- Source file: `int8_dequant_matmul.py` (upstream `data/TritonBench_G_v1/int8_dequant_matmul.py`)
- Corpus: TritonBench-G v1
- Size: 211 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `Int8DequantMatmul.lean`, main theorem
  `int8_dequant_matmul_exec_genuine` (`exec`-level, dimension-general, 0 `sorry`).

Target JIT = `_int8_matmul_rowwise_dequantize`, the file's only `@triton.jit`
kernel (launcher `int8_matmul_rowwise_dequantize`). An int8×int8→int32 GEMM
with a rowwise-dequantization epilogue: `A`, `B` are signed `torch.int8`
tensors (modelled as `Region .int`), the K loop accumulates the exact ℤ
matmul through the integer `tl.dot` — this port is the **first consumer of
`Op.dotInt`** — and the epilogue rescales each output cell as
`w_factor * (x_factor * (acc * divfactor))` (per-row `state_x` scale,
per-column `state_w` scale, host `divfactor = 1/(127·127)`), downcasts to
fp16 and, under `has_bias`, adds a per-column fp16 bias before the masked
store into the `torch.float16` output `C`. Standard group-swizzled pid
decomposition, `% M` / `% N` **wrapped** row/col offsets with unmasked
K-loop loads.

Five disclosed surface decisions (the module's `Translation-surface
blocker:` marker):

- **`SPLIT_K` fixed to `1`** — the `tl.store` arm. The autotune table sweeps
  `SPLIT_K ∈ {1, 2, 4, 8, 16}`; the `tl.atomic_add` arm (accumulating into
  the host-zeroed `C` of `pre_hook=init_to_zero("C")`) is dropped with the
  constexpr. Every `* SPLIT_K` factor folds to its `SPLIT_K = 1` value; the
  headline carries the launch fact `s.pids 1 = 0` (grid axis 1 has extent
  `SPLIT_K`), and `pid_z` / `rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)`
  stay in the surface faithfully.
- **`EVEN_K` fixed to `True`** — the unmasked-load arm. `EVEN_K` is the
  `@triton.heuristics` constexpr `K % (BLOCK_K * SPLIT_K) == 0`, so the
  masked `else` arm drops with the constexpr, and the loop bound
  `tl.cdiv(K, BLOCK_K * SPLIT_K)` is the antiquoted binder `numKBlocks`
  with side condition `K = BLOCK_K · numKBlocks` (the loads are unmasked,
  so the trip count must be exact).
- **`has_bias` kept as a genuine `Bool` parameter, both arms modeled** (the
  `matmul_dequantize` `NO_GROUPS` precedent): the host passes `0`/`1`; the
  headline covers both arms in one statement through the guarded bias term
  `if has_bias then bias(col) else 0`.
- **`.to(C.dtype.element_ty)` spelled `.to(tl.float16)`** (the host
  allocates `C` as `torch.float16`): the accumulator downcast compiles to a
  genuine `Op.castFloat real → fp16` and the bias load lands directly on
  the `.fp16` channel, so the terminal store is **`.fp16`-typed** and the
  headline reads the output back at the `MemCell` level
  (`MemCell.of .fp16 (fp16(i8Spec))` — the `matmul_dequantize`
  `matmul_kernel` precedent; the placeholder cast is the identity).
- **`$(n)` literal antiquoting** for integer literals inside index
  arithmetic (a bare literal is inferred `.real` by the DSL).

Every dimension, stride and block size stays symbolic (`GROUP_M` included).
The store mask is over the *unwrapped* output coordinates, so on every lane
it lets through, `row < M` / `col < N` make the offset wraps the identity —
the headline's closed form reads the plain `A[row, ·]` row and `B[·, col]`
column as signed ℤ values, the plain `state_x[row]` / `state_w[col]` scales,
and the bias at the plain `col` (the kernel reads it at the unwrapped `rn`
to begin with). The output readback needs distinct lanes at distinct `C`
addresses (`hInj`); `cAddr_injective` discharges it for a row-major `C`.
