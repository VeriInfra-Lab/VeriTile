# IV-dependent matmul floating-point values

The new public specifications cover the entire old accepted `pre_load` schedule:
`BLOCK_K>0`, `K=BLOCK_K*numKBlocks`, arbitrary matrix dimensions and strides,
all K blocks, and the genuine output edge mask. The old exact headline's fitting
tile is included; the old IO face also admits edge tiles, and that wider mask
scope is retained. Output minor stride is one and major stride is at least the
tile width, exactly the old injectivity conditions. No in-bounds-tile condition
was added. Other four scheduling modes remain outside the previously mechanized
scope; none is silently claimed by this package. All old public/helper theorem
statements and the entire old main file are unchanged.

`FPSource.typed` is the complete accepted body with actual fp32 input, weight and
output storage. This is the original wrapper's normal default fp32 configuration,
confirmed by the explicit native compile signatures; it is not a claim about a
process that changes Torch's global default dtype. The source retains dynamic
`cdiv(K,BLOCK_K)` evaluation, per-iteration pointer recomputation, both K-load
masks with zero fill, and the terminal `accumulator.to(float16)` followed by the
fp32 output tensor's implicit store conversion (made explicit in the typed AST).
`source_projection` establishes the operation/address/control projection to the
full old accepted source. Erasing types alone is not the value proof.

`FPReference` reads only independent initial memory. It computes program-ID
mapping, modulo input gathers, every K iteration and operand mask, dot update,
intermediate half conversion, widened fp32 output and masked stores. It never
runs the kernel or assumes an output value. `execution_memory` proves success and
the complete memory effect without output injectivity or input/output separation.
`Post` then gives active output readback and frame outside the active write set.
Its layout bridge proves injectivity from the original stride conditions. Arbitrary
initial payloads, NaNs, infinities and signed zero are allowed; no extra finite or
non-NaN premise is present. Missing/wrongly typed cells have the interpreter's
specified read behavior; actual allocated and correctly typed native inputs remain
the ordinary input-domain requirement.

Two graphs are kept separate. The source performs dot then an outer fp32 add.
The actual offline compiled graph has explicit accumulator `tt.dot`, even with
`enable_fp_fusion=false`. Its native dot uses `inputPrecision=tf32` and SM90 PTX
`mma.sync...f32.tf32.tf32.f32`. `compiled_dotacc_backend_values` specifies that
graph independently rather than asserting a universal dot+add/dotAcc equivalence.
`Dot`/`DotAcc` callbacks carry all operands, old accumulator where applicable,
dtypes, precision, options, dimensions and shape. `.real` on the promoted
accumulator register denotes this interpreter's register carrier under explicit
fp32 compute precision, not fp64 storage or ideal-real arithmetic.

The scalar source add and both casts are concrete software operations under
explicit `FP.Config`. The fp16 payload occupies the low 16 bits of the fp32 word
carrier; it is decoded as half before widening, not reinterpreted as fp32.
`source_sequential_bits` instantiates dot as an increasing-K sequence of separate
fp32 multiplications and additions from +0, then retains the distinct outer add.
`compiled_sequential_fma_bits` instantiates dotAcc as increasing-K fp32 FMA seeded
by the old accumulator. Both are full parameterized software bit profiles, not
claims that native TF32 MMA has the same reduction order or values. The complete
backend contracts keep the actual native dot boundary explicit.

A nonempty software witness executes two K blocks, nonzero PID=3, dimensions
3x3 with a 2x2 tile, modulo gathers, nonunit major strides and both masked output
axes. It reads back C[12]=20 (fp32 bits 0x41a00000 after half narrowing/widening),
preserves masked C[13] and C[17], and preserves X[12], for both graphs. An additional
kernel-checked conversion vector shows fp32 0x3f801000 rounds through half to
fp32 0x3f800000; negative zero survives. Fixed examples do not upgrade a backend
model to native equivalence.

The native probe extracts the original whole kernel body without changing its
AST; it removes only the external host wrapper/test from the compilation harness.
Triton 3.8, CUDA SM90, BM=BN=BK=32, physical fp32 pointer signatures and
`type="pre_load"` compile successfully with fusion false and true. Both artifacts
retain `arith.truncf` fp32→fp16 and `arith.extf` fp16→fp32; PTX confirms both casts.
Complete source/IR/PTX/compiler metadata and hashes are archived. No GPU was run.
The two compile configurations do not certify every native launch configuration.

Remaining native bindings: actual TF32 MMA input conversion/internal accumulation
and exceptional-value behavior to the explicit dotAcc callback; chosen fp32
rounding/FTZ/NaN policy and the two storage conversions; and the compiler rewrite
between source and accumulator graph if a cross-graph claim is desired. The last
is not assumed as a universal equality awaiting random validation. Native address
bounds, signed-width arithmetic, valid nonzero dimensions, launch shape and
compiler hints are the existing input/configuration boundary, not new numerical
laws. No ideal `round fp16=id` or assumed output relation is used.
