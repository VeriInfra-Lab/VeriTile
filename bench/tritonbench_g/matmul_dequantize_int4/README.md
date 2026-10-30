# matmul_dequantize_int4

`FPCorrect.lean` proves the complete accepted fp16 storage profile with both
`NO_GROUPS` branches and arbitrary K-loop length. Every packed load retains
signed int32 width, with a checked BitVec32 nibble bridge. The independent
initial-state result follows the actual order: zero-point × scale rounds to
binary16, weight × scale rounds to binary16, their subtraction rounds to
binary16, each dot block is added into binary32, and the stored accumulator
is cast to binary16. The source's dead local `c` remains present; the actual
store consumes `accumulator` and performs its destination conversion.

Both branches have success, output readback and outside-write preservation.
The group index remains exactly `k // (groupsize // BK)`; no divisibility
identity or new phase restriction is introduced. The typed dot callback and
compiled scalar contraction/FMA behavior need explicit GPU binding. A concrete
software profile uses separate half operations and a fixed fp32 dot schedule;
it makes no factorization, associativity or `round=id` assumption.

The nonempty bit witness uses 16×16 tiles, BK=16, two K blocks, a masked last
row and negative packed int32 words. `NO_GROUPS=true` produces half `0xd4ab`;
`false` reads a different scale in the second block and produces `0xd8ab`.
Both preserve a destination gap, the masked output row and packed source.
The underlying 32-bit carrier stores half bits in its low 16 bits.

The remaining text describes the older mathematical proof and accepted
configuration boundary, not the floating-point proof.

- Source file: `matmul_dequantize_int4.py` (upstream `data/TritonBench_G_v1/matmul_dequantize_int4.py`)
- Corpus: TritonBench-G v1
- Size: 268 lines, 1 `@triton.jit` kernel
- Status: **PORTED** — `MatmulDequantizeInt4.lean`, main theorem
  `matmul_dequantize_int4_exec_genuine` (`exec`-level, dimension-general,
  0 `sorry`).

`matmul4_kernel` is a GPTQ-style dequantizing GEMM: `C = A · dequant(qweight)`,
where `qweight` packs eight 4-bit weights per 32-bit word along K and `qzeros`
packs eight 4-bit zero-points per word along N. Every K step unpacks the weights
with a shift and a nibble mask, scales them, and subtracts the group's zero-point.

Both configurations of the `NO_GROUPS` flag are covered by the one theorem: it is
the *group row* that differs, and the spec reads `scales` / `zeros` at
`groupRow NO_GROUPS`, which is `0` at every step under the flag and
`k // (group_size // BLOCK_SIZE_K)` otherwise. Every dimension, stride, block size
and the `group_size` stays symbolic.

The output readback needs distinct lanes to have distinct `C` addresses; that is
the theorem's `hInj`, and `cAddr_injective` discharges it for a row-major `C`.

The source's `b.to(a.dtype)` remains an explicit `.real → .real` cast in the
algorithm AST. The exact body decomposition, complete execution theorem,
and full-module axiom audit are replayed with this cast; the public premises
and conclusion are unchanged. Arithmetic still uses the legacy real-valued
model, and the output-readback statement does not include a frame, finite
allocation safety, GPU scheduling, or the Python host launch.
