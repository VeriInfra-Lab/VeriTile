# int8_matmul_quantization

- Source file: `int8_matmul_quantization.py` (upstream `data/TritonBench_G_v1/int8_matmul_quantization.py`)
- Corpus: TritonBench-G v1
- Size: 267 lines, 2 `@triton.jit` kernels
- Machine value contracts: `FPCorrect.lean`, both original accepted stages.
- Historical mathematical contracts: `Int8MatmulQuantization.lean`.

`int8_matmul_quantization_producer_machine_correctness` proves the complete
original two-pass quantization producer. The first loop computes the fp32
absolute maximum of each row; the second reloads every covered K column,
divides by the separately rounded fp32 scale, converts to signed int8 by
checked truncation toward zero, and stores the values. The scale store
performs a real binary32-to-binary16 encoding conversion. Both loops have
symbolic trip counts and block widths, including ragged and extra fully
masked blocks. The source's unstrided scale offset is retained.

The producer assumes positive block width, sufficient K coverage, active
output address injectivity, and the region distinctions required by the
read-after-write loop and final scale store. Its numeric domain is exactly
that each **active** fp32 quotient has a defined finite signed-int8
conversion. Inactive conversion results are arbitrary: no padding-value
restriction is added, and the stored memory is proved independent of that
arbitrary policy. All-zero rows, NaN or overflowing active quotients fall
outside that defined conversion domain; no saturation behavior is invented.

`int8_matmul_quantization_consumer_machine_correctness` proves the complete
original grouped GEMM consumer for the already accepted `SPLIT_K=1` store
arm. Both int8 operands retain their signed eight-bit interpretation. Every
block dot and accumulator update uses a 32-bit modular word. A separate
lemma equates the accumulator to the low 32 bits of an independent integer
product sum, so overflow is modeled without a no-overflow assumption.
The epilogue rounds the signed32 accumulator into binary32, widens the
binary16 A scale, performs two separate binary32 products using the
binary32 B scale, and rounds the result into binary16 for the masked store.

The consumer covers arbitrary accepted K trip counts and remainder masks,
grouped program IDs, symbolic strides, and both output edge masks. Positive
K-block width, `K <= steps * blockInner`, and `pid[1] = 0` establish that
every original K position occurs in an active block lane. Store-map
injectivity gives exact value readback; the frame theorem preserves all
other cells. The source's `SPLIT_K=2` atomic-add arm is outside the original
accepted scope and is not claimed here.

The typed source keeps the original indexing and control flow. Mechanical
projection lemmas bind the grouped address prefix, all five original K-loop
statements, and the final store addresses/masks. The two signed load widths
and the accumulator width are explicit local AST annotations. The original
`accumulator.to(float32)` event uses `Op.intToReal` in an explicit fp32
context; half-scale widening selects that context without an extra
arithmetic operation. These changes are documented in `MatmulSource.lean`.

Kernel-checked, nonempty examples exercise three K iterations (full,
ragged, and entirely masked). The producer maps `127, -63.5, 31.75` to
`127, -63, 31` and stores half scale `0x3c00`. The consumer uses signed
products and mixed half/fp32 scales to write half `7` (`0x4700`) at a partial
M/N boundary. Both examples prove padding cells unchanged. All new proof
modules run `#axiomsClean`; no native decision axiom is used.

These are concrete software bit-model contracts. They do not establish
GPU/compiler refinement. The remaining native bindings include reduction
and scalar-max NaN/tie/signed-zero selection, scalar rounding/flush policy,
half NaN encoding, integer dot/update overflow lowering, and cast behavior.
No new reassociation or approximate numerical identity is assumed.

The older root module remains an explicitly historical mathematical model:
its producer uses `LegacyRealInt8`, and its consumer uses an unbounded
integer sum and real scaling. Those statements are preserved for source
bridging; they are not presented as machine-value evidence. Its retained
precision scopes are erased only inside the local mathematical evaluator
proofs. The new FP source never imports the legacy cast as its execution
semantics.
