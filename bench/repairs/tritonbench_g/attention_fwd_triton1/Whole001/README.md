This is a separate FP32 repair of `attention_fwd_triton1`, with a complete
conditional theorem for the public forward call. `AttentionFunction.forward`
still returns only O and accepts both flags. H is internal; its pre-update
snapshots and disabled-store frame are included in the proof.

The accepted inputs are same-shape, same-device CUDA FP32 Q/K/V with raw,
resolved, contiguous dense storage and ordinary valid backing. Inputs may
alias read-only. Unsupported views/lazy bits are rejected. Inputs are never
normalized, copied or padded. Canonical strides are supplied even when
singleton dimensions have unusual reported strides. B/H/T may be zero; T
otherwise has no divisibility restriction. D is any power of two at least 16
subject to representability and ordinary runtime resources.

The kernel retains every floating cast, `allow_tf32=False`, flag branch and
default fusion behavior. Tail Q/K/V loads zero-fill, O stores mask invalid
rows, H uses unit inner stride, and pid is widened before head displacement.
The host uses an NT loop bound and checks padded source coordinates, signed
byte displacements, aligned unsigned address formation and canonical metadata.

Read `ReviewEntry.lean` first. `native_complete` concludes the actual modeled
host return descriptor, complete sourceFP O/H word values, read-only inputs,
unchanged outside words, disabled H stores, empty launch behavior and derived
machine closure. It explicitly takes ordinary host/compiler binding plus
two local scalar bit-conformance families. `source_complete` gives concrete
sourceFP whole-word correctness without numerical premises.
`RealProof.full_output/full_history` establish the
independent input-only real target without any numerical premise.

The target is chunked linear attention, without softmax or token-causal
truncation within the current chunk. Row r includes every token below
`min(T, (r/32+1)*32)` and every feature below D, scaled by `1/sqrt(D)`.
Snapshot c includes KᵀV strictly before row `c*32`.

`Source.step` reads the current memory after the optional H snapshot. Every
grid program consumes the preceding program's returned memory. Memory,
Execution, Loop and Grid prove the physical store/read-only/frame behavior;
the reference recurrence is input-only. Math and RealOutput expand the
ordered scalar-FMA chains to independent truncated sums. Transport proves
expression evaluation into FP32 words and exact reals separately. FPEquiv
connects native words to the concrete FP32 source graph using scalar mulRN and
scalar fmaRN bit conformance, guarded at each actual graph operand. The
reference operations are executable ScalarOps with RNE, gradual underflow,
signed-zero and an explicit canonical-NaN policy. Zero/casts preserve words;
the same actual host/JIT scale word is used by both executions. A closed
reference-self-conformance witness establishes satisfiability for all shapes,
flags and words; it does not establish hardware conformance.

The proof does not validate hardware conformance to those two local bit laws
for arbitrary IEEE inputs. It does not prove the Python/typed-DSL compiler or a raw-byte endian
bridge. `Host.CompilerBinding` explicitly covers compiler/layout/operand
mapping, scheduling, runtime and word-memory effects. The six saved SM80
profiles and SSA chain audit are finite structural evidence, not a universal
compiler or GPU theorem. The 41 actual-source CPU controls use a NumPy
facade and an independent mathematical target; they are diagnostic controls,
not native FMA conformance. No GPU execution is claimed. Public acceptance
neither asserts FP=Real nor requires an exactly representable 1/sqrt(D).
For odd log2(D), including D128, the old exact-real scale equality is impossible
for a finite FP32 decoder. That held endpoint survives only in frozen-v001;
it is not a deferred obligation of this package.

The original T33 out-of-bounds defect is repaired. Independent K-stride,
batch/head-transpose and time/feature-transpose witnesses remain historical
original-source defects; those views are rejected by this new dense contract.
The H stride defect is also repaired in source. The original production file
and acceptance row are unchanged.
