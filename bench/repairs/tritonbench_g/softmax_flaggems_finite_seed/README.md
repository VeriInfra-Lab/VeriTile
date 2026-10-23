# softmax_flaggems: finite online seed

This isolated candidate repairs the selected source's two forward online paths.
The selected source and canonical selection remain unchanged.

## Defect and source change

With `(1,32769)`, dimension 1, a first value of `-inf` and all other values zero,
the selected inner online source produces 32769 NaNs. With `(1,8193,2)`,
dimension 1 and the same first-column pattern, its noninner online path produces
8193 NaNs in that column. The initial lane state is `m=-inf, z=0`; the first
`-inf` computes `exp(-inf-(-inf))` and poisons its accumulated denominator.
The independent normalized targets instead assign mass zero to `-inf` and
positive equal mass to the finite zero entries.

The candidate changes exactly two Python AST literals: the two online maxima
start at `-340282346638528859811704183484516925440.0`, FP32 word `0xff7fffff`.
A prefix of negative infinities now keeps a finite maximum and zero mass until
a finite input is reached. Every finite FP32 value is at least this seed. A
zero seed would fail on uniformly very negative finite inputs; that alternative
is explicitly covered by boundary checks and an AST mutant.

All wrapper guards, admitted words, dtypes, FP32 accumulation/casts/stores,
branches, tiles, loops, backward configurations and fusion settings are
unchanged from the selected zero-padding repair. NaNs, positive infinity and
all-negative-infinity columns remain admitted. No input rewriting is performed.

## Independent mathematical and native meanings

`Target.lean` defines `mass(-inf)=0`, `mass(x)=exp(x)` and each probability as
mass divided by the sum of masses. `Target.Defined` requires a finite entry in
each active column. `Target.decodeWord` explicitly accepts finite words and
`0xff800000`; NaNs and positive infinity have no normalized mathematical
interpretation here. The backward target is the conventional
`p[n] * (g[n] - sum(p[j]*g[j]))` on actual finite saved output and upstream
values. `Correct.saved_vjp` identifies it with the normalized softmax VJP when
`p` is the proven forward output.

This mathematical scope is not wrapper admission. No normalized result is
claimed for all-`-inf`, NaN or `+inf` columns. The task-local `Ideal.algebra`
propagates bottom through arithmetic/sums and interprets `exp(-inf)` as zero;
its reduction proofs establish that the reached summands are finite. It does
not silently discard exceptional summands. Exact Real arithmetic here is an
ideal interpretation of the same complete source, not a claim that every
rounded native answer is the exact unrounded ratio.

The native interpretation retains all raw FP32 words, source order, casts,
selected scalar reduction trees, exp lowering through FP32 multiplication by
word `0x3fb8aa3b` and `ex2`, and selected fused operations. Both zero and the
new seed have fixed encodings. Other scalar literals and fixed-arity native
instructions are local ports. Numerical conformance conditions are checked
at actual intermediate operands; no whole-softmax or whole-reduction accuracy
hypothesis replaces the source proof.

## Proof chain and public results

`Kernels.lean` contains complete readable quotations of all four kernels.
The source checker compares all four full ASTs, and checks the complete Python
module against the selected source with only the two seed substitutions.
The reviewed normalization expands `prev_multiple_of`, uses the existing
bounded Nat projection for int64 address expressions, specializes admitted
FP32 casts, makes the store cast explicit and normalizes tuple/list shapes,
keyword order and an unused loop variable. The int32 mutant is rejected.

The `FPForward*`, `FPBackward*` and `FPOneTile` proofs construct complete
execution of each quotation for every branch and loop. `ExtendedModel`,
`RealForward*` and `RealBackward*` connect those executions to independent mass
ratios and VJPs. `TraceReadback` and `FPTrace*` derive exact active output
coverage and preserve every cell outside the output region. No run-success,
complete-output-equality or whole-kernel correctness premise is assumed.

The main endpoints are:

- `Correct.checked_forward`: unchanged metadata admission, actual forward
  launch configuration, complete kernel execution, allocated Tensor metadata,
  independent normalized output, a finite saved-output read relation, and frame.
- `Correct.checked_backward`: every one of the six backward configurations,
  actual finite saved-output and upstream loads, allocated Tensor metadata,
  independent VJP and frame. Read/read aliasing remains allowed in the generic
  kernel theorem; destination freshness supplies separation.
- `RealTensor.forward_complete` and `backward_complete`: thin kernel-to-Tensor
  bindings; empty tensors return without launching and preserve all memory.
- `FPEquiv.softmax_autograd_words` and `softmax_autograd_bytes`: the retained
  complete native source path, including all tuning trials, saved-output loads,
  metadata, raw output words, four-byte readback and exterior byte frame.
- `FPEquiv.output_exact_if_local` and `gradient_exact_if_local`: instruction
  composition under local primitive numerical conditions.

New mathematics is limited to kernels and their Tensor/frame bindings. The
existing native host/autograd composition is reused; this candidate adds no
new host/autograd mathematical model or shared library.

The ordinary trust boundaries are the checked Python/DSL transcription,
Tensor descriptor/allocation observations, bounded address projection,
compiler-selected scalar trees/fusion choices, fixed-arity primitive behavior,
and the pinned Lean/import object identities. This is not a verified Triton,
LLVM, PyTorch or CUDA implementation.

## Validation and evidence

The checker reports all four full AST matches and eight rejected source
mutants. It reuses 22 forward/backward regression cases, 21 rejections and five
historical counterexamples, covering 51 launches. Forty added boundary cases
cover inner/noninner and one-tile/online paths with early/late `-inf`, very
negative finite values including FP32 minimum, a full `-inf` prefix followed
by finite values, all `-inf`, NaN and `+inf`. Defined outputs are compared with
an independent shifted ratio and Torch softmax; all six backward configurations
are checked. Input words are checked for preservation.

Two manual-signature offline SM80 compilations of the actual changed online
JITs pass and contain `FF7FFFFF`, exp2, stores and fused operations. They cover
the two defect shapes; they do not establish universal resource/compilation
success. There was no GPU numerical execution and no claim of universal native
numerical conformance.

Evidence is in `.veritile-cache/softmax-flaggems-finite-seed-evidence/`:
`candidate-report.json`, `compiler/report.json`, `proof-report.json`,
`delivery.json` and `verify_delivery.py`. The final packet pins every candidate
file, successful current source/object receipt, effective import identity,
compiler artifact and 168 preserved historical files. Passing receipts are
reused during packaging; no shared dependency rebuild is required.

To reproduce the CPU source/boundary checks, run:

```sh
python bench/repairs/tritonbench_g/softmax_flaggems_finite_seed/check_candidate.py
```

The permitted build driver is
`.veritile-cache/build-tools/workspace_lean_local_first_v2.py`; build only
changed modules and affected candidate dependencies. Read-only packet checking:

```sh
python .veritile-cache/softmax-flaggems-finite-seed-evidence/verify_delivery.py
```
