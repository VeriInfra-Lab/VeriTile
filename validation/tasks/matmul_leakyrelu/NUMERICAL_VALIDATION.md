# Pending local numerical validation

Rule: `MATMUL-LEAKYRELU-FP32-K32-DOT-ADD-TO-DOTACC`.
Instance: `matmul_leakyrelu.fp32.tf32.k32.live_acc.all_words.v1`.
Status: deferred; artifact absent; bias gate NOT_RUN; magnitude gate NOT_RUN.
No experiment is claimed to have passed. This is one pending local numerical
rewrite, not a missing source-to-program proof or a whole-output assumption.

The compared instructions are the exact local `[32,32]` source operands with
K32 dot, default TF32 input precision, FP32 dot/accumulator result, the original
FP32-to-real carrier cast and separate FP32 add, versus FP32 seeded dotAcc.
The selected historical backend is SM90/Triton 3.7.1, four warps, three stages.
This metadata identifies the retained profile; no new lowering digest or hardware
certificate is invented. Local results are FP32; the kernel's later FP16 cast and
leaky activation are preserved by the program proof and are outside this rule.

The formal conditional domain remains all admitted FP32 payloads and arbitrary
live accumulator values, including padded zero lanes and exceptional values.
An eventual experiment must explicitly classify NaN/Inf/signed-zero behavior and
reject all-word admission on a mismatch. Finite-only random samples cannot supply
all-word coverage or justify narrowing the existing input domain. Exceptional
classification and any payload policy must be recorded separately from finite
bias statistics. The retained software counterexample shows why no exact IEEE
identity is asserted.

Pending sampling plan: independently sample each a/b operand and incoming
accumulator from quantized `Normal(1,1^2)` tuples at a recorded fixed seed; report
independent replicate means. Add separately reported zero/sparse, subnormal,
large/small exponent, cancellation, ties, accumulator-dominant and padded-tail
strata. Do not pool different distributions or exceptional classifications into
one acceptance number. Retain exact source fragments, selected dot options,
compiler/backend/configuration, seed, input distribution, raw observations and
replay command with the result. Use the same quantized operands for both source
instructions and an FP64 oracle for `acc + sum(a*b)`.

Use the project's two-gates protocol in `experiments/floating_point/README.md`
and `scripts/numerical_gates.py`: local ULP scale from the rounded FP32 golden
result; bias `abs(mean)+5*SE <= 0.05` local ULP across replicate means; magnitude
ratio of peak absolute oracle errors with pass threshold 10 and the documented
POT/bootstrap checks and stopping rule. Both gates must PASS (`passOnly`); smoke,
WARN, INCONCLUSIVE and missing evidence do not admit this rule. Nonfinite oracle
or observed statistics follow the protocol's rejection/classification handling.
These local engineering criteria do not prove exact word equality or an arbitrary
kernel's global error bound.

The strict FP32 slope multiplication commutation is already covered by the shared
concrete-model theorem and is not counted as a deferred experiment. Retained
optional native primitive bindings are separate from this main-rule record.
