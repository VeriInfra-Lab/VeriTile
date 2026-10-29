# INT8 matmul scalar-cast candidate — further source defect

This sibling candidate changes only 18 runtime scalar/stride conversions from
`x.to(tl.int64)` to `tl.cast(x, tl.int64)`. Program-ID tensor casts, algorithms,
precision, guards, all four producer configurations and all 34 GEMM configurations
are unchanged. The frozen original benchmark, previous repair and its 15 partial
quoted-source proof modules remain untouched.

The candidate fixes the eight previously confirmed ordinary-specialization
failures: both kernels, singleton/non-singleton M, and two existing configurations
each. All eight compile using actual wrapper Python integers and normal
`native_specialize_impl` / `_pack_args`; unit values are still specialized.
Large-K GEMM SPLIT_K=1/2 and the actual large producer configuration also compile.

The full-domain mathematical target is still false. `SOURCE_DEFECT.md` documents
an input admitted by the public **composed** `matmul_quantize_int8` entry:
FP32 input uniformly -127, INT8 B uniformly -128, column scale `2^-31`, and
`M=N=1, K=262144`. The selected SPLIT_K=1 SM80 kernel saturates its INT32 carry
and returns 1 under the documented instructions; the independent mathematical
quantization/GEMM formula gives `127/64`. This is not a floating tolerance issue
or an experiment assumption. The raw helper's separate -128 case is supporting
evidence, not a substitute for producer reachability.

`Counterexample.lean` proves the saturation recurrence, quotient enclosure,
binary16 scale-cell enclosure, exact final scalar conversions and different
mathematical target, with standard-base axiom audits. It is diagnostic evidence;
no new whole-kernel correctness theorem or missing source-to-result connection is
claimed. `check_saturation.py` runs the actual composed admission/wrapper AST up
to producer submission and checks exact integer/scalar arithmetic separately.
`check_candidate.py` checks native specialization and offline compilation. No GPU
run, full runtime replay or claim about which configuration wins autotuning is
made. In particular the failing configuration remains eligible; the evidence
does not claim that every hardware tuner selects it.

The candidate is frozen as **source defect pending further repair**, not as a
completed correctness proof. No K bound or payload restriction was added. No
numerical experiment TODO was created. The retained partial proofs keep their
original, limited model meaning. Evidence and source/object pins are in
`.veritile-cache/task-deliveries/int8_matmul_quantization_scalar_cast/delivery.json`.
