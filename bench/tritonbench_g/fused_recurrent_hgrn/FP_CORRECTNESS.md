# Finite-precision stage correctness

`FPCorrect.lean` exposes four execution-to-value contracts and four concrete
binary32 software instances, plus a nonempty rounding witness. The public
statements preserve every load, conversion, multiplication, addition and store
in the selected stage, including masked lanes and unchanged memory cells.

The accepted stages are the forward carry step, backward dx step, backward
dx two-step composition, and backward dg step with all three predecessor
branches. Carries are materialized stage inputs. The fp32 storage profile is
explicit. `Proofs/FPShape.lean` proves the erased statement bodies agree with
the original four accepted stage slices; this bridge is structural and does
not discard floating operations from the new execution semantics.

The `forward_software_fp32`, `dx_software_fp32`, `dx_two_software_fp32` and
`dg_software_fp32` declarations use `FP.Software.scalarFP32` and give formulas
containing actual `FP.add`/`FP.mul` on `FP.Value .fp32` bit patterns. Every
operation rounds with the supplied `FP.Config`. Their output formulas do not
depend on any unsupported callback in `rest`.

`forward_nonempty_rounding` uses nearest-even rounding without FTZ/DAZ
(`Config.ieeeValue`), one valid lane and one masked tail. The exact inputs
are `1 * 1 + 2^-25`; the actual binary32 output is `1`, although the exact
rational result differs. It proves successful execution, the stored output,
and the unchanged tail. Kernel evaluation, rather than an external native
decider, checks the scalar bit-pattern calculation.

The native binding obligations are: fp32 input/output/carry storage; memory
and lane/address correspondence; the same scalar rounding and subnormal
configuration; separate multiplication/addition (no unmodeled FMA fusion);
and compiler/runtime realization of these operations. CUDA compiler
conformance is not proved here. Neither are complete recurrent/reverse loop
composition, host allocation/autograd, narrow storage profiles, or equality
to an ideal real-valued recurrence. No identity-rounding premise is used.
