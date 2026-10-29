# Vector addition: genuine binary64 word and typed route foundation

This isolated three-module package supplies a complete binary64 word decoder
and a task-local typed scalar route interpreter for the 29 successful fp64
frontend profiles. It uses actual `BitVec 64` payloads at fp64 and distinct
`Word RegType` indices; no `.real` carrier or shared dtype edits are used.

`Binary64Words.lean` proves both directions of field packing and extraction for
every 64-bit word, finite-projection classification, and kernel-reduced examples
for signed zeros, subnormals, normal limits, infinities and NaN payloads. Encoding
means exact field packing. Rational-to-binary64 rounding/addition is not supplied.

`FP64TypedCore.lean` preserves all integer widths and signedness, raw bool bytes,
fp16/bf16 input promotion through fp32, direct fp8 input conversion, f64 addition,
and direct output conversion. Bool loads use byte != 0 followed by exact i1-to-f64;
bool stores preserve the signed8 result byte. The independent result formula
matches expression evaluation. Active missing loads and undefined integer-output
conversion remain explicit outcomes. Masked lanes require no invented `other`.
Floating output is total under the explicitly chosen word callbacks.

`FP64Profiles.lean` proves exactly 29 supported ordered pairs among 15 storage
dtypes and checks their expression-dependency route labels against a literal
inventory observed from the frozen TTIR. The labels are postorder dataflow paths,
not an instruction trace. The separate SSA audit checks input/output pointer
roots, absent load `other`, ordered operands, conversion widths, direct narrowing,
fp8 RTNE, and signed-i32 offset/mask graphs in all 29 captured files. It runs no
new frontend compiler, native backend or GPU execution.

Fifteen numerical callback families are explicitly registered in
`evidence/numerical-correspondence.json`; native conformance is pending for every
family. Theorems quantify over those callbacks without adding numerical axioms.
Only integer-output callbacks may return `none`. The complete decoder is concrete,
but it does not provide an arithmetic implementation for those callbacks.

The per-lane result is a requested write, not a memory/grid execution proof.
There is no formal bridge to the shared interpreter, original Python source,
native compiler, physical bytes, addresses, scheduling or host wrapper here.
These are structural gaps and cannot be waived as numerical assumptions.
The two fnuz formats retain their separately recorded frontend rejection gaps.

This package adds **zero** completed dtype/grid profiles and **zero** completed
original programs. Prior logical value/grid progress stays 160/217; all 57
remaining pairs still need their outstanding logical-grid/source work. Whole
original-program acceptance remains 0/183.

The evidence directory binds the three actual Lean exits, named axiom checks,
module axiom audits, noncircularity checks, source/dependency and imported project
olean hashes, raw graph audit, and an actual temporary Git staged whitespace
check. The package is ready for independent review; it is not a completed kernel
acceptance certificate.
