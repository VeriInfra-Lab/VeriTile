import bench.repairs.tritonbench_g.triton_matmul.Kernels
import bench.repairs.tritonbench_g.triton_matmul.Proofs.FP
/-! Actual FP16/BK64/stages3 and E4M3FN/BK128/stages4; BM128 BN256 GM8 warps8; live FP32 dotAcc and same-type output cast/store. Every padded lane and incoming carry stays in the independent recurrence. Four exact local numeric ports, full scalar carrier, no finite-only or whole-program numeric premise.
Domain: rank-two matching FP16/E4M3FN inputs on one supported device, arbitrary valid nonnegative strides for both A/B, all logical-negative input flags, zero dimensions/tails/aliases, fresh contiguous same-type C. Tensor preparation is the ordinary coordinate-preserving library contract. Full output, returned C, caller/prepared frames and both original input frames are explicit below. -/
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
open VeriTile.Triton FP.Structural FPValue
noncomputable section

specification matmul_repaired_whole_execution {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (r : Request) (prepared : State α)
    (dimensions : r.inner=r.innerB) (storage : r.storage=r.storageB) :
    ∃ final, host M D r prepared=.ok (r.C,final) :=
  host_executes M D r prepared dimensions storage
#axiomsClean matmul_repaired_whole_execution

specification matmul_repaired_fp_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (Q : Numerics α) (r : Request)
    (law : PrimitiveLaws M D Q r.storage) (before prepared : State α) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host M D r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=stored r.storage (logicalReference Q r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset → ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      AddressSafety r prepared tensor :=
  resolved_fp M D Q r law before prepared tensor A B resolve

#axiomsClean matmul_repaired_fp_correctness


local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete result under the declared sequential-FMA software profile. -/
specification matmul_repaired_software_correctness (rest : Algebra Word) (r : Request) (before prepared : State Word) (tensor : TensorAPI r prepared)
    (A : Fin r.rows → Fin r.inner → Word) (B : Fin r.inner → Fin r.columns → Word)
    (resolve : ResolveNegContract r before prepared tensor A B) :
    ∃ final, host (software rest) sequentialDotAcc r prepared=.ok (r.C,final) ∧
      (∀ i : Fin r.rows, ∀ j : Fin r.columns,
        final.mem r.C (outputOffset r i.val j.val)=stored r.storage (logicalReference (softwareNumerics r.storage) r A B i j)) ∧
      (∀ region offset, ¬Footprint r region offset → final.mem region offset=prepared.mem region offset) ∧
      (∀ region offset, ¬Footprint r region offset → ¬CopyFootprint tensor resolve.copyA resolve.copyB region offset →
        final.mem region offset=before.mem region offset) ∧
      (∀ offset, final.mem resolve.originalA offset=before.mem resolve.originalA offset) ∧
      (∀ offset, final.mem resolve.originalB offset=before.mem resolve.originalB offset) ∧
      AddressSafety r prepared tensor :=
  resolved_fp (software rest) sequentialDotAcc (softwareNumerics r.storage) r
    (software_primitives rest r.storage) before prepared tensor A B resolve

#axiomsClean matmul_repaired_software_correctness

/-- All ordered wrapper outcomes; no successful setup or numerical premise. -/
specification matmul_repaired_wrapper_execution {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (api : WrapperAPI α) (r : Request) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper M D api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,_) => c=r.C :=
  wrapper_executes M D api r before A B apiLaw

/-- Final public native-instantiation entry: exactly four local numerical
fields. WrapperSuccess includes the pre-call frame, full finite ABI/control/
mask/pointer/byte bridge and real successful-launch PID bound. -/
specification matmul_repaired_wrapper_fp_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (D : DotAcc α) (Q : Numerics α) (api : WrapperAPI α)
    (r : Request) (law : PrimitiveLaws M D Q r.storage) (before : State α)
    (A : Fin r.rows → Fin r.inner → α) (B : Fin r.inner → Fin r.columns → α)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper M D api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,final) => WrapperSuccess Q api r before A B c final ∧
        (∀ i : Fin r.rows, ∀ j : Fin r.columns,
          final.mem r.C (outputOffset r i.val j.val)=stored r.storage (logicalReference Q r A B i j)) :=
  wrapper_fp M D Q api r law before A B apiLaw

specification matmul_repaired_wrapper_software_correctness (rest : Algebra Word)
    (api : WrapperAPI Word) (r : Request) (before : State Word)
    (A : Fin r.rows → Fin r.inner → Word) (B : Fin r.inner → Fin r.columns → Word)
    (apiLaw : WrapperAPIContract api r before A B) :
    match wrapper (software rest) sequentialDotAcc api r before with
    | .error e => WrapperFailure api r before e
    | .ok (c,final) => WrapperSuccess (softwareNumerics r.storage) api r before A B c final ∧
        (∀ i : Fin r.rows, ∀ j : Fin r.columns,
          final.mem r.C (outputOffset r i.val j.val)=stored r.storage (logicalReference (softwareNumerics r.storage) r A B i j)) :=
  by
    have h := wrapper_fp (software rest) sequentialDotAcc (softwareNumerics r.storage) api r
      (software_primitives rest r.storage) before A B apiLaw
    cases he : wrapper (software rest) sequentialDotAcc api r before with
    | error e => simpa only [he] using h
    | ok output =>
      rcases output with ⟨c,final⟩
      simpa only [he] using h

#axiomsClean matmul_repaired_wrapper_execution
#axiomsClean matmul_repaired_wrapper_fp_correctness
#axiomsClean matmul_repaired_wrapper_software_correctness

-- Exact conditional host-result interface, not a Spec.FP derivation.
#print PrimitiveLaws
-- Both concrete ideal/bit-software instances are checked in independent helpers;
-- the software dotAcc is ascending FP32 FMA and is not claimed native hardware.
#axiomsClean software_primitives

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair
