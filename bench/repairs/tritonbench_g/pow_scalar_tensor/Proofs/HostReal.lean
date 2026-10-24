import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Contract
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeReal

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole WholeSource
noncomputable section

def run (rest : Algebra ℝ) (base : ℝ) (x y : ViewMetadata) (guardSucceeds : Bool) (initial : State ℝ) :
    Except Error (Result ℝ) := do
  checkInputs Bool.true x y
  return runHost (realBackend rest) base Bool.false (buffer x) (buffer y) guardSucceeds initial

specification historical_checks_iff_domain (isFloat : Bool) (x y : ViewMetadata) :
    checkInputs isFloat x y=.ok () ↔ InputContract isFloat x y := checkInputs_iff isFloat x y

/-- Independent Real power with the same checked host admission. -/
specification wrapper_real (rest : Algebra ℝ) (base : ℝ)
    (x y : ViewMetadata) (h : InputContract Bool.true x y) (storage : StorageBindings x y)
    (initial : State ℝ) (xs : Nat → ℝ) (mathematical : ∀ i,i<count x → powerDefined base (xs i))
    (values : ∀ i,i<count x → initial.mem x.region (((buffer x).view (stride x)).address i)=Cell.mk .fp32 (xs i)) :
    ∃ final,
      run rest base x y Bool.true initial=.ok (.returned (buffer y) final) ∧
      (∀ i,i<count x → final.mem y.region (((buffer y).view (stride y)).address i)=Cell.mk .fp32 (realResult base xs i)) ∧
      (∀ r a,(r≠y.region ∨ ∀ i,i<count x → a≠((buffer y).view (stride y)).address i) →
        final.mem r a=initial.mem r a) ∧
      NativeAccess (buffer x) (buffer y) (count x) (stride x) (stride y) (native_buffers _ _ _ h storage) := by
  have vy : ((buffer y).view (stride y)).Valid (count x) (capacity y) := by
    rw [h.shape];exact view_valid y h.output
  obtain ⟨final,he,hv,_,hf,_⟩ :=
    real_complete rest base Bool.false (buffer x) (buffer y)
      h.positive (stride x) (stride y) (shape_eq x h.input)
      (by simpa [h.shape] using shape_eq y h.output) (strides_eq x h.input) (strides_eq y h.output)
      (view_valid x h.input) vy h.outputStride (footprint_separation _ _ _ h storage)
      initial xs mathematical values
  exact ⟨final,by simp [run,(checkInputs_iff _ _ _).mpr h,he],hv,hf,
    native_access (buffer x) (buffer y) h.positive h.extent (stride x) (stride y)
      (view_valid x h.input) vy (native_buffers _ _ _ h storage)⟩

#axiomsClean historical_checks_iff_domain
#axiomsClean wrapper_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.Correct
