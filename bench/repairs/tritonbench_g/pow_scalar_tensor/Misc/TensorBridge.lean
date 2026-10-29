import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Contract
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelAccess
import bench.repairs.tritonbench_g.pow_scalar_tensor.Correct
import bench.repairs.tritonbench_g.pow_scalar_tensor.FPEquiv
import bench.repairs.tritonbench_g.pow_scalar_tensor.Misc.NativeObservation

/-! Optional Python-check correspondence. -/

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
noncomputable section
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩

theorem checks_iff_domain (isFloat : Bool) (x y : ViewMetadata) :
    checkInputs isFloat x y = .ok () ↔ InputContract isFloat x y := checkInputs_iff isFloat x y


/-- Independent real power on its ordinary real-valued mathematical domain.
The Python contract remains all-word FP32; powerDefined specifies the Real
interpretation, not an unperformed Python value assertion. Both actual source
variants have the formula, active bounds and complete entry-relative frames.
Both views observe the same physical memory at absolute FP32 cell addresses;
overlapping backing storage objects require no extra premise. No wrapper
execution or returned object is asserted. -/
theorem kernel_correctness (rest : Algebra ℝ) (base : ℝ) (candidate : Bool)
    (x y : ViewMetadata) (h : InputContract Bool.true x y)
    (initial : State ℝ) (xs : Nat → ℝ)
    (_mathematical : ∀ i, i < count x → powerDefined base (xs i))
    (values : ∀ i, i < count x → initial.mem Physical.region (((Physical.buffer x).view (stride x)).address i) = Cell.mk .fp32 (xs i)) :
    Spec.Real (∃ final,
      runGrid (realBackend rest) base candidate
        (rankOneArguments (Physical.buffer x) (Physical.buffer y) (count x) (stride x) (stride y)) initial = some final ∧
      (∀ i, i < count x → final.mem Physical.region (((Physical.buffer y).view (stride y)).address i) =
        Cell.mk .fp32 (Real.rpow base (xs i))) ∧
      (∀ i, i < count x → final.mem Physical.region (((Physical.buffer x).view (stride x)).address i) =
        initial.mem Physical.region (((Physical.buffer x).view (stride x)).address i)) ∧
      (∀ r a, (r ≠ Physical.region ∨ ∀ i, i < count x → a ≠ ((Physical.buffer y).view (stride y)).address i) →
        final.mem r a = initial.mem r a) ∧
      (∀ i, i < count x → Physical.InsideBacking x i ∧ Physical.InsideBacking y i)) := by
  have vy : ((Physical.buffer y).view (stride y)).Valid (count x) Physical.capacity := by
    rw [h.shape]; exact Physical.view_valid y h.output
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := grid_result (realBackend rest) base candidate
    (Physical.buffer x) (Physical.buffer y) h.positive (stride x) (stride y) (Physical.view_valid x h.input) vy
    h.outputStride (Physical.footprint_separation _ _ _ h) initial
  refine ⟨final,ran,?_,inputFrame,frame,(KernelAccess.kernel_native_access x y h).2⟩
  intro i hi
  simp only [Physical.buffer_region] at result
  rw [result i hi,values i hi]
  simp [Cell.read,real_scalar]

/-- Kernel-grid specialization to all FP32 words. baseWord is the scalar word
already supplied by the ordinary launch ABI. The symbolic DSL scalar argument
is interpreted at that word; no FP64-to-FP32 extraction program is a proof
obligation here. Pow remains the unchanged FP32 libdevice primitive slot.
Both views use one physical memory; only the checked active spans separate,
so overlapping backing storage objects and DLPack aliases remain admitted. -/
theorem grid_fp32_equiv (rest : Algebra Float32Bits) (base : ℝ) (baseWord : Float32Bits)
    (x y : ViewMetadata) (h : InputContract Bool.true x y)
    (initial : State Float32Bits) (xs : Nat → Float32Bits)
    (values : ∀ i, i < count x → initial.mem Physical.region (((Physical.buffer x).view (stride x)).address i) = Cell.mk .fp32 (xs i)) :
    ∃ a b,
      runGrid (fp32Backend rest baseWord) base Bool.false
        (rankOneArguments (Physical.buffer x) (Physical.buffer y) (count x) (stride x) (stride y)) initial = some a ∧
      runGrid (fp32Backend rest baseWord) base Bool.true
        (rankOneArguments (Physical.buffer x) (Physical.buffer y) (count x) (stride x) (stride y)) initial = some b ∧
      a.mem = b.mem ∧
      (∀ i, i < count x → a.mem Physical.region (((Physical.buffer y).view (stride y)).address i) =
        Cell.mk .fp32 (rest.binary (some .fp32) .real .pow baseWord (xs i))) ∧
      (∀ i, i < count x → a.mem Physical.region (((Physical.buffer x).view (stride x)).address i) =
        initial.mem Physical.region (((Physical.buffer x).view (stride x)).address i)) ∧
      (∀ r o, (r ≠ Physical.region ∨ ∀ i, i < count x → o ≠ ((Physical.buffer y).view (stride y)).address i) →
        a.mem r o = initial.mem r o ∧ b.mem r o = initial.mem r o) ∧
      (∀ i, i < count x → Physical.InsideBacking x i ∧ Physical.InsideBacking y i) := by
  have vy : ((Physical.buffer y).view (stride y)).Valid (count x) Physical.capacity := by
    rw [h.shape]; exact Physical.view_valid y h.output
  obtain ⟨a,ra,va,ia,fa,ba⟩ := grid_result (fp32Backend rest baseWord) base Bool.false
    (Physical.buffer x) (Physical.buffer y) h.positive (stride x) (stride y) (Physical.view_valid x h.input) vy
    h.outputStride (Physical.footprint_separation _ _ _ h) initial
  obtain ⟨b,rb,vb,ib,fb,bb⟩ := grid_result (fp32Backend rest baseWord) base Bool.true
    (Physical.buffer x) (Physical.buffer y) h.positive (stride x) (stride y) (Physical.view_valid x h.input) vy
    h.outputStride (Physical.footprint_separation _ _ _ h) initial
  have memory : a.mem = b.mem := memory_equal_of_outputs_and_frame (Physical.buffer x) (Physical.buffer y)
    (count x) (stride y) initial a b (fun i hi => (va i hi).trans (vb i hi).symm) fa fb
  refine ⟨a,b,ra,rb,memory,?_,ia,fun r o outside => ⟨fa r o outside,fb r o outside⟩,?_⟩
  · intro i hi
    simp only [Physical.buffer_region] at va
    rw [va i hi,values i hi]
    simp [Cell.read,fp32_scalar]
  · intro i hi
    exact ⟨Physical.backing_bounds x h.input i hi,
      Physical.backing_bounds y h.output i (by simpa [←h.shape] using hi)⟩

#axiomsClean checks_iff_domain
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.TensorBridge
