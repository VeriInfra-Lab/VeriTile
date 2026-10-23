import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Contract
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32

namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ReluStridedBuffer HostPlan SignedView
open VeriTile.Bench.TritonBenchG.PowScalarTensor Whole WholeSource
noncomputable section
local instance : Inhabited Float32Bits := ⟨⟨0⟩⟩

def run (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
    (x y : ViewMetadata) (guardSucceeds : Bool) (initial : State Float32Bits) :
    Except Error (Result Float32Bits) := do
  checkInputs Bool.true x y
  return runPythonHost rest abi host Bool.false (buffer x) (buffer y) guardSucceeds initial

/-- Checked actual source, exact supplied output object, all logical values,
entry-relative cell/byte frame and the inherited native-index/pointer proof.
Only ordinary storage provenance and entry values are external bindings. -/
specification wrapper_fp32 (rest : Algebra Float32Bits) (abi : ScalarABI) (host : PythonFloat)
    (x y : ViewMetadata) (h : InputContract Bool.true x y) (storage : StorageBindings x y)
    (initial : State Float32Bits) (xs : Nat → Float32Bits)
    (values : ∀ i,i<count x → initial.mem x.region (((buffer x).view (stride x)).address i)=Cell.mk .fp32 (xs i)) :
    ∃ final,
      run rest abi host x y Bool.true initial=.ok (.returned (buffer y) final) ∧
      (∀ i,i<count x → final.mem y.region (((buffer y).view (stride y)).address i)=
        Cell.mk .fp32 (rest.binary (some .fp32) .real .pow (abi.extractFP32 host.payload) (xs i))) ∧
      (∀ i,i<count x → final.mem x.region (((buffer x).view (stride x)).address i)=
        initial.mem x.region (((buffer x).view (stride x)).address i)) ∧
      (∀ r a,(r≠y.region ∨ ∀ i,i<count x → a≠((buffer y).view (stride y)).address i) →
        final.mem r a=initial.mem r a) ∧
      (∀ r a,(r≠y.region ∨ ∀ i,i<count x → a≠((buffer y).view (stride y)).address i) →
        ∀ byte,byte<4 → memoryByte final r a byte=memoryByte initial r a byte) ∧
      NativeAccess (buffer x) (buffer y) (count x) (stride x) (stride y) (native_buffers _ _ _ h storage) := by
  have vy : ((buffer y).view (stride y)).Valid (count x) (capacity y) := by
    rw [h.shape];exact view_valid y h.output
  obtain ⟨final,he,hv,hin,hf,_⟩ :=
    python_fp32_complete rest abi host Bool.false (buffer x) (buffer y)
      h.positive (stride x) (stride y) (shape_eq x h.input)
      (by simpa [h.shape] using shape_eq y h.output) (strides_eq x h.input) (strides_eq y h.output)
      (view_valid x h.input) vy h.outputStride (footprint_separation _ _ _ h storage)
      initial xs values
  exact ⟨final,by simp [run,(checkInputs_iff _ _ _).mpr h,he],hv,hin,hf,
    fun r a outside byte _ => frame_bytes initial final r a (hf r a outside) byte,
    native_access (buffer x) (buffer y) h.positive h.extent (stride x) (stride y)
      (view_valid x h.input) vy (native_buffers _ _ _ h storage)⟩

#axiomsClean wrapper_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.PowScalarTensor.Checked.FPEquiv
