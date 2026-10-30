import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeProjection
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeTags
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.ReturnedTensor

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership TensorBinding
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly (load32)
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 20000

theorem target_centers (c : Call) (A : FlatAlloc) (before : Bytes.Memory) :
    TaggedMemory (targetState c A before) := by
  apply grid_centers
  intro off
  simp [initialState,centerLike]

theorem returned_output (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r)
    (closed : ∀ rg,rg∉A.regions → A.extent rg=0)
    (schedule : Raw.Schedule c) (ops : Raw.Ops) (before : Bytes.Memory) (scale : Word)
    (after : Nat → Torch.FP32.Word) (returned : TensorReturn.Tuple)
    (api : TensorReturn.API c r
      (fun addr => ⟨load32 (targetBytes c A (shape_of_admission c A p r a) schedule ops before scale) addr⟩)
      after returned)
    (z h row dim : Nat) (hz : z<c.Z) (hh : h<c.H) (hr : row<c.NQ) (hd : dim<c.D) :
    ∃ idx lane,active c lane ∧ Ownership.address c idx lane=(region .output,c.o.at z h row dim) ∧
      Torch.FP32.logicalValue after returned.output api.outputCopy.destinationLaws
        (index4 c.H (if c.finish=0 then c.rounded else c.NQ) c.D z h row dim)=
        ⟨projectLane c (shape_of_admission c A p r a) schedule ops scale
          (targetState c A before).mem idx lane⟩ := by
  obtain ⟨idx,lane,live,coordinate⟩ := TensorBinding.output_coordinate c A p r a z h row dim hz hh hr hd
  have index := output_index c A p r a z h row dim hz hh (query_within_workspace c row hr) hd
  refine ⟨idx,lane,live,?_,?_⟩
  · simpa only [index.2] using coordinate
  · rw [TensorReturn.output_coordinate c r _ after returned api z h row dim hz hh hr hd]
    let shape := shape_of_admission c A p r a
    let words := wordMemory c shape schedule ops scale (targetState c A before).mem
    have readback := Bytes.tensor_readback c A p r a closed words before .output
      (index4 c.H c.rounded c.D z h row dim) index.1 idx lane live coordinate
    have projected := wordMemory_readback c (Contract.safety a.checked).valid shape
      schedule ops scale (targetState c A before).mem idx lane live
    exact readback.trans (congrArg (fun bits => (⟨bits⟩ : Torch.FP32.Word)) projected)

#axiomsClean target_centers
#axiomsClean returned_output
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Native
