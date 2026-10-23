import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeReachability
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.NativeTensor
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.TensorFrame
import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.RawCorrect

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RawKernel
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership TensorBinding Native
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly (load32)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Complete source execution over a raw FlatAlloc byte heap. Kernel shape,
layout validity, scalar/fixed-K8 instruction laws and the structural schedule
suffice for all word outputs, supported source values and exterior byte frames. -/
theorem fp_equiv (c : Call) (A : FlatAlloc) (valid : Valid c) (shape : Native.Shape c)
    (schedule : Raw.Schedule c) (actual reference : Raw.Ops) (laws : Raw.LocalLaws actual reference)
    (before : Bytes.Memory) (scale : Native.Word) :
    ∃ final,Reference.Composition.schedule backend.algebra
      (Reference.launched (ScaleBinding.template c)) c.grid (Reference.Composition.allPrograms c.grid)
      (Native.initialState A before)=some final ∧
      Bytes.publish c A (wordMemory c shape schedule actual scale final.mem) before=
        targetBytes c A shape schedule reference before scale ∧
      SupportedMemory (ScaleBinding.template c) final ∧ TaggedMemory final ∧
      (∀ idx lane,active c lane →
        wordMemory c shape schedule actual scale final.mem
          (Ownership.address c idx lane).1 (Ownership.address c idx lane).2=
          projectLane c shape schedule reference scale (targetState c A before).mem idx lane) ∧
      (∀ addr,(¬∃ idx lane byte,active c lane ∧ Physical.address c A idx lane byte=addr) →
        Bytes.publish c A (wordMemory c shape schedule actual scale final.mem) before addr=before addr) := by
  obtain ⟨final,execution,meaning⟩ := Reference.complete_grid (ScaleBinding.template c) backend
    shape.blockN (Native.initialState A before)
  have memory : final.mem=(targetState c A before).mem := meaning
  have publication : Bytes.publish c A (wordMemory c shape schedule actual scale final.mem) before=
      targetBytes c A shape schedule reference before scale := by
    rw [memory]
    exact native_equiv c A shape schedule actual reference laws before scale
  refine ⟨final,execution,publication,?_,?_,?_,?_⟩
  · intro rg off
    rw [memory]
    exact target_supported c A before rg off
  · intro off
    rw [memory]
    exact target_centers c A before off
  · intro idx lane live
    rw [memory,Raw.operations_eq actual reference laws]
    exact wordMemory_readback c valid shape schedule reference scale _ idx lane live
  · intro addr outside
    exact Bytes.frame c A _ before addr outside

#axiomsClean fp_equiv
#specNonCircular Native.targetBytes avoiding [fp_equiv,Kernels.source,FP.Structural.exec,FP.Structural.run]
#specNonCircular Whole.result avoiding [whole_real,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RawKernel
