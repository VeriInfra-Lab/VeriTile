import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Real
import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.TensorBinding
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.BmmChunkFwd.HostCorrect
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
noncomputable section

structure Result where
  object : Nat
  shape : List Nat
  device : VeriTile.Torch.Device
  dtype : VeriTile.Torch.DType
  pointer : Nat
  state : BlockState

def result (p : Metadata) (a : Direct.Allocation p) (s : BlockState) : Result :=
  ⟨a.object,(shape p).outputShape,(p.tensor .a).device,.floating .fp32,a.pointer,s⟩

theorem python_checks_iff (p : Option Metadata) : checkArguments p=.ok () ↔ Admitted p :=
  checkArguments_iff p

/-- Complete fresh returned tensor over arbitrary Real inputs, for every legal
finite autotuning trace. The final causal clearing pass covers every strict
upper coordinate, including tiles skipped by all arithmetic launches. -/
specification bmm_chunk_fwd_real (p : Metadata) (h : InputContract p) (a : Direct.Allocation p)
    (inputs : Mathematical.Inputs) (trials : List Host.Config)
    (legal : ∀ c∈trials,c∈Host.configurations)
    (emptyTrials : (shape p).outputSize=0 → trials=[])
    (chosen : Host.Config) (hc : chosen∈Host.configurations) :
    ∃ out : Result, checkInputs p=.ok () ∧ Direct.RealRuns p a inputs trials chosen out.state ∧
      out.object=a.object ∧ out.shape=(shape p).outputShape ∧
      out.device=(p.tensor .a).device ∧ out.dtype=.floating .fp32 ∧ out.pointer=a.pointer ∧
      (∀ r,Used p r → out.object≠a.inputObjects r) ∧
      (∀ b ch g r n,b<(shape p).batch → ch<(shape p).chunks → g<(shape p).groups →
        r<(shape p).chunk → n<(shape p).chunk →
        out.state.readMem (Direct.flat p a).flat
          ((Direct.flat p a).addr (call p).Out (Host.returnedOffset (call p) b ch g r n))=
            Mathematical.product p inputs b ch g r n) ∧
      (∀ r off,¬Direct.OutputCell (call p) (Direct.flat p a) r off →
        out.state.mem r off=((Direct.flat p a).flattenState (Direct.realState p inputs)).mem r off) := by
  obtain ⟨final,runs,values,frame⟩ := Direct.real_complete p h a inputs trials legal emptyTrials chosen hc
  exact ⟨result p a final,(checkInputs_iff p).mpr h,runs,rfl,rfl,rfl,rfl,rfl,a.fresh,values,frame⟩

/-- Independent Real formula applied to the original actual Tensor payloads.
The projection supplies the ideal Real carrier interpretation; the native
all-word theorem separately retains actual TF32 arithmetic and raw bytes. -/
specification bmm_chunk_fwd_tensor_real (p : Metadata) (h : InputContract p)
    (a : Direct.Allocation p) (inputs : TensorBinding.Inputs p a)
    (returned : TensorBinding.Output p a) (heap : DirectFP.Heap)
    (project : VeriTile.Torch.FP32.Word → ℝ)
    (chosen : Host.Config) (hc : chosen∈Host.configurations) :
    ∃ out : Result, checkInputs p=.ok () ∧
      Direct.RealRuns p a (TensorBinding.idealInputs p a inputs heap project) [] chosen out.state ∧
      out.object=returned.tensor.object ∧ out.shape=returned.tensor.metadata.shape ∧
      out.device=returned.tensor.metadata.device ∧ out.dtype=returned.tensor.metadata.dtype ∧
      out.pointer=returned.tensor.metadata.dataPtr.toNat ∧
      (∀ b ch g r n,b<(shape p).batch → ch<(shape p).chunks → g<(shape p).groups →
        r<(shape p).chunk → n<(shape p).chunk →
        out.state.readMem (Direct.flat p a).flat
          ((Direct.flat p a).addr (call p).Out (Host.returnedOffset (call p) b ch g r n))=
            Mathematical.product p (TensorBinding.idealInputs p a inputs heap project) b ch g r n) ∧
      (∀ r off,¬Direct.OutputCell (call p) (Direct.flat p a) r off →
        out.state.mem r off=((Direct.flat p a).flattenState
          (Direct.realState p (TensorBinding.idealInputs p a inputs heap project))).mem r off) := by
  obtain ⟨out,checks,runs,obj,sh,dev,dtype,ptr,_,values,frame⟩ := bmm_chunk_fwd_real p h a
    (TensorBinding.idealInputs p a inputs heap project) [] (by simp) (fun _ => rfl) chosen hc
  exact ⟨out,checks,runs,obj.trans returned.object.symm,sh.trans returned.shape.symm,
    dev.trans returned.device.symm,dtype.trans returned.laws.dtype.symm,
    ptr.trans returned.pointer.symm,values,frame⟩

#axiomsClean python_checks_iff
#axiomsClean bmm_chunk_fwd_real
#axiomsClean bmm_chunk_fwd_tensor_real
#specNonCircular Mathematical.product avoiding [Kernels.source,Direct.RealRuns,Direct.finish]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkFwd.HostCorrect
