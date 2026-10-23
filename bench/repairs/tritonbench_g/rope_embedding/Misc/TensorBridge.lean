import bench.repairs.tritonbench_g.rope_embedding.Misc.NativeCorrect
import bench.repairs.tritonbench_g.rope_embedding.Correct
import bench.repairs.tritonbench_g.rope_embedding.FPEquiv
import bench.repairs.tritonbench_g.rope_embedding.Proofs.RealResult
import bench.repairs.tritonbench_g.rope_embedding.Proofs.KernelFP
import bench.repairs.tritonbench_g.rope_embedding.Proofs.KernelAccess
import bench.repairs.tritonbench_g.rope_embedding.Proofs.RealBinding

/-! Optional actual-Tensor applications and checked address bounds. -/
namespace VeriTile.Bench.Repairs.RopeEmbedding.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete mathematical kernel correctness from actual Tensor input reads.
The Real state is constructed from the checked data pointers, and the target
uses logical Tensor values through the same explicit input interpretation. -/
theorem rope_embedding_tensor_correct (read : Nat → Numerics.Word)
    (interpret : Numerics.Word → ℝ) (inputs : KernelBinding.Inputs) (r : Request)
    (back : Bool) (scratch : State ℝ)
    (accepted : (KernelBinding.metadata inputs r).check = Bool.true)
    (row group : Nat) (launch : r.Launched row group) :
    let s := RealBinding.state read interpret inputs scratch
    ∃ final, FP.Structural.exec ideal.backend (Source.source (parameters r) back)
        (programState s row group) = some final ∧
      Kernel.Post r row group s final (fun h c => Spec.result r back
        (fun o => interpret (VeriTile.Torch.FP32.logicalValue read inputs.q inputs.qLaws o))
        (fun o => interpret (VeriTile.Torch.FP32.logicalValue read inputs.cosine inputs.cosineLaws o))
        (fun o => interpret (VeriTile.Torch.FP32.logicalValue read inputs.sine inputs.sineLaws o)) row h c) := by
  obtain ⟨final, execution, post⟩ := KernelRealResult.rope_embedding_kernel_correct r back
    (RealBinding.state read interpret inputs scratch) row group launch
  refine ⟨final, execution, ⟨?_, post.frame⟩⟩
  intro h c hlo hhi active
  have hh : h < r.heads := lt_of_lt_of_le hhi (min_le_right _ _)
  have dimension : 2 * r.half ≤ r.dimension := by unfold Request.half; omega
  have values := post.values h c hlo hhi active
  rw [RealBinding.result read interpret inputs r back scratch accepted row h c
    launch.2.1 hh (by omega)] at values
  exact values

/-- The checked endpoint uses the same Tensor-derived Real state and input
formula as above; its finite index/pointer bounds refer to those very pointers.
Bounds include inactive lanes of the padded arange. -/
theorem rope_embedding_checked_kernel_correct (read : Nat → Numerics.Word)
    (interpret : Numerics.Word → ℝ) (inputs : KernelBinding.Inputs) (r : Request)
    (back : Bool) (scratch : State ℝ)
    (accepted : (KernelBinding.metadata inputs r).check = Bool.true)
    (row group : Nat) (launch : r.Launched row group) :
    let s := RealBinding.state read interpret inputs scratch
    (∃ final, FP.Structural.exec ideal.backend (Source.source (parameters r) back)
        (programState s row group) = some final ∧
      Kernel.Post r row group s final (fun h c => Spec.result r back
        (fun o => interpret (VeriTile.Torch.FP32.logicalValue read inputs.q inputs.qLaws o))
        (fun o => interpret (VeriTile.Torch.FP32.logicalValue read inputs.cosine inputs.cosineLaws o))
        (fun o => interpret (VeriTile.Torch.FP32.logicalValue read inputs.sine inputs.sineLaws o)) row h c)) ∧
    (∀ head lane, head < r.heads → lane < r.block →
      r.offset row head (lane + r.half) < 2^31 ∧
      row % r.sequence * r.half + lane < 2^31 ∧
      inputs.q.dataPtr.toNat + 4 * r.offset row head (lane + r.half) + 4 ≤ 2^64 ∧
      inputs.cosine.dataPtr.toNat + 4 * (row % r.sequence * r.half + lane) + 4 ≤ 2^64 ∧
      inputs.sine.dataPtr.toNat + 4 * (row % r.sequence * r.half + lane) + 4 ≤ 2^64) :=
  ⟨rope_embedding_tensor_correct read interpret inputs r back scratch accepted row group launch,
    fun head lane hh hl => KernelAccess.checked_offsets (KernelBinding.metadata inputs r)
      accepted row head lane launch.2.1 hh hl⟩

/-- The same successful relation starts from the actual Tensor data pointers.
Shared Tensor laws and the checked metadata derive every logical input read. -/
theorem rope_embedding_tensor_fp32 (cfg : FP.Config) (r : Request) (back : Bool)
    (read : Nat→Numerics.Word) (inputs : KernelBinding.Inputs)
    (other : RegionName→Nat→Numerics.Word)
    (accepted : (KernelBinding.metadata inputs r).check=Bool.true)
    (row group : Nat) (launch : r.Launched row group) :
    (∃ final, FP.Structural.exec (Numerics.operations (Numerics.softwarePrimitives cfg)).backend
        (Source.source (parameters r) back)
        (programState (KernelFP.initial (KernelBinding.memory read inputs other)) row group)=some final ∧
      KernelFP.words final=KernelFP.nativeMemory (Numerics.softwarePrimitives cfg) r back row group
        (KernelBinding.memory read inputs other)) ∧
    (∀ h c,h < r.heads → c < r.dimension →
      KernelBinding.memory read inputs other "Q" (r.offset row h c)=
        VeriTile.Torch.FP32.logicalValue read inputs.q inputs.qLaws (r.offset row h c)) ∧
    (∀ j,j < r.half →
      KernelBinding.memory read inputs other "C" (row%r.sequence*r.half+j)=
        VeriTile.Torch.FP32.logicalValue read inputs.cosine inputs.cosineLaws (row%r.sequence*r.half+j) ∧
      KernelBinding.memory read inputs other "S" (row%r.sequence*r.half+j)=
        VeriTile.Torch.FP32.logicalValue read inputs.sine inputs.sineLaws (row%r.sequence*r.half+j)) := by
  exact ⟨KernelFP.fp32_successful_equivalence cfg r back (KernelBinding.memory read inputs other)
      row group launch.2.1 launch.2.2,
    fun h c hh hc => KernelBinding.input_binding read inputs r other accepted row h c launch.2.1 hh hc,
    fun j hj => KernelBinding.coefficient_binding read inputs r other accepted row j launch.2.1 hj⟩

#axiomsClean rope_embedding_tensor_correct
#axiomsClean rope_embedding_checked_kernel_correct
#axiomsClean rope_embedding_tensor_fp32
end
end VeriTile.Bench.Repairs.RopeEmbedding.TensorBridge
