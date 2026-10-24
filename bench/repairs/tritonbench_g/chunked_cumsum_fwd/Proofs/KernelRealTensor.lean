import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelTensorMemory
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelTensorAddress
import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelRealTarget

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor
open VeriTile Triton FP.Structural
open Scalar
noncomputable section
set_option maxHeartbeats 4000000

def realPost (reverseBias : Bool) (upper : UpperBound) (read : Nat→ℝ)
    (f : Buffers c hasBias) (t : State (Term ℝ)) : Nat→ℝ :=
  post (fun x _=>interpret (RealTarget.model reverseBias upper) x) read f t

def RealOutputs (reverseBias : Bool) (upper : UpperBound) (read : Nat→ℝ)
    (f : Buffers c hasBias) (useSoftplus : Bool) (lower : ℝ) (t : State (Term ℝ)) : Prop :=
  ∀ b h k j,b<c.batches → h<c.heads → k<c.chunks → j<c.chunk →
    let off := packed c b h k j
    let a := logicalInputs c (snapshot 0 read f.inputs)
    logical (realPost reverseBias upper read f t) (f.output .prepared).tensor off=
      RealTarget.prepared c a hasBias useSoftplus lower upper b h k j ∧
    logical (realPost reverseBias upper read f t) (f.output .cumulative).tensor off=
      RealTarget.cumulative c a hasBias useSoftplus lower upper b h k j

theorem real_outputs (reverseBias : Bool) (upper : UpperBound) (read : Nat→ℝ)
    (f : Buffers c hasBias) (useSoftplus : Bool) (lower : ℝ) (plan : PrefixPlan c.BH c.BC)
    (t : State (Term ℝ))
    (values : LogicalOutputs (RealTarget.model reverseBias upper) c plan
      (snapshot 0 read f.inputs) hasBias useSoftplus lower (RealTarget.upperWord upper) t) :
    RealOutputs reverseBias upper read f useSoftplus lower t := by
  intro b h k j hb hh hk hj
  have bound := packed_bound c b h k j hb hh hk hj
  have value := values b h k j hb hh hk hj
  have written (r : Output) :
      logical (realPost reverseBias upper read f t) (f.output r).tensor (packed c b h k j)=
        interpret (RealTarget.model reverseBias upper) ((t.mem r.region (packed c b h k j)).read .real) := by
    unfold logical
    rw [output_coordinates (f.output r) _ bound]
    simpa only [Fin.val_zero,Nat.add_zero] using
      post_coordinate (fun x _=>interpret (RealTarget.model reverseBias upper) x) read f t r _ bound 0
  dsimp only
  rw [written .prepared,written .cumulative]
  simp only [Output.region]
  rw [value.1,value.2,
    reference_prepared,reference_cumulative,RealTarget.preparation,RealTarget.cumulative_formula]
  exact ⟨rfl,rfl⟩

def RealFrame (reverseBias : Bool) (upper : UpperBound) (read : Nat→ℝ)
    (f : Buffers c hasBias) (t : State (Term ℝ)) : Prop :=
  (∀ addr,¬SharedWritten f addr → realPost reverseBias upper read f t addr=read addr) ∧
  (∀ input∈f.inputs.all,∀ addr,
    input.storage.base ≤ addr ∧ addr < input.storage.base+4*input.storage.elements →
      realPost reverseBias upper read f t addr=read addr)

theorem real_complete (c : Geometry) (sizes : Sizes c) (reverseBias : Bool) (upper : UpperBound)
    (read : Nat→ℝ) (f : Buffers c hasBias) (plan : PrefixPlan c.BH c.BC)
    (useSoftplus : Bool) (lower : ℝ) :
    ∃ t,Kernel.launch (graphAlgebra (RealTarget.model reverseBias upper) c plan) c hasBias useSoftplus
        (.input lower) (.input (RealTarget.upperWord upper)) (entryState 0 read f)=some t ∧
      RealOutputs reverseBias upper read f useSoftplus lower t ∧
      RealFrame reverseBias upper read f t ∧
      Kernel.Frame c c.grid (entryState 0 read f) t := by
  obtain ⟨t,ran,values,frame⟩ := Scalar.complete_from (RealTarget.model reverseBias upper) c sizes plan
    (snapshot 0 read f.inputs) (entryState 0 read f) (entry_graph 0 read f)
    hasBias useSoftplus lower (RealTarget.upperWord upper)
  exact ⟨t,ran,real_outputs reverseBias upper read f useSoftplus lower plan t values,
    ⟨post_exterior _ read f t,post_input_frame _ read f t⟩,frame⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.Tensor
