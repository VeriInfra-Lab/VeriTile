import bench.repairs.tritonbench_g.layer_norm_ops.IO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.SourceSafety

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

@[simp] theorem invocation_pid (M : Algebra α) (source : ComputeKernel) (pid : Nat) (s : State α) :
    ((invocation M source pid).initial s).pids 0=pid := rfl

theorem forward_safe (M : Algebra α) (p : ForwardParameters) (pid : Nat)
    (v : ForwardValues α p) (bounds : RegionName → Nat)
    (hb : (forwardIO M p pid).InBounds bounds v) (s : State α) :
    ScheduleSafe bounds ((forwardIO M p pid).schedule v) s := by
  simp only [forwardIO,scheduleSafe_singleton,Invocation.Safe,invocation]
  apply SourceSafety.forward
  · intro j hj
    exact hb.2.1 ⟨0,by simp [forwardIO,forwardInputs]⟩ (j,()) hj
  · intro j hj
    exact hb.2.1 ⟨1,by simp [forwardIO,forwardInputs]⟩ (j,()) hj
  · intro h j hj
    exact hb.2.1 ⟨2,by simp [forwardIO,forwardInputs]⟩ (j,()) ⟨h,hj⟩
  · intro h j hj
    exact hb.2.1 ⟨3,by simp [forwardIO,forwardInputs]⟩ (j,()) ⟨h,hj⟩
  · intro j hj
    exact hb.2.2.1 ⟨0,by simp [forwardIO,forwardOutputs]⟩ (j,()) hj
  · intro h j hj
    exact hb.2.2.1 ⟨1,by simp [forwardIO,forwardOutputs]⟩ (j,()) ⟨h,hj⟩
  · intro h
    exact hb.2.2.1 ⟨2,by simp [forwardIO,forwardOutputs]⟩ () h
  · exact hb.2.2.1 ⟨3,by simp [forwardIO,forwardOutputs]⟩ () trivial

theorem backward_safe (M : Algebra α) (p : BackwardParameters) (pid : Nat)
    (v : BackwardValues α p) (bounds : RegionName → Nat)
    (hb : (backwardIO M p pid).InBounds bounds v) (s : State α) :
    ScheduleSafe bounds ((backwardIO M p pid).schedule v) s := by
  simp only [backwardIO,scheduleSafe_singleton,Invocation.Safe,invocation]
  apply SourceSafety.Backward.source_safe
  constructor
  · intro j hj
    exact hb.2.1 ⟨1,by simp [backwardIO,backwardInputs]⟩ (j,()) hj
  · intro hy hbb j hj
    exact hb.2.1 ⟨2,by simp [backwardIO,backwardInputs]⟩ (j,()) ⟨hy,hbb,hj⟩
  · intro row lower upper j hj
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.1 ⟨0,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,j,()) ⟨⟨lower,upper⟩,hj⟩
  · intro row lower upper j hj
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.1 ⟨3,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,j,()) ⟨⟨lower,upper⟩,hj⟩
  · intro hm row lower upper
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.1 ⟨5,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,()) ⟨hm,lower,upper⟩
  · intro row lower upper
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.1 ⟨6,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,()) ⟨lower,upper⟩
  · intro hd row lower upper j hj
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.1 ⟨4,by simp [backwardIO,backwardInputs]⟩ (⟨row,hr⟩,j,()) ⟨hd,⟨lower,upper⟩,hj⟩
  · intro hy row lower upper j hj
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.2.1 ⟨1,by simp [backwardIO,backwardOutputs]⟩ (⟨row,hr⟩,j,()) ⟨hy,⟨lower,upper⟩,hj⟩
  · intro hd row lower upper j hj
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.2.1 ⟨2,by simp [backwardIO,backwardOutputs]⟩ (⟨row,hr⟩,j,()) ⟨hd,⟨lower,upper⟩,hj⟩
  · intro row lower upper j hj
    have hr : row<p.rows := Nat.lt_of_lt_of_le upper (Nat.min_le_right _ _)
    exact hb.2.2.1 ⟨0,by simp [backwardIO,backwardOutputs]⟩ (⟨row,hr⟩,j,()) ⟨⟨lower,upper⟩,hj⟩
  · intro j hj
    exact hb.2.2.1 ⟨3,by simp [backwardIO,backwardOutputs]⟩ (j,()) hj
  · intro h j hj
    exact hb.2.2.1 ⟨4,by simp [backwardIO,backwardOutputs]⟩ (j,()) ⟨h,hj⟩

#axiomsClean forward_safe
#axiomsClean backward_safe
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
