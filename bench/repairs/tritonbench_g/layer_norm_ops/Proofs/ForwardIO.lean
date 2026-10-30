import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.FiniteReference
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

structure ForwardBinding (p : ForwardParameters) (v : ForwardValues α p) (s : State α) : Prop where
  x : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.x (s.pids 0*p.xStride+i.1.val)).read .fp32=forwardX v i
  weight : ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.weight i.1.val).read .fp32=forwardWeight v i
  bias : p.hasBias=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.bias i.1.val).read .fp32=forwardBias v i
  residual : p.hasResidual=Bool.true → ∀ i : TileIndex [p.block],i.1.val<p.columns →
    (s.mem p.residual (s.pids 0*p.residualStride+i.1.val)).read .fp32=forwardResidual v i

theorem forward_binding (A : Algebra α) (p : ForwardParameters) (pid : Nat)
    (v : ForwardValues α p) (s : State α) (hp : s.pids 0=pid)
    (h : (forwardIO A p pid).InputBinding v s) : ForwardBinding p v s := by
  constructor
  · intro i hi
    rw [hp]
    exact h ⟨0,by simp [forwardIO,forwardInputs]⟩ i hi
  · intro i hi
    exact h ⟨1,by simp [forwardIO,forwardInputs]⟩ i hi
  · intro hb i hi
    exact h ⟨2,by simp [forwardIO,forwardInputs]⟩ i ⟨hb,hi⟩
  · intro hr i hi
    rw [hp]
    exact h ⟨3,by simp [forwardIO,forwardInputs]⟩ i ⟨hr,hi⟩

theorem forward_x_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) :
    Forward.loaded A p s p.x (s.pids 0*p.xStride)=forwardLoad A p (forwardX v) := by
  funext i
  unfold Forward.loaded forwardLoad
  by_cases hi : i.1.val<p.columns <;> simp only [hi,if_true,if_false,h.x i]

theorem forward_residual_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) (hr : p.hasResidual=Bool.true) :
    Forward.loaded A p s p.residual (s.pids 0*p.residualStride)=forwardLoad A p (forwardResidual v) := by
  funext i
  unfold Forward.loaded forwardLoad
  by_cases hi : i.1.val<p.columns <;> simp only [hi,if_true,if_false,h.residual hr i]

theorem forward_z_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) : Forward.z A p s=forwardZ A p v := by
  funext i
  unfold Forward.z forwardZ
  rw [forward_x_eq A p v s h]
  by_cases hr : p.hasResidual=Bool.true
  · simp only [hr,if_true,forward_residual_eq A p v s h hr]
  · simp only [hr,Bool.false_eq_true,if_false]

theorem forward_mean_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) : Forward.meanValue A p s=forwardMean A p v := by
  simp only [Forward.meanValue,forwardMean,forward_z_eq A p v s h]

theorem forward_centered_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) : Forward.centered A p s=forwardCentered A p v := by
  funext i
  simp only [Forward.centered,forwardCentered,forward_z_eq A p v s h,forward_mean_eq A p v s h]

theorem forward_inverse_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) : Forward.rstdValue A p s=forwardInverse A p v := by
  simp only [Forward.rstdValue,forwardInverse,Forward.varianceValue,forward_centered_eq A p v s h]

theorem forward_y_eq (A : Algebra α) (p : ForwardParameters) (v : ForwardValues α p)
    (s : State α) (h : ForwardBinding p v s) (i : TileIndex [p.block]) (hi : i.1.val<p.columns) :
    KernelReference.forwardY A p s i=forwardY A p v i := by
  unfold KernelReference.forwardY forwardY
  simp only [forward_z_eq A p v s h,forward_mean_eq A p v s h,
    forward_inverse_eq A p v s h,Forward.unspecified,hi,if_true,h.weight i hi]
  by_cases hb : p.hasBias=Bool.true
  · simp only [hb,if_true,h.bias hb i hi]
  · simp only [hb,Bool.false_eq_true,if_false]

omit [Inhabited α] in
@[simp] theorem mapCell_identity (cell : Cell α) : mapCell id cell=cell := by
  cases cell with | mk d v => cases d <;> rfl

theorem forward_result (A : Algebra α) (p : ForwardParameters) (pid : Nat)
    (admissible : KernelExecution.ForwardAdmissible p)
    (v : ForwardValues α p) (initial : State α) (hin : (forwardIO A p pid).InputBinding v initial) :
    ∃ final,runSchedule ((forwardIO A p pid).schedule v) initial=some final ∧
      (forwardIO A p pid).OutputBinding v (forwardGraph A p v) final ∧
      (forwardIO A p pid).Frame v initial final := by
  let u := (invocation A (forwardKernel p) pid).initial initial
  have hp : u.pids 0=pid := rfl
  have hm : u.mem=initial.mem := Invocation.initial_mem _ _
  have hi : (forwardIO A p pid).InputBinding v u := by
    intro port lane live
    simpa only [hm] using hin port lane live
  have bound := forward_binding A p pid v u hp hi
  obtain ⟨final,ht⟩ := KernelExecution.forward A p u admissible
  refine ⟨final,?_,?_,?_⟩
  · simpa only [forwardIO,runSchedule_singleton,Invocation.run] using ht.execution
  · intro o j live
    rcases o with ⟨o,ho⟩
    have n : o<4 := ho
    interval_cases o
    · change mapCell id (final.mem p.y (pid*p.yStride+j.1.val))=_
      rw [mapCell_identity,←hp,ht.y j live,forward_y_eq A p v u bound j live]
      rfl
    · change mapCell id (final.mem p.residualOut (pid*p.residualOutStride+j.1.val))=_
      rw [mapCell_identity,←hp,ht.residual live.1 j live.2,forward_z_eq A p v u bound]
      rfl
    · change mapCell id (final.mem p.mean pid)=_
      rw [mapCell_identity,←hp,ht.mean live,forward_mean_eq A p v u bound]
      rfl
    · change mapCell id (final.mem p.rstd pid)=_
      rw [mapCell_identity,←hp,ht.rstd,forward_inverse_eq A p v u bound]
      rfl
  · intro r o outside
    rw [←hm]
    apply ht.frame
    refine ⟨?_,?_,?_,?_⟩
    · by_cases hy : r=p.y
      · right
        intro j hj
        simpa only [hp] using (outside.1 ⟨0,by simp [forwardIO,forwardOutputs]⟩ j hj).resolve_left
          (not_not_intro hy)
      · exact Or.inl hy
    · intro hs
      by_cases hy : r=p.residualOut
      · right
        intro j hj
        simpa only [hp] using (outside.1 ⟨1,by simp [forwardIO,forwardOutputs]⟩ j ⟨hs,hj⟩).resolve_left
          (not_not_intro hy)
      · exact Or.inl hy
    · intro hr
      simpa only [hp] using outside.1 ⟨2,by simp [forwardIO,forwardOutputs]⟩ () hr
    · simpa only [hp] using outside.1 ⟨3,by simp [forwardIO,forwardOutputs]⟩ () trivial

#axiomsClean forward_result
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
