import bench.tritonbench_g.layer_norm_ops.Proofs.BackwardValues
import bench.tritonbench_g.layer_norm_ops.Proofs.ForwardIO

namespace VeriTile.Bench.Repairs.LayerNormOps.Interface
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

theorem backward_result (A : Algebra α) (p : BackwardParameters) (pid : Nat)
    (admissible : KernelReadback.Admissible p)
    (v : BackwardValues α p) (initial : State α) (hin : (backwardIO A p pid).InputBinding v initial) :
    ∃ final,runSchedule ((backwardIO A p pid).schedule v) initial=some final ∧
      (backwardIO A p pid).OutputBinding v (backwardGraph A p pid v) final ∧
      (backwardIO A p pid).Frame v initial final := by
  let u := (invocation A (backwardKernel p) pid).initial initial
  have hp : u.pids 0=pid := rfl
  have hm : u.mem=initial.mem := Invocation.initial_mem _ _
  have hi : (backwardIO A p pid).InputBinding v u := by
    intro port lane live
    simpa only [hm] using hin port lane live
  have bound := backward_binding A p pid v u hi
  have interval (row : Nat) (live : rowActive p pid row) :
      Backward.start p u≤row ∧ row<Backward.stop p u := by
    simpa only [Backward.start,Backward.stop,hp] using live
  obtain ⟨final,ht⟩ := KernelExecution.backward A p u admissible
  refine ⟨final,?_,?_,?_⟩
  · simpa only [backwardIO,runSchedule_singleton,Invocation.run] using ht.execution
  · intro o j live
    rcases o with ⟨o,ho⟩
    have n : o<5 := ho
    interval_cases o
    · change rowActive p pid j.1.val ∧ j.2.1.val<p.columns at live
      change mapCell id (final.mem p.dx (j.1.val*p.dxStride+j.2.1.val))=_
      rw [mapCell_identity,ht.dx _ (interval _ live.1).1 (interval _ live.1).2 j.2 live.2,
        backward_dx_eq A p pid v u bound _ live.1]
      rfl
    · change p.recompute=Bool.true ∧ rowActive p pid j.1.val ∧ j.2.1.val<p.columns at live
      change mapCell id (final.mem p.y (j.1.val*p.yStride+j.2.1.val))=_
      rw [mapCell_identity,ht.y live.1 _ (interval _ live.2.1).1 (interval _ live.2.1).2 j.2 live.2.2,
        backward_y_eq A p pid v u bound _ live.2.1 live.1]
      rfl
    · change p.storeDres=Bool.true ∧ rowActive p pid j.1.val ∧ j.2.1.val<p.columns at live
      change mapCell id (final.mem p.dresIn (j.1.val*p.dresInStride+j.2.1.val))=_
      rw [mapCell_identity,ht.residual live.1 _ (interval _ live.2.1).1 (interval _ live.2.1).2 j.2 live.2.2,
        backward_dx_eq A p pid v u bound _ live.2.1]
      rfl
    · change mapCell id (final.mem p.dw (pid*p.columns+j.1.val))=_
      rw [mapCell_identity,←hp,ht.dw j live,backward_partials_eq A p pid v u bound hp]
      rfl
    · change mapCell id (final.mem p.db (pid*p.columns+j.1.val))=_
      rw [mapCell_identity,←hp,ht.db live.1 j live.2,backward_partials_eq A p pid v u bound hp]
      rfl
  · intro r o outside
    rw [←hm]
    apply ht.frame
    refine ⟨?_,?_,?_⟩
    · intro row lower upper
      have live : rowActive p pid row := by
        simpa only [Backward.start,Backward.stop,hp] using And.intro lower upper
      have hr := row_lt p pid row live
      refine ⟨?_,?_,?_⟩
      · by_cases hx : r=p.dx
        · right
          intro j hj
          exact (outside.1 ⟨0,by simp [backwardIO,backwardOutputs]⟩ (⟨row,hr⟩,j) ⟨live,hj⟩).resolve_left
            (not_not_intro hx)
        · exact Or.inl hx
      · intro hy
        by_cases hx : r=p.y
        · right
          intro j hj
          exact (outside.1 ⟨1,by simp [backwardIO,backwardOutputs]⟩ (⟨row,hr⟩,j) ⟨hy,live,hj⟩).resolve_left
            (not_not_intro hx)
        · exact Or.inl hx
      · intro hd
        by_cases hx : r=p.dresIn
        · right
          intro j hj
          exact (outside.1 ⟨2,by simp [backwardIO,backwardOutputs]⟩ (⟨row,hr⟩,j) ⟨hd,live,hj⟩).resolve_left
            (not_not_intro hx)
        · exact Or.inl hx
    · by_cases hw : r=p.dw
      · right
        intro j hj
        simpa only [hp] using (outside.1 ⟨3,by simp [backwardIO,backwardOutputs]⟩ j hj).resolve_left
          (not_not_intro hw)
      · exact Or.inl hw
    · intro hb
      by_cases hdb : r=p.db
      · right
        intro j hj
        simpa only [hp] using (outside.1 ⟨4,by simp [backwardIO,backwardOutputs]⟩ j ⟨hb,hj⟩).resolve_left
          (not_not_intro hdb)
      · exact Or.inl hdb

#axiomsClean backward_result
end
end VeriTile.Bench.Repairs.LayerNormOps.Interface
