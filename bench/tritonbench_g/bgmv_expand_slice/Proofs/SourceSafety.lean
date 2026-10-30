import bench.tritonbench_g.bgmv_expand_slice.Proofs.AccessSafety

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
open VeriTile.Triton FP.Structural Direct Direct.FPValues Signed.Program
noncomputable section
set_option maxHeartbeats 200000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem loop_safe (M : Algebra A) (h : Call) (s t : State A) (bounds : RegionName → Nat)
    (length : s.regs .int [] "split_n_length"=some (fun _ => 1))
    (safe : Safety.traceSafeList M bounds (body h) (s.setReg "n" .int [] (fun _ => 0)))
    (ran : run M (body h) (s.setReg "n" .int [] (fun _ => 0))=some t) :
    Safety.stepSafe M bounds (loop h) s := by
  change Safety.stepSafe M bounds (.forRangeInt "n" (.constInt 0)
    (.ref .int [] "split_n_length") (.constNat 256) (body h)) s
  rw [Safety.stepSafe_unfold]
  simp only [Safety.safeOp_unfold,evalOp_unfold,length,true_and]
  rw [Safety.rangeIntSafe_unfold]
  simp only [show (256:Nat)≠0 by decide,↓reduceIte,show (0:Int)<1 by decide]
  refine ⟨safe,?_⟩
  rw [ran]
  change Safety.rangeIntSafe M bounds "n" 256 1 256 (body h) t
  rw [Safety.rangeIntSafe_unfold]
  simp


theorem source_safe (M : Algebra A) (R : Reduction A) (h : Call)
    (p : Geometry.Program h) (s : State A)
    (valid : (data s).indices p.batch=-1 ∨ 0≤(data s).indices p.batch)
    (bounds : RegionName → Nat) (indices : p.batch<bounds "Indices")
    (input : (data s).indices p.batch≠-1 →
      ∀ j : Fin h.BK,h.even=Bool.true ∨ j.val<h.K → p.batch*h.K+j.val<bounds "X")
    (weights : (data s).indices p.batch≠-1 →
      ∀ j : Fin h.BK,j.val<h.K →
        ((data s).indices p.batch).toNat*(h.N*h.K)+p.row*h.K+j.val<bounds "W")
    (output : (data s).indices p.batch≠-1 → FPGrid.address h p<bounds "Y") :
    (invocation M R h p.batch p.row).Safe bounds s := by
  change Safety.traceSafeList (machine M R) bounds (code h) (entry h p.batch p.row s)
  rw [complete_code,Safety.traceSafeList_append]
  refine ⟨prelude_safe M R h p.batch p.row s bounds indices,?_⟩
  intro t ran
  have same := Option.some.inj ((preguard M R h p.batch p.row s).property.symm.trans ran)
  subst t
  rw [Safety.traceSafeList_cons]
  refine ⟨?_,fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe_unfold]
  refine ⟨?_,?_⟩
  · change Safety.safeOp (machine M R) none bounds _
      (.boolNot (.eq .int .nil (.ref .int [] "lora_index") (.constInt (-1))))
    simp only [Safety.safeOp_unfold,true_and]
  rw [condition_eval]
  by_cases skipped : (data s).indices p.batch=-1
  · simp [skipped]
  · simp only [skipped,ne_eq,not_false_eq_true,decide_true,↓reduceIte]
    have metadata : (data s).indices p.batch=(((data s).indices p.batch).toNat:Int) :=
      (Int.toNat_of_nonneg (valid.resolve_left skipped)).symm
    have positiveN : 0<h.N := by have := p.rowInside;omega
    rw [complete_active,Safety.traceSafeList_append]
    refine ⟨setup_safe M R h p.batch p.row _ s metadata bounds (input skipped),?_⟩
    intro t ran
    have same := Option.some.inj ((preloop M R h p.batch p.row _ s metadata).property.symm.trans ran)
    subst t
    rw [Safety.traceSafeList_cons]
    refine ⟨?_,fun _ _ => Safety.traceSafeList_nil _ _ _⟩
    exact loop_safe (machine M R) h _ _ bounds
      (preloop_length M R h p.batch p.row _ s metadata positiveN)
      (body_safe M R h p.batch p.row _ s metadata positiveN bounds
        (weights skipped) (output skipped))
      (iteration M R h p.batch p.row _ s metadata).property

#axiomsClean source_safe

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.PublicExecution
