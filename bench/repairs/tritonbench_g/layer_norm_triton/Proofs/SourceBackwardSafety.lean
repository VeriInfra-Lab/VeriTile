import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution
import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceFinalGraph
import VeriTile.Triton.Float.StructuralSafety

/-! Actual typed source access safety through every backward-loop successor.
The proofs reuse the established exact source execution invariants. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceSafety
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem append_safe (m : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (safe : Safety.traceSafeList m bounds a s) (ran : run m a s=some t)
    (next : Safety.traceSafeList m bounds b t) :
    Safety.traceSafeList m bounds (a++b) s := by
  apply (Safety.traceSafeList_append m bounds a b s).mpr
  refine ⟨safe,?_⟩
  intro u hu
  have eq : u=t := Option.some.inj (hu.symm.trans ran)
  simpa only [eq] using next

namespace Final
open SourceFinalGraph

theorem prelude_safe (m : Model α) (s : State α) (M N BM BN : Nat)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine m) bounds (SourceFinal.prelude M N BM BN) s := by
  simp [SourceFinal.prelude,SourceFinal.code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,engine,numeric,bop,State.setReg,Option.bind]

theorem body_safe (m : Model α) (s t : State α) (M N BM BN cur : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t)
    (bounds : RegionName → Nat)
    (hw : ∀ row, row<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      row*N+(s.pids 0*BN+j.val)<bounds "DW")
    (hb : ∀ row, row<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      row*N+(s.pids 0*BN+j.val)<bounds "DB") :
    Safety.traceSafeList (engine m) bounds (SourceFinal.body M N BM BN)
      (t.setReg "i" .nat [] (fun _ => cur)) := by
  obtain ⟨hm,hcols,hdw,hdb⟩ := h
  simp [SourceFinal.body,SourceFinal.code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,columns,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hcols,hdw,hdb,hm]
  exact ⟨fun row j hr hj => hw (cur+row.val) hr j hj,
    fun row j hr hj => hb (cur+row.val) hr j hj⟩

theorem range_safe (m : Model α) (s t : State α) (M N BM BN cur : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t)
    (bounds : RegionName → Nat)
    (hw : ∀ row,row<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      row*N+(s.pids 0*BN+j.val)<bounds "DW")
    (hb : ∀ row,row<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      row*N+(s.pids 0*BN+j.val)<bounds "DB") :
    Safety.rangeSafe (engine m) bounds "i" cur M BM (SourceFinal.body M N BM BN) t := by
  rw [Safety.rangeSafe_unfold]
  split
  · trivial
  · split
    · obtain ⟨u,hu,hnext⟩ := body_refines m s t M N BM BN cur a h
      refine ⟨body_safe m s t M N BM BN cur a h bounds hw hb,?_⟩
      rw [hu]
      exact range_safe m s u M N BM BN (cur+BM) (next m s M N BM BN cur a) hnext bounds hw hb
    · trivial
termination_by M-cur
decreasing_by omega

theorem suffix_safe (m : Model α) (s t : State α) (M N BM BN : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t)
    (bounds : RegionName → Nat)
    (hw : ∀ j : Fin BN,s.pids 0*BN+j.val<N → s.pids 0*BN+j.val<bounds "FINAL_DW")
    (hb : ∀ j : Fin BN,s.pids 0*BN+j.val<N → s.pids 0*BN+j.val<bounds "FINAL_DB") :
    Safety.traceSafeList (engine m) bounds (SourceFinal.suffix M N BM BN) t := by
  obtain ⟨hm,hcols,hdw,hdb⟩ := h
  simp [SourceFinal.suffix,SourceFinal.code,TypedSource.finalBackward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,columns,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hcols,hdw,hdb]
  exact ⟨hw,hb⟩

theorem source_safe (m : Model α) (s : State α) (M N BM BN : Nat)
    (bounds : RegionName → Nat)
    (hw : ∀ row,row<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      row*N+(s.pids 0*BN+j.val)<bounds "DW")
    (hb : ∀ row,row<M → ∀ j : Fin BN,s.pids 0*BN+j.val<N →
      row*N+(s.pids 0*BN+j.val)<bounds "DB")
    (how : ∀ j : Fin BN,s.pids 0*BN+j.val<N → s.pids 0*BN+j.val<bounds "FINAL_DW")
    (hob : ∀ j : Fin BN,s.pids 0*BN+j.val<N → s.pids 0*BN+j.val<bounds "FINAL_DB") :
    TraceSafe (engine m) bounds (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) s := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s M N BM BN
  obtain ⟨b,hbRun,hbready⟩ := range_refines m s a M N BM BN 0 (zero m BM BN) hready
  change Safety.traceSafeList _ _ (SourceFinal.code M N BM BN) s
  rw [SourceFinal.code_split]
  simp only [List.append_assoc]
  apply append_safe _ _ _ _ s a (prelude_safe m s M N BM BN bounds) ha
  apply append_safe _ _ _ _ a b
  · apply Safety.traceSafeList_cons_intro
    · simpa only [Safety.stepSafe_unfold] using
        range_safe m s a M N BM BN 0 (zero m BM BN) hready bounds hw hb
    · intro t _
      exact Safety.traceSafeList_nil _ _ _
  · simp [run,step,hbRun]
  · exact suffix_safe m s b M N BM BN _ hbready bounds how hob
end Final

end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceSafety
