import bench.tritonbench_g.layer_norm_triton.Proofs.SourceGroupedExecution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceSafety.Grouped
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph
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

theorem prelude_safe (m : Model α) (s : State α) (stride M N G B : Nat)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine m) bounds (SourceGrouped.prelude stride M N G B) s := by
  simp [SourceGrouped.prelude,SourceGrouped.code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,engine,numeric,bop,natLt,numericLt,
    State.setReg,Option.bind]

theorem body_safe (m : Model α) (s t : State α) (stride M N G B group row : Nat)
    (a : Carries α B) (h : Invariant s N B group a t)
    (bounds : RegionName → Nat)
    (hx : ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "X")
    (hdy : ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "DY")
    (hw : ∀ j : Fin B,j.val<N → j.val<bounds "W")
    (hm : row<bounds "Mean") (hr : row<bounds "Rstd")
    (hdx : ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "DX") :
    Safety.traceSafeList (engine m) bounds (SourceGrouped.body stride M N G B)
      (t.setReg "row" .nat [] (fun _ => row)) := by
  obtain ⟨hmem,hu,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb⟩ := h
  by_cases same : row=group <;>
    simp [SourceGrouped.body,SourceGrouped.code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,
      engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
      State.setReg,FP.Structural.store,ComputeDType.eraseDType,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb,same,hm,hr]
  · simpa only [same] using And.intro hx (And.intro hdy (And.intro hw (And.intro hm (And.intro hr hdx))))
  · exact ⟨hx,hdy,hw,hdx⟩

theorem range_safe (m : Model α) (s t : State α) (stride M N G B group cur : Nat)
    (a : Carries α B) (h : Invariant s N B group a t)
    (bounds : RegionName → Nat)
    (hx : ∀ row,row<M → ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "X")
    (hdy : ∀ row,row<M → ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "DY")
    (hw : ∀ j : Fin B,j.val<N → j.val<bounds "W")
    (hm : ∀ row,row<M → row<bounds "Mean") (hr : ∀ row,row<M → row<bounds "Rstd")
    (hdx : ∀ row,row<M → ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "DX") :
    Safety.rangeSafe (engine m) bounds "row" cur M G (SourceGrouped.body stride M N G B) t := by
  rw [Safety.rangeSafe_unfold]
  split
  · trivial
  · split
    · rename_i hc
      obtain ⟨u,hu,hnext⟩ := body_refines m s t stride M N G B group cur a h
      refine ⟨body_safe m s t stride M N G B group cur a h bounds
        (hx cur hc) (hdy cur hc) hw (hm cur hc) (hr cur hc) (hdx cur hc),?_⟩
      rw [hu]
      exact range_safe m (storeDX m s stride N B cur) u stride M N G B group (cur+G)
        (nextCarries m s stride N B group cur a) hnext bounds hx hdy hw hm hr hdx
    · trivial
termination_by M-cur
decreasing_by omega

theorem suffix_safe (m : Model α) (s t : State α) (stride M N G B group : Nat)
    (a : Carries α B) (h : Invariant s N B group a t)
    (bounds : RegionName → Nat)
    (hw : ∀ j : Fin B,j.val<N → group*N+j.val<bounds "DW")
    (hb : ∀ j : Fin B,j.val<N → group*N+j.val<bounds "DB") :
    Safety.traceSafeList (engine m) bounds (SourceGrouped.suffix stride M N G B) t := by
  obtain ⟨hmem,hu,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb⟩ := h
  simp [SourceGrouped.suffix,SourceGrouped.code,TypedSource.groupedBackward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb]
  exact ⟨hw,hb⟩

theorem source_safe (m : Model α) (s : State α) (stride M N G B : Nat)
    (bounds : RegionName → Nat)
    (hx : ∀ row,row<M → ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "X")
    (hdy : ∀ row,row<M → ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "DY")
    (hw : ∀ j : Fin B,j.val<N → j.val<bounds "W")
    (hm : ∀ row,row<M → row<bounds "Mean") (hr : ∀ row,row<M → row<bounds "Rstd")
    (hdx : ∀ row,row<M → ∀ j : Fin B,j.val<N → row*stride+j.val<bounds "DX")
    (hdw : ∀ j : Fin B,j.val<N → s.pids 0*N+j.val<bounds "DW")
    (hdb : ∀ j : Fin B,j.val<N → s.pids 0*N+j.val<bounds "DB") :
    TraceSafe (engine m) bounds
      (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s stride M N G B
  have hg := hready.2.2.1
  obtain ⟨b,hbRun,hbready⟩ := SourceGroupedExecution.range_refines m stride M N G B
    (s.pids 0) (s.pids 0) s a (zero m B) hready
  change Safety.traceSafeList _ _ (SourceGrouped.code stride M N G B) s
  rw [SourceGrouped.code_split]
  simp only [List.append_assoc]
  apply append_safe _ _ _ _ s a (prelude_safe m s stride M N G B bounds) ha
  apply append_safe _ _ _ _ a b
  · apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,hg] using
        range_safe m s a stride M N G B (s.pids 0) (s.pids 0) (zero m B) hready bounds hx hdy hw hm hr hdx
    · intro t _
      exact Safety.traceSafeList_nil _ _ _
  · simp [run,step,evalOp_unfold,hg,hbRun]
  · exact suffix_safe m _ b stride M N G B (s.pids 0) _ hbready bounds hdw hdb

end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceSafety.Grouped
