import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.PairGroupedSource

namespace VeriTile.Bench.Repairs.LayerNormTriton.PairGrouped
open VeriTile.Triton FP.Structural SourceModel SourceGroupedGraph
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range_eq (m : Model α) (s t : State α) (stride M N G B group cur : Nat)
    (a : Carries α B) (h : Invariant s N B group a t) :
    FP.Structural.range (engine m) "row" cur M G (body stride M N G B) t =
      FP.Structural.range (engine m) "row" cur M G (SourceGrouped.body stride M N G B) t := by
  conv_lhs => rw [FP.Structural.range]
  conv_rhs => rw [FP.Structural.range]
  split
  · rfl
  · split
    · obtain ⟨u,hu,hnext⟩ := body_refines m s t stride M N G B group cur a h
      rw [body_eq m s t stride M N G B group cur a h,hu]
      simpa only [bind,Option.bind] using range_eq m (storeDX m s stride N B cur) u
        stride M N G B group (cur+G) (nextCarries m s stride N B group cur a) hnext
    · rfl
termination_by M-cur
decreasing_by omega

theorem exact (m : Model α) (s : State α) (stride M N G B : Nat) :
    FP.Structural.exec (engine m) (kernel stride M N G B) s =
      FP.Structural.exec (engine m)
        (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s stride M N G B
  have hg := hready.2.2.1
  change run (engine m) (kernel stride M N G B).surfaceBody s =
    run (engine m) (SourceGrouped.code stride M N G B) s
  rw [split,SourceGrouped.code_split,run_append,run_append,run_append,run_append,ha]
  simp only [bind,Option.bind,run,step,evalOp_unfold,hg]
  rw [range_eq m s a stride M N G B (s.pids 0) (s.pids 0) (zero m B) hready]

#axiomsClean range_eq
#axiomsClean exact

theorem body_safe_iff (m : Model α) (s t : State α) (stride M N G B group row : Nat)
    (a : Carries α B) (h : Invariant s N B group a t) (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine m) bounds (body stride M N G B)
      (t.setReg "row" .nat [] (fun _ => row)) ↔
    Safety.traceSafeList (engine m) bounds (SourceGrouped.body stride M N G B)
      (t.setReg "row" .nat [] (fun _ => row)) := by
  obtain ⟨hmem,hu,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb⟩ := h
  by_cases same : row=group <;>
    simp [body,kernel,TypedSource.groupedBackwardCandidate,SourceGrouped.body,SourceGrouped.code,
      TypedSource.groupedBackward,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,
      engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
      State.setReg,FP.Structural.store,ComputeDType.eraseDType,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      Region.cast,Option.bind,ofFloat,toFloat,hgroup,hcols,hmask,hdwp,hdbp,hdw,hdb,same]
  all_goals tauto

theorem range_safe_iff (m : Model α) (s t : State α) (stride M N G B group cur : Nat)
    (a : Carries α B) (h : Invariant s N B group a t) (bounds : RegionName → Nat) :
    Safety.rangeSafe (engine m) bounds "row" cur M G (body stride M N G B) t ↔
      Safety.rangeSafe (engine m) bounds "row" cur M G (SourceGrouped.body stride M N G B) t := by
  conv_lhs => rw [Safety.rangeSafe_unfold]
  conv_rhs => rw [Safety.rangeSafe_unfold]
  split
  · rfl
  · split
    · obtain ⟨u,hu,hnext⟩ := body_refines m s t stride M N G B group cur a h
      rw [body_safe_iff m s t stride M N G B group cur a h,
        body_eq m s t stride M N G B group cur a h,hu]
      simp only [Option.some.injEq]
      rw [range_safe_iff m (storeDX m s stride N B cur) u stride M N G B group (cur+G)
        (nextCarries m s stride N B group cur a) hnext bounds]
    · rfl
termination_by M-cur
decreasing_by omega

theorem safe_iff (m : Model α) (s : State α) (stride M N G B : Nat) (bounds : RegionName → Nat) :
    TraceSafe (engine m) bounds (kernel stride M N G B) s ↔
      TraceSafe (engine m) bounds
        (TypedSource.groupedBackward "DX" "DY" "DW" "DB" "X" "W" "Mean" "Rstd" stride M N G B) s := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s stride M N G B
  have hg := hready.2.2.1
  have eq := range_eq m s a stride M N G B (s.pids 0) (s.pids 0) (zero m B) hready
  change Safety.traceSafeList _ _ (kernel stride M N G B).surfaceBody s ↔
    Safety.traceSafeList _ _ (SourceGrouped.code stride M N G B) s
  rw [split,SourceGrouped.code_split]
  simp only [Safety.traceSafeList_append,run_append,ha,Option.bind_some,
    Option.some.injEq,forall_eq']
  have hRange := range_safe_iff m s a stride M N G B (s.pids 0) (s.pids 0) (zero m B) hready bounds
  have hRun : run (engine m)
      [.forRangeDyn "row" (.ref .nat [] "group") (.constNat M) (.constNat G) (body stride M N G B)] a =
    run (engine m)
      [.forRangeDyn "row" (.ref .nat [] "group") (.constNat M) (.constNat G) (SourceGrouped.body stride M N G B)] a := by
    simp only [run,step,evalOp_unfold,hg,bind,Option.bind,eq]
  have hSafe : Safety.traceSafeList (engine m) bounds
      [.forRangeDyn "row" (.ref .nat [] "group") (.constNat M) (.constNat G) (body stride M N G B)] a ↔
    Safety.traceSafeList (engine m) bounds
      [.forRangeDyn "row" (.ref .nat [] "group") (.constNat M) (.constNat G) (SourceGrouped.body stride M N G B)] a := by
    simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,hg,step,bind,Option.bind,eq]
    simp only [hRange]
  simp only [hSafe,hRun]

#axiomsClean exact
#axiomsClean safe_iff

#axiomsClean body_eq
end
end VeriTile.Bench.Repairs.LayerNormTriton.PairGrouped
