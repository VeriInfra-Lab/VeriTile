import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceBackwardSafety

namespace VeriTile.Bench.Repairs.LayerNormTriton.PairFinal
open VeriTile.Triton FP.Structural SourceModel SourceFinalGraph
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def kernel (M N BM BN : Nat) :=
  TypedSource.finalBackwardCandidate "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN

def suffix (M N BM BN : Nat) := (kernel M N BM BN).surfaceBody.drop 5

theorem split (M N BM BN : Nat) : (kernel M N BM BN).surfaceBody =
    SourceFinal.prelude M N BM BN ++ [.forRange "i" 0 M BM (SourceFinal.body M N BM BN)] ++
      suffix M N BM BN := by rfl

theorem masked_setReg {ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) (n : RegName) (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) (s.setReg n d sh v) =
      (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).setReg n d sh v := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons]
    split
    · simpa only [show (s.setReg n d sh v).write r (off i) (val i) =
        (s.write r (off i) (val i)).setReg n d sh v from rfl] using ih (s.write r (off i) (val i))
    · exact ih s

theorem suffix_eq (m : Model α) (s t : State α) (M N BM BN : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t) :
    run (engine m) (suffix M N BM BN) t = run (engine m) (SourceFinal.suffix M N BM BN) t := by
  obtain ⟨hm,hcols,hdw,hdb⟩ := h
  simp [suffix,kernel,TypedSource.finalBackwardCandidate,SourceFinal.suffix,SourceFinal.code,
    TypedSource.finalBackward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hcols,hdw,hdb,masked_setReg]

theorem exact (m : Model α) (s : State α) (M N BM BN : Nat) :
    FP.Structural.exec (engine m) (kernel M N BM BN) s =
      FP.Structural.exec (engine m)
        (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) s := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s M N BM BN
  obtain ⟨b,hb,hbready⟩ := range_refines m s a M N BM BN 0 (zero m BM BN) hready
  change run (engine m) (kernel M N BM BN).surfaceBody s = run (engine m) (SourceFinal.code M N BM BN) s
  rw [split,SourceFinal.code_split,run_append,run_append,run_append,run_append,ha]
  simpa only [bind,Option.bind,run,step,hb] using suffix_eq m s b M N BM BN _ hbready

theorem suffix_safe_iff (m : Model α) (s t : State α) (M N BM BN : Nat)
    (a : Carries α BM BN) (h : Invariant s BM BN a t) (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine m) bounds (suffix M N BM BN) t ↔
      Safety.traceSafeList (engine m) bounds (SourceFinal.suffix M N BM BN) t := by
  obtain ⟨hm,hcols,hdw,hdb⟩ := h
  simp [suffix,kernel,TypedSource.finalBackwardCandidate,SourceFinal.suffix,SourceFinal.code,
    TypedSource.finalBackward,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,columns,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat,hcols,hdw,hdb]

theorem safe_iff (m : Model α) (s : State α) (M N BM BN : Nat) (bounds : RegionName → Nat) :
    TraceSafe (engine m) bounds (kernel M N BM BN) s ↔
      TraceSafe (engine m) bounds
        (TypedSource.finalBackward "DW" "DB" "FINAL_DW" "FINAL_DB" M N BM BN) s := by
  obtain ⟨a,ha,hready⟩ := prelude_refines m s M N BM BN
  obtain ⟨b,hb,hbready⟩ := range_refines m s a M N BM BN 0 (zero m BM BN) hready
  change Safety.traceSafeList _ _ (kernel M N BM BN).surfaceBody s ↔
    Safety.traceSafeList _ _ (SourceFinal.code M N BM BN) s
  rw [split,SourceFinal.code_split]
  simp only [Safety.traceSafeList_append,run_append,ha,Option.bind_some,
    run,step,bind,hb,Option.some.injEq]
  simp [suffix_safe_iff m s b M N BM BN _ hbready bounds]

#axiomsClean exact
#axiomsClean safe_iff

#axiomsClean suffix_eq
end
end VeriTile.Bench.Repairs.LayerNormTriton.PairFinal
