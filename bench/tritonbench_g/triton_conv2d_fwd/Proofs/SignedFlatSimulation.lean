import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedPostSimulation

/-! Complete flat-memory execution transport from the internal positive-offset
model to the actual signed-pointer source. Only the existing zero-undef input
condition is used; no pointer-underflow side condition is added. -/
namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedFlatSimulation
open VeriTile.Triton MaskedPointerSimulation SignedBodySimulation
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

private def assignment : Stmt → Bool
  | .assign .. => Bool.true
  | _ => Bool.false

private theorem assignments_undef (R : RoundingModel) (body : List Stmt)
    (onlyAssignments : body.all assignment = Bool.true)
    (s t : BlockState) (run : stepStmtsR R body s = some t) : t.undef = s.undef := by
  induction body generalizing s with
  | nil =>
    simp only [stepStmtsR,Option.some.injEq] at run
    subst t
    rfl
  | cons st rest ih =>
    cases st <;> simp [assignment] at onlyAssignments
    rename_i dtype shape name value
    rw [stepStmtsR,stepStmtR] at run
    cases hv : evalOpR R value s with
    | none => simp [hv,bind,Option.bind] at run
    | some v =>
      simp only [hv,bind,Option.bind] at run
      exact (ih (by simpa [List.all_eq_true,assignment] using onlyAssignments) (s.setReg name dtype shape v) run).trans rfl

private theorem flatten_append (A : FlatAlloc) (xs ys : List Stmt) :
    A.flattenStmts (xs ++ ys) = A.flattenStmts xs ++ A.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons st rest ih => simp only [List.cons_append,FlatAlloc.flattenStmts,ih]

private theorem flatten_source_inner (A : FlatAlloc)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat)
    (tf32 : Bool) :
    A.flattenStmts (SignedPointerModel.inner BHW BIN OF IIFS IHS IWS WIFS WHS WWS
      batch_dim in_height in_width PH PW SH SW tf32) =
      SignedPointerModel.inner BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW tf32 := by
  simp [SignedPointerModel.inner,SignedPointerModel.innerFactored,
    PositivePointerModel.convInnerBody,PositivePointerModel.convInnerBodyFactored,
    PositivePointerModel.conv_newBranch,PositivePointerModel.conv_casts,PositivePointerModel.conv_newAcc,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    FlatAlloc.flattenOp.eq_def]

private theorem flatten_model_inner (A : FlatAlloc)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat)
    (tf32 : Bool) :
    A.flattenStmts (PositivePointerModel.convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
      batch_dim in_height in_width PH PW SH SW tf32) =
      PositivePointerModel.convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW tf32 := by
  simp [SignedPointerModel.inner,SignedPointerModel.innerFactored,
    PositivePointerModel.convInnerBody,PositivePointerModel.convInnerBodyFactored,
    PositivePointerModel.conv_newBranch,PositivePointerModel.conv_casts,PositivePointerModel.conv_newAcc,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    FlatAlloc.flattenOp.eq_def]

theorem execution (R : RoundingModel) (A : FlatAlloc) (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat)
    (s t : BlockState) (zero : ∀ r o, s.undef r o = 0)
    (run : execR R (A.flattenKernel
      (PositivePointerModel.surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups tf32 BHW BIN OF).toAlgKernel) s = some t) :
    ∃ u, execR R (A.flattenKernel
      (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel) s = some u ∧
      u.mem = t.mem := by
  let pre := PositivePointerModel.convPreBody Input Weight in_feat_dim out_feat_dim
    out_height out_width IBS IIFS WOFS groups BHW OF
  let post := PositivePointerModel.convPostBody Input Weight Output batch_dim out_height out_width
    OBS OOFS OHS OWS (in_feat_dim / groups) BHW OF
  let source : List Stmt := [.forRange "h" 0 KH 1 [.forRange "w" 0 KW 1
    [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat BIN)
      (SignedPointerModel.inner BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW tf32)]]]
  let model : List Stmt := [.forRange "h" 0 KH 1 [.forRange "w" 0 KW 1
    [.forRangeDyn "c" (.constNat 0) (.ref .nat [] "in_group_dim") (.constNat BIN)
      (PositivePointerModel.convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW tf32)]]]
  have fs : A.flattenStmts source = source := by
    simp only [source,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
      flatten_source_inner]
  have fm : A.flattenStmts model = model := by
    simp only [model,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
      flatten_model_inner]
  have model_body :
      (PositivePointerModel.surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups tf32 BHW BIN OF).toAlgKernel.body = pre ++ model ++ post := by rfl
  change stepStmtsR R (A.flattenStmts _) s = some t at run
  rw [model_body,flatten_append,flatten_append,fm] at run
  change ∃ u, stepStmtsR R (A.flattenStmts _) s = some u ∧ u.mem = t.mem
  rw [SignedPointerModel.source_body]
  change ∃ u, stepStmtsR R (A.flattenStmts (pre ++ source ++ post)) s = some u ∧ u.mem = t.mem
  rw [flatten_append,flatten_append,fs]
  rw [List.append_assoc,stepStmtsR_append] at run ⊢
  cases hp : stepStmtsR R (A.flattenStmts pre) s with
  | none => simp [hp,bind,Option.bind] at run
  | some p =>
    have unchanged : p.undef = s.undef := assignments_undef R (A.flattenStmts pre)
      (by simp [pre,PositivePointerModel.convPreBody,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,assignment]) s p hp
    have related : Related p p := ⟨AgreeExcept.refl _ p,fun r o => by rw [unchanged]; exact zero r o⟩
    have loops := SignedLoopSimulation.nest R BHW BIN OF IIFS IHS IWS WIFS WHS WWS
      batch_dim in_height in_width PH PW SH SW KH KW tf32
    have last := SignedPostSimulation.postlude R A Input Weight Output batch_dim
      out_height out_width OBS OOFS OHS OWS (in_feat_dim / groups) BHW OF
    obtain ⟨u,hu,hr⟩ := append R loops last p p related t (by simpa [hp,bind,Option.bind] using run)
    exact ⟨u,by simpa [hp,bind,Option.bind] using hu,hr.1.memory⟩

#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.TritonConv2dFwd.SignedFlatSimulation
