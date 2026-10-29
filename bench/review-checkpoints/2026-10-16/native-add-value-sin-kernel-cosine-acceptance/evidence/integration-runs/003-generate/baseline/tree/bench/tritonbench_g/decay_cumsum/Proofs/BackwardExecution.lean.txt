import bench.tritonbench_g.decay_cumsum.Proofs.BackwardPrelude
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardProjection
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardExecution
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

theorem reference_body (dqi dqo dki dko q k g dg : RegionName)
    (sH DK BT BK : Nat) (scale : ℝ) :
    (RealReference.bwd_decay_global_cumsum_surface dqi dqo dki dko q k g dg sH DK BT BK).toAlgKernel.body =
      BackwardPrelude.referencePrelude dqi dqo dki dko q k g dg sH DK BT BK scale ++
        [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
          (RealReference.bwdIterBodyG DK BT BK)] := by rfl

theorem singleton (st : Stmt) (u : BlockState) :
    stepStmts [st] u = stepStmt st u := by
  simp only [stepStmts]
  cases stepStmt st u <;> rfl


def sourceLoop (DK BT BK : Nat) : Stmt :=
  .forRangeInt "__rev_j"
    (.sub .int .nil (.constInt 0) (.sub .int .nil (.constInt (Int.ofNat BT)) (.constInt (Int.ofNat 1))))
    (.constInt (Int.ofNat 1)) (.constNat 1) (BackwardLoop.sourceBody DK BT BK)

theorem loop (DK BT BK : Nat) (hBT : 0<BT) (s t : BlockState)
    (he : BackwardLoop.norm s = BackwardLoop.norm t) :
    (stepStmts [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
      (RealReference.bwdIterBodyG DK BT BK)] s).map BackwardLoop.norm =
    (stepStmts [sourceLoop DK BT BK] t).map BackwardLoop.norm := by
  simp only [singleton,sourceLoop,stepStmt,BackwardProjection.stop,evalOp,bind,Option.bind_some,
    NumericDType.add,NumericDType.sub,NumericDType.div,Tile.bop,Tile.scalar_data,
    Broadcast.leftIndex,Broadcast.rightIndex,Tile.scalar_data,Nat.div_one]
  have hn : BT-1+1=BT := by omega
  simp only [hn,Int.ofNat_eq_natCast]
  convert BackwardLoop.range DK BT BK 0 s t he using 1 <;> congr 2 <;> omega

/-- The faithful source and internal Nat reference have identical execution
and memory results after deleting their dead, differently typed counters and
the reference's unused cached arange. The existing positive-BT premise is
needed because the historical Nat reference has one iteration at BT=0. -/
theorem region (source gate : Storage) (dqi dqo dki dko q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (hBT : 0<BT) (u : BlockState) :
    (exec (RealReference.bwd_decay_global_cumsum_surface dqi dqo dki dko q k g dg sH DK BT BK).toAlgKernel u).map BackwardLoop.norm =
      (exec (backward source gate Bool.false dqi dqo dki dko q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel u).map
        BackwardLoop.norm := by
  simp only [exec,reference_body dqi dqo dki dko q k g dg sH DK BT BK scale,
    BackwardProjection.original_body,BackwardProjection.canonicalBody,
    ProgramRewrite.append,BackwardPrelude.equivalent]
  cases h : stepStmts (BackwardProjection.prelude dqi dqo dki dko q k g dg sH DK BT BK scale) u
  · rfl
  · simp only [Option.map_some,Option.bind_some]
    apply loop DK BT BK hBT
    simp [BackwardRegisterFrame.setOffs,BackwardLoop.norm_set_dead,BackwardLoop.dead]

theorem flat_body (A : FlatAlloc) (DK BT BK : Nat) :
    A.flattenStmts (RealReference.bwdIterBodyG DK BT BK) =
      RealReference.bwdIterBodyG DK BT BK := by
  simp only [RealReference.bwdIterBodyG,ComputeDType.eraseDType,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    Op.broadcast]

theorem flat_source_body (A : FlatAlloc) (DK BT BK : Nat) :
    A.flattenStmts (BackwardLoop.sourceBody DK BT BK) = BackwardLoop.sourceBody DK BT BK := by
  simp only [BackwardLoop.sourceBody,RealReference.bwdIterBodyG,List.drop,
    List.cons_append,List.nil_append,ComputeDType.eraseDType,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    Op.broadcast]

/-- Flattening preserves the same observational execution correspondence. -/
theorem flat (A : FlatAlloc) (source gate : Storage) (dqi dqo dki dko q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (hBT : 0<BT) (u : BlockState) :
    (exec (A.flattenKernel (RealReference.bwd_decay_global_cumsum_surface dqi dqo dki dko q k g dg sH DK BT BK).toAlgKernel) u).map BackwardLoop.norm =
      (exec (A.flattenKernel (backward source gate Bool.false dqi dqo dki dko q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u).map
        BackwardLoop.norm := by
  have hl : A.flattenStmts [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
      (RealReference.bwdIterBodyG DK BT BK)] =
      [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
        (RealReference.bwdIterBodyG DK BT BK)] := by
    simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,flat_body,
      BackwardProjection.stop]
  have hi : A.flattenStmts [sourceLoop DK BT BK] = [sourceLoop DK BT BK] := by
    simp only [sourceLoop,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,flat_source_body]
  simp only [exec,FlatAlloc.flattenKernel,
    reference_body dqi dqo dki dko q k g dg sH DK BT BK scale,
    BackwardProjection.original_body,BackwardProjection.canonicalBody,
    FlatAlloc.flattenStmts_append,hl,show Stmt.forRangeInt _ _ _ _ _ = sourceLoop DK BT BK from rfl,
    hi,ProgramRewrite.append,BackwardPrelude.flat_equivalent]
  cases h : stepStmts (A.flattenStmts (BackwardProjection.prelude dqi dqo dki dko q k g dg sH DK BT BK scale)) u
  · rfl
  · simp only [Option.map_some,Option.bind_some]
    apply loop DK BT BK hBT
    simp [BackwardRegisterFrame.setOffs,BackwardLoop.norm_set_dead,BackwardLoop.dead]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardExecution
