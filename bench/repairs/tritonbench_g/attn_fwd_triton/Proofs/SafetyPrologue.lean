import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.AccessSafety
import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.Prologue
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealSafetyParts

/-! Memory safety of coordinates, pointers and the two initial loads. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyPrologue
open VeriTile.Triton Memory AccessSafety
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem coordinates_safe (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN : Nat) :
    Stmt.TraceSafeList bounds (Coordinates.code p BM BN) s := by
  apply list_safe
  simp [Coordinates.code,Coordinates.ceilExpr,Op.SafeAt.eq_def]

theorem pointers_safe (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN BD : Nat) :
    Stmt.TraceSafeList bounds (Pointers.code p BM BN BD) s := by
  apply list_safe
  simp [Pointers.code,Pointers.rowExpr,Pointers.keyExpr,RealPointers.rowExpr,
    RealPointers.queryScaleExpr,RealPointers.keyScaleExpr,Op.SafeAt.eq_def]

theorem initialization_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BM BN BD : Nat)
    (hp : Pointers.Registers c BM BN BD s)
    (hctx : s.regs .nat [] "n_ctx" = some (Tile.scalar c.p.context))
    (hbQ : QueryBound bounds c BM BD) (hbS : qsbase c BM+c.startM < bounds c.p.QS) :
    Stmt.TraceSafeList bounds (Initialization.code BM BD c.p.headDim) s := by
  simp only [Initialization.code,RealInit.code,List.take_succ_cons,List.take_zero,List.cons_append,List.nil_append]
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro tm
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro tl
  apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
  intro ta
  apply TraceProofs.assign_cons (query_safe bounds _ c BM BD (by simp [hp.query])
    (by simpa [RealPointers.rows] using hp.rows) (by simp [hctx]) hbQ)
  intro tq
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    exact scale_safe bounds _ c.p.QS (qsbase c BM+c.startM) "Q_scale_ptr"
      (by simpa [queryScalePointer] using hp.queryScale) hbS
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem safe (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN BD : Nat)
    (hbQ : QueryBound bounds (Coordinates.configuration s p) BM BD)
    (hbS : qsbase (Coordinates.configuration s p) BM+(Coordinates.configuration s p).startM < bounds p.QS) :
    Stmt.TraceSafeList bounds (coordinates BM BN BD p ++ pointers BM BN BD p ++ initialization BM BN BD p) s := by
  let c := Coordinates.configuration s p
  obtain ⟨sc,hcs,hcv,_⟩ := Coordinates.executes s p BM BN
  have hqsb : sc.regs .nat [] "q_scale_offset" = some (Tile.scalar (qsbase c BM)) := by
    change sc.regs .nat [] "q_scale_offset" = some (Tile.scalar ((c.z*p.heads+c.h)*((p.context+BM-1)/BM)))
    rw [show c.z*p.heads+c.h = s.pids 1 from Coordinates.packed_head s p]
    exact hcv.queryScaleBase
  have hksb : sc.regs .nat [] "k_scale_offset" = some (Tile.scalar (ksbase c BN)) := by
    change sc.regs .nat [] "k_scale_offset" = some (Tile.scalar ((c.z*p.heads+c.h)*((p.context+BN-1)/BN)))
    rw [show c.z*p.heads+c.h = s.pids 1 from Coordinates.packed_head s p]
    exact hcv.keyScaleBase
  obtain ⟨sp,hps,hpv,hpf⟩ := Pointers.executes sc c BM BN BD hcv.row hcv.queryBase hcv.keyBase
    hcv.valueBase hcv.outputBase hqsb hksb hcv.keyStride hcv.valueStride
  rw [Coordinates.source,Pointers.source,Initialization.source,List.append_assoc]
  apply append_of_run (coordinates_safe bounds s p BM BN) hcs
  apply append_of_run (pointers_safe bounds sc p BM BN BD) hps
  exact initialization_safe bounds sp c BM BN BD hpv
    ((hpf.regs _ _ "n_ctx" (by simp [Pointers.writes])).trans hcv.context) hbQ hbS

theorem setup_safe (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat) (second : Bool) :
    Stmt.TraceSafeList bounds (Setup.code BM BN BD second) s := by
  apply list_safe
  simp [Setup.code,Setup.lowerExpr,Setup.upperExpr,RealSetup.lowerExpr,RealSetup.upperExpr,Op.SafeAt.eq_def]

theorem first_setup_safe (bounds : RegionBounds) (s : BlockState) (BM BN BD : Nat) (diagonal : Bool) :
    Stmt.TraceSafeList bounds (FirstSetup.code BM BN BD diagonal) s := by
  apply list_safe
  simp [FirstSetup.code,FirstSetup.copies,RealFirstSetup.copies,Setup.code,Setup.lowerExpr,Setup.upperExpr,
    RealSetup.lowerExpr,RealSetup.upperExpr,Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.SafetyPrologue
