import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatPrologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatSource
import bench.tritonbench_g.attn_fwd_triton.Proofs.TraceSafety

/-! Bounded traces for the actual shared-memory translation. The only memory
premises are the six ordinary role windows; read-only overlap is allowed. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatTraceSafety
open VeriTile.Triton Memory Relocation AccessSafety
open VeriTile.Bench.TritonBenchG.AttnFwdCausal
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem pointers_safe (A : FlatAlloc) (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN BD : Nat) :
    Stmt.TraceSafeList bounds (A.flattenStmts (Pointers.code p BM BN BD)) s := by
  apply list_safe
  simp only [Pointers.code,Pointers.rowExpr,Pointers.keyExpr,RealPointers.rowExpr,
    RealPointers.queryScaleExpr,RealPointers.keyScaleExpr,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatCongruence.vec0,FlatCongruence.vec1]
  simp [Op.SafeAt.eq_def]

theorem prologue_safe (A : FlatAlloc) (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN BD : Nat)
    (hbQ : QueryBound bounds (config A (Coordinates.configuration s p)) BM BD)
    (hbS : qsbase (config A (Coordinates.configuration s p)) BM+(Coordinates.configuration s p).startM < bounds A.flat) :
    Stmt.TraceSafeList bounds (Coordinates.code p BM BN ++ A.flattenStmts (Pointers.code p BM BN BD) ++ Initialization.code BM BD p.headDim) s := by
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
  obtain ⟨sp,hps,hpv,hpf⟩ := FlatPointers.executes A sc c BM BN BD hcv.row hcv.queryBase hcv.keyBase
    hcv.valueBase hcv.outputBase hqsb hksb hcv.keyStride hcv.valueStride
  rw [List.append_assoc]
  apply append_of_run (SafetyPrologue.coordinates_safe bounds s p BM BN) hcs
  apply append_of_run (pointers_safe A bounds sc p BM BN BD) hps
  exact SafetyPrologue.initialization_safe bounds sp (config A c) BM BN BD hpv
    ((hpf.regs _ _ "n_ctx" (by simp [Pointers.writes])).trans hcv.context) hbQ hbS

theorem safe (A : FlatAlloc) (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BD : Nat) (reciprocal : Bool)
    (hv : ViewBounds.Windows bounds (parameters p A.flat) (offsets A p)) (hp : ViewBounds.Program (Coordinates.configuration s p)) :
    Stmt.TraceSafeList bounds (A.flattenStmts (Verified.code 128 64 BD p reciprocal)) s := by
  let c := config A (Coordinates.configuration s p)
  have hp : ViewBounds.Program c := ⟨hp.batch,hp.head,hp.row⟩
  let diagonal := decide (p.stage=2)
  have hb1 : SafetyLoop.Bounds bounds c 64 BD (FirstSetup.firstLower c.startM 128 diagonal)
      (FirstSetup.firstUpper c.startM 128 p.context diagonal) := by
    apply ViewBounds.loop_bounds bounds c BD _ _ hv hp
    cases hd : diagonal
    · exact Nat.le_of_lt hp.row
    · exact Nat.min_le_right _ _
  have hb2 := ViewBounds.loop_bounds bounds c BD (c.startM*128) (min ((c.startM+1)*128) p.context) hv hp (Nat.min_le_right _ _)
  obtain ⟨sp,hpstep,hfixed,hcarry,hcaller,hout,hpf⟩ := FlatPrologue.executes A s p 128 64 BD
  have hpsafe := prologue_safe A bounds s p 128 64 BD
    (ViewBounds.query bounds c BD hv hp) (ViewBounds.query_scale bounds c hv hp)
  have hlsafe := SafetyMiddle.safe bounds s sp c 128 64 BD diagonal (Initialization.initial 128 BD)
    (by omega) hfixed hcarry hcaller hb1 hb2
  obtain ⟨sl,hlstep,hlfixed,hlcarry,hlf⟩ := TwoLoops.executes s sp c 128 64 BD diagonal (Initialization.initial 128 BD)
    (by omega) hfixed hcarry hcaller
  have hop : sl.regs .ptr [128,BD] "O_block_ptr" = some (outputPointer c 128 BD) :=
    (hlf.regs _ _ "O_block_ptr" (by decide)).trans hout
  rw [FlatSource.body,List.append_assoc]
  apply append_of_run hpsafe hpstep
  apply append_of_run hlsafe hlstep
  exact TraceSafety.epilogue_safe bounds sl c 128 BD reciprocal hop hlfixed.rows hlfixed.context
    (ViewBounds.output bounds c BD hv hp)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatTraceSafety
