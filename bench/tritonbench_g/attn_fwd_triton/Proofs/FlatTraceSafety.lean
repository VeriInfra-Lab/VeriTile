import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatPrologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualAssembly
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualHelperSafety
import bench.tritonbench_g.attn_fwd_triton.Proofs.TraceSafety

/-! Bounded traces for the actual shared-memory translation. The only memory
premises are the six ordinary role windows; read-only overlap is allowed. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatTraceSafety
open VeriTile.Triton Memory Relocation AccessSafety
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem pointers_safe (A : FlatAlloc) (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN BD : Nat) :
    Stmt.TraceSafeList bounds (A.flattenStmts (Pointers.code p BM BN BD)) s := by
  apply list_safe
  simp only [Pointers.code,Pointers.rowExpr,Pointers.keyExpr,Pointers.rowOffsets,Pointers.queryScaleExpr,Pointers.keyScaleExpr,wide,RealPointers.rowExpr,
    RealPointers.queryScaleExpr,RealPointers.keyScaleExpr,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatCongruence.vec0,FlatCongruence.vec1]
  simp [Op.SafeAt.eq_def]

theorem prologue_safe (A : FlatAlloc) (bounds : RegionBounds) (s : BlockState) (p : Parameters) (BM BN BD : Nat)
    (hbQ : QueryBound bounds (config A (Coordinates.configuration s p)) BM BD)
    (hbS : qsbase (config A (Coordinates.configuration s p)) BM+(Coordinates.configuration s p).startM < bounds A.flat) :
    Stmt.TraceSafeList bounds (Coordinates.code p BM BN ++ A.flattenStmts (Pointers.code p BM BN BD) ++ Initialization.code BM BD p.headDim) s := by
  let c := Coordinates.configuration s p
  obtain ⟨sc,hcs,hcv,_⟩ := Coordinates.executes s p BM BN
  have hqsb : sc.regs .int [] "q_scale_offset" = some (Tile.scalar (qsbase c BM)) := by
    change sc.regs .int [] "q_scale_offset" = some (Tile.scalar ((((c.z*p.heads+c.h)*((p.context+BM-1)/BM)) : Nat) : Int))
    rw [show c.z*p.heads+c.h = s.pids 1 from Coordinates.packed_head s p]
    exact hcv.queryScaleBase
  have hksb : sc.regs .int [] "k_scale_offset" = some (Tile.scalar (ksbase c BN)) := by
    change sc.regs .int [] "k_scale_offset" = some (Tile.scalar ((((c.z*p.heads+c.h)*((p.context+BN-1)/BN)) : Nat) : Int))
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
  have window (second : Bool) : SafetyLoop.Bounds bounds c 64 BD
      (ActualSetup.lower c second) (ActualSetup.upper c second) := by
    apply ViewBounds.loop_bounds bounds c BD _ _ hv hp
    simp only [ActualSetup.upper]
    split
    · exact Nat.min_le_right _ _
    · exact Nat.le_of_lt hp.row
  obtain ⟨sp,hpstep,hfixed,hcarry,hcaller,hout,hpf⟩ := ActualPrologue.executes A s p BD
  have hpsafe : Stmt.TraceSafeList bounds (ActualPrologue.code A p BD) s := by
    rw [ActualPrologue.code_eq]
    exact prologue_safe A bounds s p 128 64 BD
      (ViewBounds.query bounds c BD hv hp) (ViewBounds.query_scale bounds c hv hp)
  have hs1 := ActualHelperSafety.call_safe bounds s sp c BD Bool.false
    (Initialization.initial 128 BD) (window Bool.false) hfixed hcarry hcaller
  obtain ⟨a,first,afixed,acarry,aframe⟩ := ActualCall.executes s sp c BD Bool.false
    (Initialization.initial 128 BD) hfixed hcarry hcaller
  have hs2 := ActualHelperSafety.call_safe bounds s a c BD Bool.true
    (ActualCall.result s c BD Bool.false (Initialization.initial 128 BD)) (window Bool.true)
    afixed acarry (ActualMiddle.caller_frame hcaller aframe)
  obtain ⟨sl,second,lfixed,lcarry,lframe⟩ := ActualCall.executes s a c BD Bool.true
    (ActualCall.result s c BD Bool.false (Initialization.initial 128 BD))
    afixed acarry (ActualMiddle.caller_frame hcaller aframe)
  have hop : sl.regs .ptr [128,BD] "O_block_ptr"=some (outputPointer c 128 BD) :=
    ((aframe.trans lframe).regs _ _ "O_block_ptr" (by decide)).trans hout
  change Stmt.TraceSafeList bounds (A.flattenStmts (Verified.code 128 64 BD p Bool.false)) s
  rw [ActualAssembly.flat,List.append_assoc,List.append_assoc]
  apply append_of_run hpsafe hpstep
  apply append_of_run hs1 first
  apply append_of_run hs2 second
  rw [ActualTail.source]
  exact TraceSafety.epilogue_safe bounds sl c 128 BD Bool.false hop lfixed.rows lfixed.context
    (ViewBounds.output bounds c BD hv hp)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatTraceSafety
