import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.RealComplete
import VeriTile.Triton.Memory.RegionalKernelIO

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.RealSafety
open VeriTile.Triton RealIndexBridge
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

private theorem body_safe (input output : RegionName) (T BT pid index : Nat)
    (bounds : RegionBounds) (s : BlockState) (ready : Ready pid s)
    (hi : ∀ j,j<T → pid*T+j<bounds input)
    (ho : ∀ j,j<T → pid*T+j<bounds output) :
    Stmt.TraceSafeList bounds (actualBody input output T BT)
      (s.setReg "i_t" .int [] (Tile.scalar (index:Int))) := by
  obtain ⟨hp,hz⟩ := ready
  obtain ⟨carry,hz⟩ := Option.isSome_iff_exists.mp hz
  simp (config := {maxSteps := 300000}) [-Tile.scalar_eta,-Tile.scan_data,-Tile.scan_reverse_data,
    -Tile.reduceSum_false,-Tile.reduceSum_true,-Tile.reduceSumDrop_data,-Tile.reduceSumKeep_data,
    actualBody,pointer,mask,ccLoopBody,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    MemAccess.SafeAt,MaskOpt.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn,Op.MemorySafe.eq_def,MaskOpt.Active,
    stepStmts,stepStmt,evalOp.eq_def,bind,pure,expand_ref,sum_ref,evalOp_expandDim,evalOp_reduceSum,
    Tile.cop,List.mapM,List.mapM.loop,Option.bind,Option.map,
    Tile.bop,Tile.ptrAdd,Tile.uop,NumericDType.add,NumericDType.mul,ComparableDType.lt,
    TileShape.axisDim,Region.cast,BlockState.readMem,BlockState.readMemValue,
    BlockState.readMemTyped,BlockState.paddingValue,BlockState.unspecifiedValue,
    hp,hz,TileShape.indexToList]
  constructor
  · intro i h
    exact hi _ h
  · intro i h
    exact ho _ h

private theorem loop_safe (input output : RegionName) (T BT pid q stop : Nat)
    (bounds : RegionBounds) (s : BlockState) (ready : Ready pid s)
    (hi : ∀ j,j<T → pid*T+j<bounds input)
    (ho : ∀ j,j<T → pid*T+j<bounds output) :
    Stmt.forRangeIntTraceSafe bounds "i_t" q stop 1 (actualBody input output T BT) s := by
  rw [Stmt.forRangeIntTraceSafe,if_neg (by decide : ¬(1:Nat)=0)]
  by_cases active : q<stop
  · rw [if_pos (by exact_mod_cast active)]
    refine ⟨body_safe input output T BT pid q bounds s ready hi ho,?_⟩
    obtain ⟨x,y,hx,_,rel,hy⟩ := RealIndexBridge.body input output T BT pid q s s
      ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩ ready
    rw [hx]
    have readyX : Ready pid x := ⟨rel.2.2.2.2.1.trans hy.1,by rw [rel.2.2.2.2.2]; exact hy.2⟩
    exact_mod_cast loop_safe input output T BT pid (q+1) stop bounds x readyX hi ho
  · rw [if_neg (by exact_mod_cast active)]
    trivial
termination_by stop-q
 decreasing_by omega

/-- Every reached signed-loop body accesses only its displayed active row
windows. Unspecified tail values remain unrestricted. -/
theorem trace_safe (input output : RegionName) (T BT : Nat) (bounds : RegionBounds)
    (s : BlockState) (hi : ∀ j,j<T → s.pids 0*T+j<bounds input)
    (ho : ∀ j,j<T → s.pids 0*T+j<bounds output) :
    Kernel.TraceSafe bounds (actual input output T BT) s := by
  unfold Kernel.TraceSafe
  rw [actual_body]
  simp [-Tile.scalar_eta,prelude,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
    stepStmt,evalOp.eq_def,Option.bind,bind,pure]
  constructor
  · simpa using loop_safe input output T BT (s.pids 0) 0 ((T+BT-1)/BT) bounds
      ((s.setReg "i_bh" .nat [] (Tile.scalar (s.pids 0))).setReg "b_z" .real [] (Tile.scalar (some 0)))
      (by simp [Ready]) hi ho
  · split <;> trivial

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.RealSafety
