import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedCumProgram
import bench.tritonbench_g.chunk_gated_attention.Sources.RebasedAccess

/-! Repaired cumulative tile computation. The existing triangular-dot,
load/store and prefix/frame proofs are instantiated on the actual local tile. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumAccess
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
open RealCumProgram
open ChunkCumsumVector.RealStep (pointer inputTile chunkPrefix address active)
open ChunkCumsumVector (lowerTriTile)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def tileBase (p : Layout) (l : Launch) (c : Nat) :=
  l.head*p.headStride+c*p.BT*p.timeStride+l.feature*p.BS*p.channelStride

def tileLayout (p : Layout) (l : Launch) (c : Nat) : Layout :=
  {p with T := p.BT, S := p.BS, headStride := tileBase p l c}

def localLaunch : Launch := ⟨0,1⟩

theorem pointer_eq (p : Layout) (l : Launch) (c : Nat) (r : RegionName) :
    pointer (tileLayout p l c) localLaunch r 0 =
      RebasedAccess.matrix r (tileBase p l c) p.BT p.BS p.timeStride p.channelStride := by
  simp [pointer,tileLayout,localLaunch,RebasedAccess.matrix,RealHPointers.matrixPointer]

theorem pointer_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (swapped : Bool)
    (u : BlockState) (hs : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (RebasedCumProgram.pointerOp p r swapped) u =
      some (pointer (tileLayout p l c) localLaunch r 0) := by
  rw [pointer_eq]
  apply RebasedAccess.local2_eval
  simp only [evalOp_add,evalOp_mul,RealCumExecution.base_eval p swapped u l.head hh,
    evalOp_ref,evalOp_constNat,hs,ht,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem computes (p : Layout) (l : Launch) (s o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (swapped : Bool) (u : BlockState)
    (hu : ∀ off, u.undef s off = padding off)
    (hs : u.regs .nat [] "i_s" = some (Tile.scalar l.feature))
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (ht : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hm : ∀ off, u.readMem s off = a off)
    (htri : u.regs .real [p.BT,p.BT] "m_s" = some (lowerTriTile p.BT)) :
    stepStmts (RebasedCumProgram.computeCode p s o swapped) u =
      some (RealCumExecution.result (tileLayout p l c) localLaunch s o a padding 0 u) := by
  let q := tileLayout p l c
  simp only [RebasedCumProgram.computeCode,RealCumProgram.computeCode,List.drop,
    List.cons_append,List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l s c swapped u hs hh ht))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l o c swapped _
    (by simp [hs]) (by simp [hh]) (by simp [ht])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (ChunkCumsumVector.RealStep.load_eval q localLaunch s a padding 0 _ (by simpa using hu)
      (by intro off; simp [hm]) (by simp [q,tileLayout])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (RealCumExecution.dot_eval q localLaunch a padding 0 _
    (by simp [q,tileLayout,htri]) (by simp)))]
  erw [stepStmts.cons_some (RealCumExecution.store_eval q localLaunch o a padding 0 _
    (by simp [q,tileLayout]) (by simp)),stepStmts.nil]
  rfl

theorem all_active (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) :
    RealCumExecution.tileActive (tileLayout p l c) localLaunch 0 i := by
  exact ⟨by simpa [RealCumExecution.tileActive,active,tileLayout,localLaunch] using i.1.isLt,
    by simpa [RealCumExecution.tileActive,active,tileLayout,localLaunch] using i.2.1.isLt⟩

/-- Every computed lane reads the physical prefix of this staged tile. -/
theorem readback (p : Layout) (l : Launch) (c : Nat) (s o : RegionName)
    (a padding : Nat → ℝ) (u : BlockState)
    (hinj : RealCumExecution.NoCollision (tileLayout p l c) localLaunch 0)
    (i : TileIndex [p.BT,p.BS]) :
    (RealCumExecution.result (tileLayout p l c) localLaunch s o a padding 0 u).readMem o
      (RealCumExecution.offset (tileLayout p l c) localLaunch 0 i) =
      RealCumExecution.expected (tileLayout p l c) localLaunch 0 a i := by
  let q := tileLayout p l c
  have ha := all_active p l c i
  have hr := BlockState.scatter_readback_prop_masked_nd_of_true (region := o)
    (RealCumExecution.prepared q localLaunch s o a padding 0 u)
    (RealCumExecution.offset q localLaunch 0)
    (fun j => chunkPrefix q localLaunch a padding 0 j.1 j.2.1)
    (RealCumExecution.tileActive q localLaunch 0) i ha (fun j hj he => hinj j i hj ha he)
  exact hr.trans (RealCumExecution.prefix_active q localLaunch a padding 0 i ha)

theorem frame (p : Layout) (l : Launch) (c : Nat) (s o : RegionName)
    (a padding : Nat → ℝ) (u : BlockState) :
    BlockState.WriteWithin (RealCumExecution.footprint (tileLayout p l c) localLaunch 0 o) u
      (RealCumExecution.result (tileLayout p l c) localLaunch s o a padding 0 u) :=
  by
    intro region off hn
    exact RealCumExecution.scatter_frame (tileLayout p l c) localLaunch o a padding 0
      (RealCumExecution.prepared (tileLayout p l c) localLaunch s o a padding 0 u) region off hn

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.RebasedCumAccess
