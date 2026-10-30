import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardProgram
import bench.repairs.tritonbench_g.chunk_linear_attn.Readback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe foldRange_last scatter_readback)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset scatter_frame)
namespace G
export VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry (history_collision final_collision chunk_lt chunk_injective)
end G
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

theorem history_write_read (M : Algebra α) (p : Parameters) (pid : PID) 
    (n : Nat) (h : Carry p α) (m : Memory α) (i : TileIndex [p.BK,p.BV])
    (hn : n<p.NT) (hi : hActive p pid i) :
    writeHistory M p pid n h m "dhistory" (hOffset p pid (time p n) i)=.mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (G.history_collision p pid pid _ _ a b ha hb (G.chunk_lt p Bool.true n hn)
      (G.chunk_lt p Bool.true n hn) he).2.2)

theorem final_write_read (M : Algebra α) (p : Parameters) (pid : PID) 
    (h : Carry p α) (m : Memory α) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    writeInitial M p pid Bool.true h m "initialGradient" (fOffset p pid i)=.mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he => (G.final_collision p pid pid a b ha hb he).2)

theorem state_read_history (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (terminal initial : Bool) (n : Nat) (i : TileIndex [p.BK,p.BV])
    (hn : n<p.NT) (hi : hActive p pid i) :
    memory M D p pid input m scale terminal initial "dhistory" (hOffset p pid (time p n) i)=
      .mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (values M D p pid input scale terminal n i)) := by
  have hf : ∀ mem, writeInitial M p pid initial (values M D p pid input scale terminal p.NT)
      mem "dhistory" (hOffset p pid (time p n) i)=mem "dhistory" (hOffset p pid (time p n) i) := by
    intro mem
    cases initial
    · rfl
    · apply scatter_frame
      left
      decide
  unfold memory
  rw [hf]
  apply foldRange_last _ (fun mem : Memory α => mem "dhistory" (hOffset p pid (time p n) i)) p.NT n m _ hn
  · intro mem
    exact history_write_read M p pid n _ mem i hn hi
  · intro k hnk hk mem
    apply scatter_frame
    right
    intro j _ hj he
    have hc := (G.history_collision p pid pid _ _ i j hi hj
      (G.chunk_lt p Bool.true n hn) (G.chunk_lt p Bool.true k hk) he).2.1
    have hnkeq := G.chunk_injective p Bool.true n k hn hk hc
    omega

theorem state_read_final (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (pid : PID)
    (input m : Memory α) (scale : ℝ) (terminal : Bool) (i : TileIndex [p.BK,p.BV]) (hi : hActive p pid i) :
    memory M D p pid input m scale terminal Bool.true "initialGradient" (fOffset p pid i)=
      .mk .real (VeriTile.Bench.Repairs.ChunkLinearAttn.cast M (values M D p pid input scale terminal p.NT i)) :=
  final_write_read M p pid _ _ i hi

#axiomsClean state_read_history
#axiomsClean state_read_final
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelBackward
