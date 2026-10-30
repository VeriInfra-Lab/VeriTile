import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardMemory
import bench.repairs.tritonbench_g.chunk_linear_attn.Readback
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelForward
open VeriTile Triton FP.Structural
open KernelBackward (PID)
open KernelReal (parameters)
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.Repairs.ChunkDeltaFwd (foldRange_observe foldRange_last scatter_readback)
open VeriTile.Bench.Repairs.ChunkLinearAttn (hActive hOffset fOffset scatter_frame)
namespace G
export VeriTile.Bench.Repairs.ChunkLinearAttn.Geometry (history_collision final_collision)
end G
noncomputable section
variable {α : Type} [Inhabited α]

def Footprint (c : Shape) (pid : PID) (final : Bool) (r : RegionName) (o : Nat) : Prop :=
  (r="h" ∧ ∃ n,n<c.chunks ∧ ∃ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i ∧ o=hOffset (parameters c) pid n i) ∨
  (r="ht" ∧ final=Bool.true ∧ ∃ i : TileIndex [c.BK,c.BV],hActive (parameters c) pid i ∧ o=fOffset (parameters c) pid i)

theorem history_write_read (M : Algebra α) (c : Shape) (pid : PID)
    (n : Nat) (h : KernelBackward.Matrix α c.BK c.BV) (m : Memory α) (i : TileIndex [c.BK,c.BV])
    (hn : n<c.chunks) (hi : hActive (parameters c) pid i) :
    writeHistory M c pid n h m "h" (hOffset (parameters c) pid n i)=.mk .real (M.cast (some .fp32) .real .real (h i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (G.history_collision (parameters c) pid pid _ _ a b ha hb hn hn he).2.2)

theorem state_read_history (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input m : Memory α) (seed final : Bool) (n : Nat) (i : TileIndex [c.BK,c.BV])
    (hn : n<c.chunks) (hi : hActive (parameters c) pid i) :
    memory M D c pid input m seed final "h" (hOffset (parameters c) pid n i)=
      .mk .real (M.cast (some .fp32) .real .real (values M D c pid input seed n i)) := by
  have hf : ∀ mem, writeFinal M c pid final (values M D c pid input seed c.chunks)
      mem "h" (hOffset (parameters c) pid n i)=mem "h" (hOffset (parameters c) pid n i) := by
    intro mem
    cases final
    · rfl
    · exact scatter_frame _ _ _ _ _ _ _ _ (Or.inl (by decide))
  unfold memory
  rw [hf]
  apply foldRange_last _ (fun mem : Memory α => mem "h" (hOffset (parameters c) pid n i)) c.chunks n m _ hn
  · intro mem
    exact history_write_read M c pid n _ mem i hn hi
  · intro k hnk hk mem
    apply scatter_frame
    right
    intro j _ hj he
    have hc := (G.history_collision (parameters c) pid pid n k i j hi hj hn hk he).2.1
    omega

theorem state_read_final (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input m : Memory α) (seed : Bool) (i : TileIndex [c.BK,c.BV]) (hi : hActive (parameters c) pid i) :
    memory M D c pid input m seed Bool.true "ht" (fOffset (parameters c) pid i)=
      .mk .real (M.cast (some .fp32) .real .real (values M D c pid input seed c.chunks i)) :=
  scatter_readback _ _ _ _ _ i hi (fun a b ha hb he =>
    (G.final_collision (parameters c) pid pid a b ha hb he).2)

theorem memory_frame (M : Algebra α) (D : DotPrimitive α) (c : Shape) (pid : PID)
    (input m : Memory α) (seed final : Bool) (r : RegionName) (o : Nat)
    (safe : ¬Footprint c pid final r o) : memory M D c pid input m seed final r o=m r o := by
  have hf : ∀ carry mem,writeFinal M c pid final carry mem r o=mem r o := by
    intro carry mem
    cases final
    · rfl
    · apply scatter_frame
      by_cases hr : r="ht"
      · right
        intro i _ hi he
        exact safe (Or.inr ⟨hr,rfl,i,hi,he⟩)
      · exact Or.inl hr
  unfold memory
  rw [hf]
  apply foldRange_observe _ (fun mem : Memory α=>mem r o)
  intro n hn mem
  apply scatter_frame
  by_cases hr : r="h"
  · right
    intro i _ hi he
    exact safe (Or.inl ⟨hr,n,hn,i,hi,he⟩)
  · exact Or.inl hr

#axiomsClean state_read_history
#axiomsClean state_read_final
#axiomsClean memory_frame
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelForward
