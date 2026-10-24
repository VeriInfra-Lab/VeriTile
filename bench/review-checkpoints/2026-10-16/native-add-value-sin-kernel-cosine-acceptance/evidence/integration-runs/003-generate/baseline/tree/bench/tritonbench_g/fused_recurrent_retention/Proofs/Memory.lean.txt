import bench.tritonbench_g.fused_recurrent_retention.Proofs.Primitives
import VeriTile.Triton.KernelLemmas.ScatterStore

/-! Actual masked vector-store execution and complete tagged-cell frame.
No readMem projection is used for the frame or for the written-cell tag. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Memory
open VeriTile.Triton Primitives
noncomputable section
set_option maxHeartbeats 600000

def scatter (n : Nat) (r : RegionName) (addr : Fin n → Nat) (value : Fin n → ℝ)
    (active : Fin n → Prop) [DecidablePred active] (s : BlockState) : BlockState :=
  (TileShape.allIndices [n]).foldl (fun u j =>
    if active j.1 then u.writeMem r (addr j.1) (value j.1) else u) s

theorem store_vector (n : Nat) (r : RegionName) (ptr valueName mask : RegName)
    (addr : Fin n → Nat) (value : Fin n → ℝ)
    (active : Fin n → Prop) [DecidablePred active] (s : BlockState)
    (hp : s.regs .ptr [n] ptr = some (pointers n r addr))
    (hv : s.regs .real [n] valueName = some (vector n value))
    (hm : s.regs .bool [n] mask = some (maskTile n active)) :
    stepStmt (.store .real [n] (.ptr (.ref .ptr [n] ptr))
      (.ref .real [n] valueName) (.mask (.ref .bool [n] mask))) s =
        some (scatter n r addr value active s) := by
  simp only [stepStmt, evalOp_ref, hp, hv, hm, Option.bind, bind, Option.pure_def]
  apply congrArg some
  change (TileShape.allIndices [n]).foldl _ s = (TileShape.allIndices [n]).foldl _ s
  apply congrArg (fun f : BlockState → TileIndex [n] → BlockState =>
    (TileShape.allIndices [n]).foldl f s)
  funext u j
  simp only [pointers, vector, maskTile, Tile.vec, decide_eq_true_eq]
  split_ifs <;> rfl

theorem written_cell (n : Nat) (r : RegionName) (addr : Fin n → Nat) (value : Fin n → ℝ)
    (active : Fin n → Prop) [DecidablePred active] (s : BlockState)
    (i : Fin n) (hi : active i)
    (huniq : ∀ j, active j → addr j = addr i → j = i) :
    (scatter n r addr value active s).mem r (addr i) = MemCell.real (value i) := by
  obtain ⟨before, after, he⟩ := List.append_of_mem (TileShape.mem_allIndices [n] (i,PUnit.unit))
  have hn := TileShape.allIndices_nodup [n]
  rw [he, List.nodup_append, List.nodup_cons] at hn
  have hafter : (i,PUnit.unit) ∉ after := hn.2.1.1
  unfold scatter
  rw [he, List.foldl_append, List.foldl_cons, if_pos hi]
  rw [foldl_store_preserve_cell]
  · simp [BlockState.writeMem]
  · intro j hj ha heq
    have hjEq : j = (i,PUnit.unit) := by
      have heq' := huniq j.1 ha heq.2
      rcases j with ⟨k,⟨⟩⟩
      simp_all
    exact hafter (hjEq ▸ hj)

theorem full_frame (n : Nat) (r : RegionName) (addr : Fin n → Nat) (value : Fin n → ℝ)
    (active : Fin n → Prop) [DecidablePred active] (s : BlockState)
    (region : RegionName) (offset : Nat)
    (h : ∀ j, active j → (r,addr j) ≠ (region,offset)) :
    (scatter n r addr value active s).mem region offset = s.mem region offset := by
  unfold scatter
  apply foldl_store_preserve_cell
  intro j hj ha heq
  exact h j.1 ha (Prod.ext heq.1 heq.2)

#axiomsClean store_vector
#axiomsClean written_cell
#axiomsClean full_frame
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.Memory
