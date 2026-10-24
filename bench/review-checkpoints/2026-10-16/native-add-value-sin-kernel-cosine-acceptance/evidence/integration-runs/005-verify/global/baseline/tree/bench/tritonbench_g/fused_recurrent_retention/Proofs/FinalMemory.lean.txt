import bench.tritonbench_g.fused_recurrent_retention.Proofs.StatePrimitives
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Addressing
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Memory

/-! Actual optional final-state matrix store, with complete tagged-cell results
and frame. The formula is the initial-snapshot state at time T, including T=0. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.FinalMemory
open VeriTile.Triton Primitives MaskedMath ForwardState StatePrimitives
noncomputable section
set_option maxHeartbeats 900000
set_option linter.unusedSimpArgs false

def scatter (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BV,p.BK]).foldl (fun s i =>
    if activeKey p l i.2.1.val ∧ activeValue p l i.1.val then
      s.writeMem r.final (stateAddress p l i.2.1.val i.1.val)
        (stateValue p l initial r.k r.v r.initial hasSeed p.T i.2.1 i.1)
    else s) u

theorem store (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (u : BlockState)
    (hp : u.regs .ptr [p.BV,p.BK] "p_final_s" = some (statePointers p l r.final))
    (hh : u.regs .real [p.BV,p.BK] "h" = some (carryTile p l initial r hasSeed p.T))
    (hm : u.regs .bool [p.BV,p.BK] "mask_kv" = some (stateMask p l)) :
    stepStmt (.store .real [p.BV,p.BK] (.ptr (.ref .ptr [p.BV,p.BK] "p_final_s"))
      (.ref .real [p.BV,p.BK] "h") (.mask (.ref .bool [p.BV,p.BK] "mask_kv"))) u =
      some (scatter p l initial r hasSeed u) := by
  simp only [stepStmt,evalOp_ref,hp,hh,hm,Option.bind,bind,Option.pure_def]
  apply congrArg some
  change (TileShape.allIndices [p.BV,p.BK]).foldl _ u =
    (TileShape.allIndices [p.BV,p.BK]).foldl _ u
  apply congrArg (fun f : BlockState → TileIndex [p.BV,p.BK] → BlockState =>
    (TileShape.allIndices [p.BV,p.BK]).foldl f u)
  funext s i
  simp only [statePointers,stateMask,carryTile,matrix,Tile.mat,decide_eq_true_eq]
  split_ifs <;> rfl

theorem cell (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (u : BlockState) (k : Fin p.BK) (j : Fin p.BV)
    (hk : activeKey p l k.val) (hj : activeValue p l j.val) :
    (scatter p l initial r hasSeed u).mem r.final (stateAddress p l k.val j.val) =
      MemCell.real (stateValue p l initial r.k r.v r.initial hasSeed p.T k j) := by
  obtain ⟨before,after,he⟩ := List.append_of_mem
    (TileShape.mem_allIndices [p.BV,p.BK] (j,k,PUnit.unit))
  have hn := TileShape.allIndices_nodup [p.BV,p.BK]
  rw [he,List.nodup_append,List.nodup_cons] at hn
  have hafter : (j,k,PUnit.unit) ∉ after := hn.2.1.1
  unfold scatter
  rw [he,List.foldl_append,List.foldl_cons,if_pos ⟨hk,hj⟩]
  rw [foldl_store_preserve_cell]
  · simp [BlockState.writeMem]
  · intro i hi ha heq
    obtain ⟨hkey,hval⟩ := Addressing.state_unique p l i.2.1 k i.1 j ha.2 hj heq.2
    have hieq : i = (j,k,PUnit.unit) := by
      rcases i with ⟨v,e,⟨⟩⟩
      simp_all
    exact hafter (hieq ▸ hi)

theorem frame (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (u : BlockState) (region : RegionName) (off : Nat)
    (h : ∀ (k : Fin p.BK) (j : Fin p.BV), activeKey p l k.val → activeValue p l j.val →
      (r.final,stateAddress p l k.val j.val) ≠ (region,off)) :
    (scatter p l initial r hasSeed u).mem region off = u.mem region off := by
  unfold scatter
  apply foldl_store_preserve_cell
  intro i _ ha heq
  exact h i.2.1 i.1 ha.1 ha.2 (Prod.ext heq.1 heq.2)

#axiomsClean store
#axiomsClean cell
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.FinalMemory
