import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardHeaderSource
import bench.tritonbench_g.fused_recurrent_retention.Proofs.HeaderPrimitives
import bench.tritonbench_g.fused_recurrent_retention.Proofs.FinalMemory

/-! The actual conditional final-state epilogue. It runs after every time loop,
including an empty loop, and requires only the reached register invariant. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardFinal
open VeriTile.Triton ForwardState StatePrimitives ForwardHeaderSource
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def result (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) (u : BlockState) : BlockState :=
  if storeFinal then FinalMemory.scatter p l initial r hasSeed
    (u.setReg "p_final_s" .ptr [p.BV,p.BK] (statePointers p l r.final))
  else u

theorem body_executes (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (u : BlockState) (hu : Ready p l initial r hasSeed p.T u) :
    stepStmts
      [.assign .ptr [p.BV,p.BK] "p_final_s" (statePointerOp p r.final),
       .store .real [p.BV,p.BK] (.ptr (.ref .ptr [p.BV,p.BK] "p_final_s"))
         (.ref .real [p.BV,p.BK] "h") (.mask (.ref .bool [p.BV,p.BK] "mask_kv"))] u =
      some (result p l initial r hasSeed Bool.true u) := by
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (HeaderPrimitives.state_pointer p l r.final u hu.headId hu.keyId hu.valueId))]
  rw [stepStmts.cons_some (FinalMemory.store p l initial r hasSeed _
    (by simp) (by simp [hu.carry]) (by simpa [stateMask] using hu.maskKV))]
  rw [stepStmts.nil]
  rfl

theorem executes (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal swapped : Bool) (u : BlockState)
    (hu : Ready p l initial r hasSeed p.T u) :
    stepStmts (ForwardSource.finalCode p r hasSeed storeFinal swapped) u =
      some (result p l initial r hasSeed storeFinal u) := by
  rw [ForwardHeaderSource.final_exact]
  have hif : stepStmt (.ifThen (.constBool storeFinal)
      [.assign .ptr [p.BV,p.BK] "p_final_s" (statePointerOp p r.final),
       .store .real [p.BV,p.BK] (.ptr (.ref .ptr [p.BV,p.BK] "p_final_s"))
         (.ref .real [p.BV,p.BK] "h") (.mask (.ref .bool [p.BV,p.BK] "mask_kv"))]) u =
      some (result p l initial r hasSeed storeFinal u) := by
    have hc (b : Bool) (v : BlockState) :
        evalOp (.constBool b) v = some (Tile.scalar b) := by
      simp only [evalOp.eq_def]
    cases storeFinal
    · simp only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,
        Bool.false_eq_true,↓reduceIte,result]
    · simpa only [stepStmt,hc,Tile.scalar_data,Option.bind,bind,↓reduceIte]
        using body_executes p l initial r hasSeed u hu
  rw [finalCode,stepStmts.cons_some hif,stepStmts.nil]

theorem output_unchanged (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) (u : BlockState)
    (hsep : storeFinal = Bool.true → r.o ≠ r.final) (off : Nat) :
    (result p l initial r hasSeed storeFinal u).mem r.o off = u.mem r.o off := by
  cases storeFinal
  · rfl
  · apply FinalMemory.frame
    intro k j _ _ heq
    exact hsep rfl (congrArg Prod.fst heq).symm

theorem final_cell (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed : Bool) (u : BlockState) (k : Fin p.BK) (j : Fin p.BV)
    (hk : activeKey p l k.val) (hj : activeValue p l j.val) :
    (result p l initial r hasSeed Bool.true u).mem r.final (stateAddress p l k.val j.val) =
      MemCell.real (MaskedMath.stateValue p l initial r.k r.v r.initial hasSeed p.T k j) := by
  exact FinalMemory.cell p l initial r hasSeed _ k j hk hj

theorem frame (p : Layout) (l : Launch) (initial : BlockState) (r : ForwardRegions)
    (hasSeed storeFinal : Bool) (u : BlockState) (region : RegionName) (off : Nat)
    (h : storeFinal = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
      activeKey p l k.val → activeValue p l j.val →
      (r.final,stateAddress p l k.val j.val) ≠ (region,off)) :
    (result p l initial r hasSeed storeFinal u).mem region off = u.mem region off := by
  cases storeFinal
  · rfl
  · exact FinalMemory.frame p l initial r hasSeed _ region off (h rfl)

#axiomsClean body_executes
#axiomsClean executes
#axiomsClean output_unchanged
#axiomsClean final_cell
#axiomsClean frame
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardFinal
