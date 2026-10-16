import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardValues
import bench.tritonbench_g.fused_recurrent_retention.Proofs.Memory

/-! Every statement of a forward time step, including its actual scatter and
four pointer advances. No output value or supplied carry buffer is assumed. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardStep
open VeriTile.Triton Primitives MaskedMath ForwardState
noncomputable section
set_option maxHeartbeats 900000
set_option linter.unusedSimpArgs false

theorem advance_key (p : Layout) (l : Launch) (r : RegionName) (t : Nat)
    (ptr : RegName) (u : BlockState)
    (hp : u.regs .ptr [p.BK] ptr = some (keyPointers p l r t)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [p.BK] ptr) (.constNat p.DK)) u =
      some (keyPointers p l r (t+1)) := by
  simpa only [keyPointers, keyAddress, Nat.add_mul, one_mul, Nat.add_assoc] using
    (advance_pointer p.BK r ptr (fun j => keyAddress p l t j.val) p.DK u hp)

theorem advance_value (p : Layout) (l : Launch) (r : RegionName) (t : Nat)
    (ptr : RegName) (u : BlockState)
    (hp : u.regs .ptr [p.BV] ptr = some (valuePointers p l r t)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [p.BV] ptr) (.constNat p.DV)) u =
      some (valuePointers p l r (t+1)) := by
  simpa only [valuePointers, valueAddress, Nat.add_mul, one_mul, Nat.add_assoc] using
    (advance_pointer p.BV r ptr (fun j => valueAddress p l t j.val) p.DV u hp)

theorem advance_output (p : Layout) (l : Launch) (r : RegionName) (t : Nat)
    (u : BlockState)
    (hp : u.regs .ptr [p.BV] "p_o" = some (outputPointers p l r t)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [p.BV] "p_o") (.constNat p.DV)) u =
      some (outputPointers p l r (t+1)) := by
  simpa only [outputPointers, outputAddress, Nat.add_mul, one_mul, Nat.add_assoc] using
    (advance_pointer p.BV r "p_o" (fun j => outputAddress p l t j.val) p.DV u hp)

theorem output_store (p : Layout) (l : Launch) (initial : BlockState)
    (r : ForwardRegions) (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r hasSeed t u) :
    stepStmt (.store .real [p.BV] (.ptr (.ref .ptr [p.BV] "p_o"))
      (.ref .real [p.BV] "_o") (.mask (.ref .bool [p.BV] "mask_bv")))
      (beforeStore p l initial r hasSeed t u) =
        some (ForwardState.scatter p l initial r hasSeed t
          (beforeStore p l initial r hasSeed t u)) := by
  exact Memory.store_vector p.BV r.o "p_o" "_o" "mask_bv"
    (fun j => outputAddress p l t j.val) (output p l initial r hasSeed t)
    (fun j => activeValue p l j.val) _
    (by simpa [beforeStore, outputPointers] using hu.o)
    (by simp [beforeStore]) (by simp [beforeStore,hu.maskV])

theorem executes (p : Layout) (l : Launch) (initial : BlockState)
    (r : ForwardRegions) (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r hasSeed t u)
    (hqmem : ∀ off, u.readMem r.q off = initial.readMem r.q off)
    (hkmem : ∀ off, u.readMem r.k off = initial.readMem r.k off)
    (hvmem : ∀ off, u.readMem r.v off = initial.readMem r.v off) :
    stepStmts (ForwardSource.loopBody p) u =
      some (result p l initial r hasSeed t u) := by
  rw [← List.take_append_drop 6 (ForwardSource.loopBody p),
    stepStmts.append_some (ForwardValues.before_store p l initial r hasSeed t u
      hu hqmem hkmem hvmem)]
  dsimp only [ForwardSource.loopBody, List.drop]
  rw [stepStmts.cons_some (output_store p l initial r hasSeed t u hu)]
  dsimp only [ForwardSource.advance]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_key p l r.q t "p_q" _
    (by simp [ForwardState.scatter,beforeStore,hu.q])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_key p l r.k t "p_k" _
    (by simp [ForwardState.scatter,beforeStore,hu.k])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_output p l r.o t _
    (by simp [ForwardState.scatter,beforeStore,hu.o])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_value p l r.v t "p_v" _
    (by simp [ForwardState.scatter,beforeStore,hu.v])))]
  rw [stepStmts.nil]
  rfl

#axiomsClean advance_key
#axiomsClean advance_value
#axiomsClean advance_output
#axiomsClean output_store
#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardStep
