import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardFirstValues
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardStep

/-! One complete first backward iteration, retaining the d_q scatter and all
four original pointer updates. No externally prepared state is required. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstStep
open VeriTile.Triton Primitives MaskedMath BackwardFirstState
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem advance_gradient (p : Layout) (l : Launch) (r : RegionName) (t : Nat)
    (ptr : RegName) (u : BlockState)
    (hp : u.regs .ptr [p.BK] ptr = some (keyGradientPointers p l r t)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [p.BK] ptr) (.constNat p.DK)) u =
      some (keyGradientPointers p l r (t+1)) := by
  simpa only [keyGradientPointers,keyGradientAddress,Nat.add_mul,one_mul,Nat.add_assoc] using
    (advance_pointer p.BK r ptr (fun k => keyGradientAddress p l t k.val) p.DK u hp)

theorem output_store (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r hasSeed t u) :
    stepStmt (.store .real [p.BK] (.ptr (.ref .ptr [p.BK] "p_dq"))
      (.ref .real [p.BK] "d_q") (.mask (.ref .bool [p.BK] "mask_bk")))
      (beforeStore p l initial r hasSeed t u) =
        some (scatter p l initial r hasSeed t (beforeStore p l initial r hasSeed t u)) := by
  exact Memory.store_vector p.BK r.dq "p_dq" "d_q" "mask_bk"
    (fun k => keyGradientAddress p l t k.val) (MaskedMath.dq p l initial r hasSeed t)
    (fun k => activeKey p l k.val) _
    (by simpa [beforeStore,keyGradientPointers] using hu.dq)
    (by simp [beforeStore]) (by simp [beforeStore,hu.maskK])

theorem executes (p : Layout) (l : Launch) (initial : BlockState)
    (r : BackwardRegions) (hasSeed : Bool) (t : Nat) (u : BlockState)
    (hu : Ready p l initial r hasSeed t u)
    (hkmem : ∀ off, u.readMem r.k off = initial.readMem r.k off)
    (hvmem : ∀ off, u.readMem r.v off = initial.readMem r.v off)
    (hdmem : ∀ off, u.readMem r.do_ off = initial.readMem r.do_ off) :
    stepStmts (BackwardSource.firstBody p) u = some (result p l initial r hasSeed t u) := by
  rw [←List.take_append_drop 6 (BackwardSource.firstBody p),
    stepStmts.append_some (BackwardFirstValues.before_store p l initial r hasSeed t u hu hkmem hvmem hdmem)]
  dsimp only [BackwardSource.firstBody,List.drop]
  rw [stepStmts.cons_some (output_store p l initial r hasSeed t u hu)]
  dsimp only [ForwardSource.advance]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ForwardStep.advance_key p l r.k t "p_k" _
    (by simp [scatter,beforeStore,hu.k])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ForwardStep.advance_value p l r.do_ t "p_do" _
    (by simp [scatter,beforeStore,hu.dout])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (ForwardStep.advance_value p l r.v t "p_v" _
    (by simp [scatter,beforeStore,hu.v])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (advance_gradient p l r.dq t "p_dq" _
    (by simp [scatter,beforeStore,hu.dq])))]
  rw [stepStmts.nil]
  rfl

#axiomsClean advance_gradient
#axiomsClean output_store
#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardFirstStep
