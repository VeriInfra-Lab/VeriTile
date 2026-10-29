import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Real
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.PublicFacts

namespace VeriTile.Bench.Repairs.L2NormTriton1.RealExecution
open VeriTile.Bench.Repairs.SwigluTriton.Memory
noncomputable section
open Classical

def adapter : Source.Adapter ℝ ℝ := ⟨algebra,id,fun _ => id,fun _ _ => rfl⟩
def negation : Negation ℝ := ⟨fun x => -x,neg_neg⟩

/-- Independent formula at original logical input coordinates. -/
def reference {M β : Type} (ops : Ops M ℝ β) (call : Host.Call ℝ)
    (entry : Host.World M) (row col : Nat) : ℝ :=
  formula (call.input.shape.getLastD 0) call.eps
    (fun j => logical ops negation entry.memory call.input (row*(call.input.shape.getLastD 0)+j)) col

theorem expected_real {M β : Type} (ops : Ops M ℝ β) (call : Host.Call ℝ)
    (entry : Host.World M) (row col : Nat) (accepted : call.input.shape.getLastD 0≤16384)
    (active : col<call.input.shape.getLastD 0) :
    Host.expected ops negation adapter call entry row col=reference ops call entry row col := by
  dsimp only [Host.expected,adapter,id]
  have same := Tile.scalar_congr algebra (call.input.shape.getLastD 0)
    (Host.blockSize (call.input.shape.getLastD 0)) call.eps
    (Host.entryRow ops negation call (call.input.shape.getLastD 0) row entry)
    (fun j => logical ops negation entry.memory call.input (row*(call.input.shape.getLastD 0)+j))
    (fun j hj => by simp only [Host.entryRow,if_pos hj]) col active
  rw [same,scalar_real _ _ _ _ _ (Host.block_covers _ accepted)]
  rfl

#specNonCircular reference avoiding [Tile.scalar,Host.expected,Host.execute]
#axiomsClean expected_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.RealExecution
