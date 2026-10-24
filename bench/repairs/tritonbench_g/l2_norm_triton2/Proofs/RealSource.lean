import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.Real
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.PublicFacts

namespace VeriTile.Bench.Repairs.L2NormTriton2.RealExecution
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormTriton1.Host (World blockSize block_covers)
noncomputable section
open Classical

def adapter : Source.Adapter ℝ ℝ := ⟨algebra,id,fun _ => id,fun _ _ => rfl⟩
def negation : Negation ℝ := ⟨fun x => -x,neg_neg⟩

def reference {M β : Type} (ops : Ops M ℝ β) (call : Host.Call ℝ)
    (entry : World M) (row col : Nat) : ℝ :=
  formula call.cols call.eps
    (fun j => logical ops negation entry.memory call.input (row*call.cols+j))
    (fun j => logical ops negation entry.memory call.gradient (row*call.cols+j)) col

theorem expected_real {M β : Type} (ops : Ops M ℝ β) (call : Host.Call ℝ)
    (entry : World M) (row col : Nat) (accepted : call.cols≤16384) (active : col<call.cols) :
    Host.expected ops negation adapter call entry row col=reference ops call entry row col := by
  dsimp only [Host.expected,adapter,id]
  have same := Tile.scalar_congr algebra call.cols (blockSize call.cols) call.eps
    (Host.entryRow ops negation call call.cols row entry Bool.false)
    (Host.entryRow ops negation call call.cols row entry Bool.true)
    (fun j => logical ops negation entry.memory call.input (row*call.cols+j))
    (fun j => logical ops negation entry.memory call.gradient (row*call.cols+j))
    (fun j hj => by simp [Host.entryRow,Host.Call.inputFor,hj])
    (fun j hj => by simp [Host.entryRow,Host.Call.inputFor,hj]) col active
  rw [same,scalar_real _ _ _ _ _ _ (block_covers _ accepted)]
  rfl

#specNonCircular reference avoiding [Tile.scalar,Host.expected,Host.execute]
#axiomsClean expected_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.RealExecution
