import bench.tritonbench_g.isfinite_kernel.Proofs.WitnessProgram

namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
open VeriTile.Triton FP.Structural
noncomputable section

def GridReady {α : Type} (s : State α) : Prop :=
  s.regs .nat [] "pid" = some (fun _ => 0) ∧
  s.regs .nat [] "num_ctas" = some (fun _ => 1)

def gridEntered {α : Type} (s : State α) : State α :=
  ((s.setReg "pid" .nat [] (fun _ => s.pids 0)).setReg
    "num_tiles0" .nat [] (fun _ => 2)).setReg
    "num_ctas" .nat [] (fun _ => s.numPids 0)

theorem grid_header_eval {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    FP.Structural.run M gridHeader s = some (gridEntered s) := by
  have hcount :
      FP.Structural.bop (fun a b : Nat => a / b) .nil
        (FP.Structural.bop (fun a b : Nat => a - b) .nil
          (FP.Structural.bop (fun a b : Nat => a + b) .nil
            (fun _ => 14) (fun _ => 8)) (fun _ => 1)) (fun _ => 8) =
        (fun _ : TileIndex [] => (2 : Nat)) := by
    funext i
    cases i
    change (14 + 8 - 1) / 8 = (2 : Nat)
    decide
  simp [gridHeader, gridBefore, gridKernel32, TypedSource.source32,
    ComputeKernel.surfaceBody, gridNumCtas, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalOp_unfold, FP.Structural.numeric,
    hcount, Option.bind, gridEntered] <;> rfl

theorem grid_entered_ready {α : Type} (s : State α)
    (hp : s.pids 0 = 0) (hn : s.numPids 0 = 1) : GridReady (gridEntered s) := by
  simp [GridReady, gridEntered, hp, hn]

#axiomsClean grid_header_eval
#axiomsClean grid_entered_ready
end
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits
