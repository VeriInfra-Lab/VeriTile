import bench.tritonbench_g.llama_ff_triton.Proofs.Grid
import bench.tritonbench_g.llama_ff_triton.Proofs.Real

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.RealGrid
open VeriTile.Triton FP.Structural Execution
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section

def unusedGate : GateInstructions ℝ := ⟨Neg.neg,fun x y z => x*y+z,fun x => 1/(1+Real.exp x)⟩
def run (p : Parameters) (initial : State ℝ) := Grid.grid Bool.false realArithmetic unusedGate p initial

theorem complete (p : Parameters) (initial : State ℝ) (bytes : ByteValues initial) :
    ∃ final,run p initial=some final ∧
      GridExecution.Post "OUT" (Grid.address p)
        (fun x : Grid.Index p => Cell.mk .fp16 (Real.target p initial x.1.val x.2.val))
        (fun _ => True) initial final := by
  obtain ⟨final,he,values,frame⟩ := Grid.whole Bool.false realArithmetic unusedGate p initial bytes
  refine ⟨final,he,?_,frame⟩
  intro x _
  simpa only [Grid.target,Bool.false_eq_true,ite_false,Real.source_formula] using values x trivial

#axiomsClean complete
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.RealGrid
