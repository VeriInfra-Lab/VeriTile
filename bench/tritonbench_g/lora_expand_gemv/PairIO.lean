import bench.tritonbench_g.lora_expand_gemv.IO
import bench.tritonbench_g.lora_expand_gemv.CandidateKernels
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Pair
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section

def invocation (rest : Algebra Word) (plans : ∀ n,ReductionPlan n) (g : Geometry)
    (q : Nat) : Invocation Word :=
  { launch rest plans g q with kernel := unitSplitSource g.parameters }

def variant (rest : Algebra Word) (plans : ∀ n,ReductionPlan n) (g : Geometry)
    (order : List Nat) : ProgramIO.Variant (io rest plans g order) where
  schedule := fun _ => order.map (invocation rest plans g)
  observe := (io rest plans g order).observe

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Pair
