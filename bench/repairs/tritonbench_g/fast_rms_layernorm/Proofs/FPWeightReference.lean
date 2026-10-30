import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPReference

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightReference
open VeriTile.Triton FP.Structural FPReference
noncomputable section
variable {α : Type}

def normed (M : Algebra α) (gemma : Bool) (x saved : α) : α :=
  let z := mul M (cast M x) (cast M saved)
  if gemma then z else cast M (cast M z)

def term (M : Algebra α) (gemma : Bool) (x g saved : α) : α :=
  mul M (cast M g) (normed M gemma x saved)

/-- Increasing-row accumulation, including the exact initial FP32 zero. -/
def carry (M : Algebra α) (gemma : Bool) (x g saved : Nat → α) : Nat → α
  | 0 => literal M 0
  | n+1 => add M (carry M gemma x g saved n) (term M gemma (x n) (g n) (saved n))

def output (M : Algebra α) (gemma : Bool) (x g saved : Nat → α) (rows : Nat) : Cell α :=
  .mk .real (carry M gemma x g saved rows)
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightReference
