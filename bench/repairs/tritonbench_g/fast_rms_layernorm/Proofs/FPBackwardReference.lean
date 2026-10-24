import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPReference
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPBackwardReference
open VeriTile.Triton FP.Structural FPReference
noncomputable section
variable {α : Type}

def normed (M : Algebra α) (h : Call) (x : Fin h.block → α) (saved : α) (j : Fin h.block) :=
  mul M (input M h x j) (cast M saved)

def weighted (M : Algebra α) (h : Call) (g w : Fin h.block → α) (j : Fin h.block) :=
  mul M (input M h g j) (if h.gemma then add M (input M h w j) (literal M 1.0) else input M h w j)

def rowSum (M : Algebra α) (h : Call) (x w g : Fin h.block → α) (saved : α) :=
  reduce M h.block (fun j => mul M (weighted M h g w j) (normed M h x saved j))

def output (M : Algebra α) (h : Call) (x w g : Fin h.block → α) (saved : α) (j : Fin h.block) : Cell α :=
  .mk .real (cast M (mul M (div M (cast M saved) (literal M (h.cols:ℝ)))
    (sub M (mul M (literal M (h.cols:ℝ)) (weighted M h g w j))
      (mul M (normed M h x saved j) (rowSum M h x w g saved)))))
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPBackwardReference
