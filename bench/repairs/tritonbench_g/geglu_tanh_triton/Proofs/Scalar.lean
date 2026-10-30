import bench.repairs.tritonbench_g.geglu_tanh_triton.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.GegluTanhTriton.Tile
open VeriTile.Triton FP.Structural
noncomputable section
/-- Exact original scalar order, now evaluated for each active flat element.
The ordinary alpha assignment materializes an FP32 scalar tensor.
The constexpr 3*beta is folded before conversion. -/
def scalar (M : Algebra α) (ch : Channel) (a b dc : α) : α :=
  let a := M.cast (some .fp32) .real .real a
  let b := M.fp32Load b
  let dc := M.fp32Load dc
  let lit := M.literal (some .fp32) .real
  let alpha := M.literal (some .fp32) .real 0.7978845608028654
  let mul := M.binary (some .fp32) .real .mul
  let add := M.binary (some .fp32) .real .add
  let sub := M.binary (some .fp32) .real .sub
  let cubed := mul (mul a a) a
  let z := mul alpha (add a (mul (lit 0.044715) cubed))
  let t := M.unary (some .fp32) .libdeviceTanh z
  let g := mul (mul (lit 0.5) a) (add (lit 1) t)
  let term1 := mul (lit 0.5) (add (lit 1) t)
  let term2 := mul (mul (mul (lit 0.5) a) (sub (lit 1) (mul t t)))
    (mul alpha
      (add (lit 1) (mul (mul (lit 0.134145) a) a)))
  match ch with
  | .c => mul g b
  | .da => mul (mul dc b) (add term1 term2)
  | .db => mul dc g

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.GegluTanhTriton.Tile
