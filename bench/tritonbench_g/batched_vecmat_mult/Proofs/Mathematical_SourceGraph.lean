import bench.tritonbench_g.batched_vecmat_mult.Proofs.Mathematical_ValueModel

/-! Explicit arithmetic graph for the unchanged typed source. Capturing a graph
is separate from evaluating it: source additions and products are retained,
and the selected contraction pass creates actual ternary FMA nodes before
word evaluation. No equation between fused and separate rounding is used. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
open VeriTile.Triton FP.Structural

inductive Expr where
  | word (value : ValueProofs.Word)
  | input (region : RegionName) (offset : Nat) (value : ValueProofs.Word)
  | zero
  | mul (left right : Expr)
  | add (left right : Expr)
  | fma (left right carry : Expr)

instance : Inhabited Expr := ⟨.zero⟩

/-- Local contraction of one already captured source addition. Prefer its left
product, matching the first butterfly addition in the pinned SM90 profiles;
a right product is contracted with the left carry in the BK=1 profile. -/
def contractAdd : Expr → Expr → Expr
  | .mul x y, z => .fma x y z
  | z, .mul x y => .fma x y z
  | x, y => .add x y

/-- This pass acts on expression constructors, before any FP product is
rounded. It is deliberately not an identity theorem over FP words. -/
def contract : Expr → Expr
  | .word v => .word v
  | .input r o v => .input r o v
  | .zero => .zero
  | .mul x y => .mul (contract x) (contract y)
  | .add x y => contractAdd (contract x) (contract y)
  | .fma x y z => .fma (contract x) (contract y) (contract z)

def evaluate (cfg : FP.Config) : Expr → ValueProofs.Word
  | .word v => v
  | .input _ _ v => v
  | .zero => FP.Value.zero .fp32
  | .mul x y => FP.mul cfg (evaluate cfg x) (evaluate cfg y)
  | .add x y => FP.add cfg (evaluate cfg x) (evaluate cfg y)
  | .fma x y z => FP.fma cfg (evaluate cfg x) (evaluate cfg y) (evaluate cfg z)

/-- Exact-real interpretation with arbitrary input decoding; useful for the
mathematical target, without identifying word decoding with exact arithmetic. -/
def realValue (decode : ValueProofs.Word → ℝ) : Expr → ℝ
  | .word v => decode v
  | .input _ _ v => decode v
  | .zero => 0
  | .mul x y => realValue decode x * realValue decode y
  | .add x y => realValue decode x + realValue decode y
  | .fma x y z => realValue decode x * realValue decode y + realValue decode z

theorem contractAdd_real (decode : ValueProofs.Word → ℝ) (x y : Expr) :
    realValue decode (contractAdd x y) = realValue decode x + realValue decode y := by
  cases x <;> cases y <;> simp [contractAdd, realValue, add_comm]

theorem contraction_preserves_real (decode : ValueProofs.Word → ℝ) (x : Expr) :
    realValue decode (contract x) = realValue decode x := by
  induction x with
  | word v => rfl
  | input r o v => rfl
  | zero => rfl
  | mul x y hx hy => simp [contract, realValue, hx, hy]
  | add x y hx hy => simp [contract, contractAdd_real, realValue, hx, hy]
  | fma x y z hx hy hz => simp [contract, realValue, hx, hy, hz]

/-- The actual raw operands, not their rounded product, reach FMA. -/
theorem carry_fma (cfg : FP.Config) (a b : ValueProofs.Word) (carry : Expr) :
    evaluate cfg (contract (.add carry (.mul (.word a) (.word b)))) =
      evaluate cfg (contractAdd (contract carry) (.mul (.word a) (.word b))) := rfl

theorem first_butterfly_fma (cfg : FP.Config) (a b c d : ValueProofs.Word) :
    evaluate cfg (contract (.add (.mul (.word a) (.word b)) (.mul (.word c) (.word d)))) =
      FP.fma cfg a b (FP.mul cfg c d) := rfl

#axiomsClean contraction_preserves_real
#axiomsClean first_butterfly_fma
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.SourceGraph
