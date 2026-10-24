import bench.tritonbench_g.uniform_sampling.Proofs.Algorithm
import VeriTile.Triton.Semantics.Eval

/-! Local execution of the unchanged uniform conversion and affine statement.
The input consists of actual uint32 words. The decoder keeps its canonical
fp32 bit-level meaning; only the surrounding arithmetic uses the real
projection here. This is not a native fused-arithmetic refinement theorem.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.AffineExecution
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def affineValues {B : Nat} (words : TileIndex [B] → Random.Word)
    (lower upper : ℝ) : Tile .real [B] :=
  ⟨fun i => some (Random.uniformReal (words i) * (upper - lower) + lower)⟩

theorem affine_eval {B : Nat} (words : TileIndex [B] → Random.Word)
    (name : RegName) (lower upper : ℝ) (s : BlockState)
    (hw : s.regs .nat [B] name = some ⟨fun i => (words i).toNat⟩) :
    evalOp (.add .real .scalarR
      (.mul .real .scalarR (.uniform32 (.ref .nat [B] name))
        (.sub .real .nil (.const upper) (.const lower))) (.const lower)) s =
      some (affineValues words lower upper) := by
  have hu := RandomDomain.eval_uniform words s (.ref .nat [B] name)
    (by simpa only [evalOp.eq_def] using hw)
  rw [evalOp_add, evalOp_mul, hu]
  simp [-Tile.scalar_eta, evalOp_sub, evalOp.eq_def, Tile.bop, Tile.scalar,
    NumericDType.add, NumericDType.mul, NumericDType.sub, affineValues]

theorem affine_step {B : Nat} (words : TileIndex [B] → Random.Word)
    (name : RegName) (lower upper : ℝ) (s : BlockState)
    (hw : s.regs .nat [B] name = some ⟨fun i => (words i).toNat⟩) :
    stepStmt (Algorithm.affineStmt B name lower upper) s =
      some (s.setReg name .real [B] (affineValues words lower upper)) := by
  simp only [Algorithm.affineStmt, stepStmt]
  rw [affine_eval words name lower upper s hw]
  rfl

theorem scalar_formula {B : Nat} (seed offset : Int) (i4 : TileIndex [B] → Nat)
    (f : Random.Direct.Field) (lower upper : ℝ) :
    affineValues (fun i => ScalarSpec.word seed offset (i4 i) f) lower upper =
      (⟨fun i => some (ScalarSpec.value seed offset (i4 i) f lower upper)⟩ : Tile .real [B]) := rfl

#axiomsClean affine_eval
#axiomsClean affine_step
#axiomsClean scalar_formula

end
end VeriTile.Bench.TritonBenchG.UniformSampling.AffineExecution
