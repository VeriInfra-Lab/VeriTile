import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Tile
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Math
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivTriton.RealModel
open VeriTile.Triton FP.Structural
open scoped BigOperators
noncomputable section
open Classical
set_option autoImplicit false

/-- Independent mathematical interpretation. It retains the source's scalar
operations and selections; it makes no claim that IEEE words equal Reals. -/
def algebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b
    | .max => max a b | .pow => a ^ b
  unary := fun _ op a => match op with
    | .log => Real.log a | .exp => Real.exp a | .neg _ => -a | _ => 0
  compareEq := fun _ _ => some (fun a b => decide (a = b))
  cast := fun _ _ _ a => a
  fromNat := fun _ a => a
  fromInt := fun _ a => a
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ => L2NormBwd.Math.sumValues

def model : Tile.Model ℝ := ⟨algebra, (fun a b => decide (a = b)), rfl⟩

theorem loss_real (flag : Bool) (y t : ℝ) :
    Tile.loss model flag y t = Math.loss flag y t := by
  cases flag <;> norm_num [Tile.loss, Tile.rawLoss, Tile.zero, model, algebra, Math.loss]

theorem gradient_real (flag : Bool) (t : ℝ) :
    Tile.gradient model flag t = Math.gradient flag t := by
  cases flag <;> simp [Tile.gradient, model, algebra, Math.gradient]

theorem chunk_real (flag : Bool) (N B k : Nat) (ys ts : Nat → ℝ) :
    Tile.chunkSum model flag N B k ys ts =
      VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss flag ys ts N B k := by
  change (∑ j : Fin B, Tile.laneLoss model flag N (k * B + j.val) ys ts) =
    ∑ j : Fin B, VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.paddedLoss flag ys ts N (k * B + j.val)
  apply Finset.sum_congr rfl
  intro j _
  rw [VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.paddedLoss_eq]
  by_cases h : k * B + j.val < N
  · simp only [Tile.laneLoss, Tile.loaded, h, if_true]
    change Tile.loss model flag (ys (k * B + j.val)) (ts (k * B + j.val)) = _
    rw [loss_real, Math.loss_original]
  · norm_num [Tile.laneLoss, h, Tile.zero, model, algebra]

theorem all_chunks_real (rows cols block : Nat) (hBlock : 0 < block) (flag : Bool)
    (ys ts : Nat → Nat → ℝ) :
    (∑ r : Fin rows,
      ∑ k : Fin (VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.rounds cols block),
        Tile.chunkSum model flag cols block k.val (ys r.val) (ts r.val)) =
      Math.total rows cols flag ys ts := by
  simp only [chunk_real]
  exact Math.row_chunks_total rows cols block hBlock flag ys ts

#axiomsClean loss_real
#axiomsClean gradient_real
#axiomsClean chunk_real
#axiomsClean all_chunks_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.RealModel
