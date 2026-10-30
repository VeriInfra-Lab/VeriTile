import bench.repairs.tritonbench_g.matmul_triton1.IO
import bench.repairs.tritonbench_g.matmul_triton1.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.MatmulTriton1.LogicalIOProof
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem sum_fin (f : Nat → ℝ) (n : Nat) : RealSum.sum f n = ∑ i : Fin n, f i.val := by
  induction n with
  | zero => simp [RealSum.sum]
  | succ n ih => simpa [RealSum.sum,Fin.sum_univ_castSucc] using congrArg (fun x => x+f n) ih

def target (p : Core.Parameters)
    (ab : (Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ))
    (row : Fin p.rows) (col : Fin p.columns) : ℝ :=
  ∑ k : Fin p.innerSize, ab.1 row k * ab.2 k col

theorem headline_iff (rest : Algebra ℝ) (p : Core.Parameters) :
    (MatmulIO.logicalIO rest Core.realPrimitives p ⊨ target p) ↔
    (MatmulIO.io rest Core.realPrimitives p).Implements (MatmulIO.realTarget p) := by
  have formula : (MatmulIO.logicalIO rest Core.realPrimitives p).lift (target p) =
      MatmulIO.realTarget p := by
    funext xs port i
    fin_cases port
    change (∑ k : Fin p.innerSize, ((MatmulIO.matrixInputs p) xs).1 i.1 k *
      ((MatmulIO.matrixInputs p) xs).2 k i.2.1) = _
    change (∑ k : Fin p.innerSize, ((MatmulIO.matrixInputs p) xs).1 i.1 k *
      ((MatmulIO.matrixInputs p) xs).2 k i.2.1) =
      RealSum.sum (fun k => MatmulIO.left p xs 0 i.1 k * MatmulIO.right p xs 0 i.2.1 k) p.innerSize
    rw [sum_fin]
    apply Finset.sum_congr rfl
    intro k _
    simp [MatmulIO.matrixInputs,MatmulIO.left,MatmulIO.right,k.isLt]
  change (MatmulIO.io _ _ _).Implements _ ↔ _
  rw [formula] <;> rfl

theorem implements (rest : Algebra ℝ) (p : Core.Parameters)
    (canonical : FiniteSource.Canonical p) (ready : FiniteIndex.Ready p)
    (storage : FiniteIndex.StorageBounds p) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (MatmulIO.logicalIO rest Core.realPrimitives p ⊨ target p) :=
  (headline_iff rest p).2 (IOCorrect.real rest p canonical ready storage geometry ha hb)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulTriton1.LogicalIOProof
