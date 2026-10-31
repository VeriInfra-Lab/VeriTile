import bench.tritonbench_g.matmul_triton2.IO
import bench.tritonbench_g.matmul_triton2.Proofs.SourceIOCorrect

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.LogicalIOProof
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem sum_fin (f : Nat → ℝ) (n : Nat) : VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum f n = ∑ i : Fin n, f i.val := by
  induction n with
  | zero => simp [VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum]
  | succ n ih => simpa [VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum,Fin.sum_univ_castSucc] using congrArg (fun x => x+f n) ih

def target (p : Core.Parameters)
    (ab : (Fin p.rows → Fin p.innerSize → ℝ) × (Fin p.innerSize → Fin p.columns → ℝ))
    (row : Fin p.rows) (col : Fin p.columns) : ℝ :=
  ∑ k : Fin p.innerSize, ab.1 row k * ab.2 k col

theorem headline_iff (rest : Algebra ℝ) (p : Core.Parameters) :
    (IO.logicalIO rest Core.realPrimitives p ⊨ target p) ↔
    (IO.matmul rest Core.realPrimitives p).Implements (IO.realTarget p) := by
  have formula : (IO.logicalIO rest Core.realPrimitives p).lift (target p) =
      IO.realTarget p := by
    funext xs port i
    fin_cases port
    change (∑ k : Fin p.innerSize, ((IO.matrixInputs p) xs).1 i.1 k *
      ((IO.matrixInputs p) xs).2 k i.2.1) = _
    change (∑ k : Fin p.innerSize, ((IO.matrixInputs p) xs).1 i.1 k *
      ((IO.matrixInputs p) xs).2 k i.2.1) =
      VeriTile.Bench.Repairs.MatmulTriton1.Revision001.RealSum.sum (fun k => IO.left p xs 0 i.1 k * IO.right p xs 0 i.2.1 k) p.innerSize
    rw [sum_fin]
    apply Finset.sum_congr rfl
    intro k _
    simp [IO.matrixInputs,IO.left,IO.right,
      VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.left,
      VeriTile.Bench.Repairs.MatmulTriton1.MatmulIO.right,k.isLt]
  change (IO.matmul _ _ _).Implements _ ↔ _
  rw [formula] <;> rfl

theorem implements (rest : Algebra ℝ) (p : Core.Parameters)
    (scope : SourceRuntime.SourceBounds p) (inner : 0<p.blockInner) (geometry : Geometry.Valid (Core.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) :
    (IO.logicalIO rest Core.realPrimitives p ⊨ target p) :=
  (headline_iff rest p).2 (SourceIOProof.real rest p scope inner geometry ha hb)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.LogicalIOProof
