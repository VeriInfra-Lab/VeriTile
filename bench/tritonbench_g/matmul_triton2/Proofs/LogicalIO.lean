import bench.tritonbench_g.matmul_triton2.IO
import bench.tritonbench_g.matmul_triton2.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.LogicalIOProof
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
    (IO.logicalIO rest Core.realPrimitives p ⊨ target p) ↔
    (IO.matmul rest Core.realPrimitives p).Implements
      (IO.formula p (fun a b r c => RealSum.sum (fun k => a r k * b k c) p.innerSize)) := by
  have formula : (IO.logicalIO rest Core.realPrimitives p).lift (target p) =
      IO.formula p (fun a b r c => RealSum.sum (fun k => a r k * b k c) p.innerSize) := by
    funext xs port i
    fin_cases port
    change (∑ k : Fin p.innerSize, ((IO.matrixInputs p) xs).1 i.1 k *
      ((IO.matrixInputs p) xs).2 k i.2.1) = _
    change (∑ k : Fin p.innerSize, ((IO.matrixInputs p) xs).1 i.1 k *
      ((IO.matrixInputs p) xs).2 k i.2.1) =
      RealSum.sum (fun k => IO.matrix (xs (0 : Fin 2)) i.1.val k *
        IO.matrix (xs (1 : Fin 2)) k i.2.1.val) p.innerSize
    rw [sum_fin]
    apply Finset.sum_congr rfl
    intro k _
    simp [IO.matrixInputs,IO.matrix,i.1.isLt,i.2.1.isLt,k.isLt]
  change (IO.matmul _ _ _).Implements _ ↔ _
  rw [formula] <;> rfl

theorem implements (rest : Algebra ℝ) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (inner : 0<p.blockInner)
    (input : p.input≠p.output) (weight : p.weight≠p.output) :
    (IO.logicalIO rest Core.realPrimitives p ⊨ target p) := by
  apply (headline_iff rest p).2
  apply PublicIO.implements rest Core.realPrimitives p geometry inner input weight
  intro initial a b ha hb r c hr hc
  exact KernelInputs.real_reference p initial a b ha hb r c hr hc inner

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.LogicalIOProof
