import bench.tritonbench_g.matmul_tma.IO
import bench.tritonbench_g.matmul_tma.Proofs.PublicIO
import bench.tritonbench_g.matmul_tma.Proofs.RealExecution

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.LogicalIOProof
open VeriTile.Triton FP.Structural FPValue Host
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem headline_iff (rest : Algebra ℝ) (h : Shape) (f : Flags) :
    (PublicIO.logicalIO (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot
      (params h f) f.half ⊨ fun ⟨a,b⟩ row col => ∑ k : Fin h.inner, a row k * b k col) ↔
    (PublicIO.io (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot
      (params h f) f.half).Implements
      (PublicIO.formula (params h f) f.half (fun a b i =>
        ∑ k : Fin h.inner, a (i.1,k,PUnit.unit) * b (k,i.2.1,PUnit.unit))) := by
  have target : (PublicIO.logicalIO (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot
      (params h f) f.half).lift (fun ⟨a,b⟩ row col => ∑ k : Fin h.inner, a row k * b k col) =
      PublicIO.formula (params h f) f.half (fun a b i =>
        ∑ k : Fin h.inner, a (i.1,k,PUnit.unit) * b (k,i.2.1,PUnit.unit)) := by
    funext xs port i
    fin_cases port
    rfl
  change (PublicIO.io _ _ _ _).Implements _ ↔ _
  rw [target]

theorem implements (rest : Algebra ℝ) (h : Shape) (f : Flags) :
    (PublicIO.logicalIO (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot
      (params h f) f.half ⊨ fun ⟨a,b⟩ row col => ∑ k : Fin h.inner, a row k * b k col) := by
  apply (headline_iff rest h f).2
  refine PublicIO.implements (StaticChecked.RealModel.algebra rest) StaticChecked.RealModel.dot
    (params h f) f.half (c_injective h f StaticChecked.regions) _ ?_
  intro a b i
  simp only [StaticChecked.RealModel.algebra,StaticChecked.RealModel.dot,
    StaticChecked.RealModel.contraction]
  apply Finset.sum_congr rfl
  intro k _
  rfl

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.LogicalIOProof
