import bench.tritonbench_g.matrix_reduction.IO
import bench.tritonbench_g.matrix_reduction.Proofs.IOReal

namespace VeriTile.Bench.TritonBenchG.MatrixReduction.LogicalIOProof
open VeriTile.Triton TypedSource
open scoped VeriTile.Triton.LogicalIO
noncomputable section

theorem headline_iff (storage : Storage) (M N : Nat) (hN : 0<N) :
    (rowMaximumIO storage M N ⊨ fun xs row _ =>
      (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩,Finset.mem_univ _⟩ (fun col => xs row col)) ↔
    (matrixReductionIO storage M N).Implements (fun _ xs i =>
      (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩,Finset.mem_univ _⟩ (fun j => xs (i.1,j,()))) := Iff.rfl

theorem implements (storage : Storage) (M N : Nat) (hN : 0<N) :
    (rowMaximumIO storage M N ⊨ fun xs row _ =>
      (Finset.univ : Finset (Fin N)).sup' ⟨⟨0,hN⟩,Finset.mem_univ _⟩ (fun col => xs row col)) :=
  (headline_iff storage M N hN).2 (IOReal.correct storage M N hN)

#axiomsClean headline_iff
#axiomsClean implements
end
end VeriTile.Bench.TritonBenchG.MatrixReduction.LogicalIOProof
