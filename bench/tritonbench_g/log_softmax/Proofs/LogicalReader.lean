import bench.tritonbench_g.log_softmax.IO
import bench.tritonbench_g.log_softmax.Proofs.PublicIO
namespace VeriTile.Bench.Repairs.LogSoftmax.ReaderProof
open VeriTile.Triton FP.Structural Address ReaderIO
open scoped BigOperators LogicalProgramIO
noncomputable section

theorem vector_eq (xs : Values ℝ .real [M*N*K]) (q : Coord M N K) :
    IO.vector 0 xs (linear q) = tensor M N K xs q.1 q.2.1 q.2.2 := by
  simp [IO.vector,linear_bound,tensor]

theorem offset_value (xs : Values ℝ .real [M*N*K]) (m : Fin M) (n : Fin N) (k : Fin K) :
    IO.vector 0 xs (offset N K m.val n.val k.val) = tensor M N K xs m n k :=
  vector_eq xs (m,n,k)

theorem forward_target (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    (ReaderIO.forward M N K BM plan).lift (fun xs m n k => xs m n k - Real.log (∑ j : Fin N, Real.exp (xs m j k))) = Reference.forwardFormula M N K BM plan := by
  funext xs o j
  fin_cases o
  change _ = Reference.forward M N K (xs (0 : Fin 1)) j
  simp only [Reference.forward,RealProof.targetForward]
  simp only [offset_value]
  rfl

theorem backward_target (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    (ReaderIO.backward M N K BM plan).lift (fun xs m n k => xs.upstream m n k - Real.exp (xs.saved m n k) * ∑ j : Fin N, xs.upstream m j k) = Reference.backwardFormula M N K BM plan := by
  funext xs o j
  fin_cases o
  change _ = Reference.backward M N K (xs (0 : Fin 2)) (xs (1 : Fin 2)) j
  simp only [Reference.backward,RealProof.targetBackward]
  rw [← Fin.sum_univ_eq_sum_range]
  simp only [offset_value]
  rfl

/-- Exact full-grid judgment equivalences, including safety and exterior frame. -/
theorem forward_iff (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    (ReaderIO.forward M N K BM plan ⊨ fun xs m n k => xs m n k - Real.log (∑ j : Fin N, Real.exp (xs m j k))) ↔
      (IO.realForward M N K BM plan).Implements (Reference.forwardFormula M N K BM plan) := by
  rw [LogicalProgramIO.implements_iff,forward_target] <;> rfl

theorem backward_iff (M N K BM : Nat) (plan : Plan N (Grid.blockColumns N)) :
    (ReaderIO.backward M N K BM plan ⊨ fun xs m n k => xs.upstream m n k - Real.exp (xs.saved m n k) * ∑ j : Fin N, xs.upstream m j k) ↔
      (IO.realBackward M N K BM plan).Implements (Reference.backwardFormula M N K BM plan) := by
  rw [LogicalProgramIO.implements_iff,backward_target] <;> rfl

theorem forward_correct (M N K BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan N (Grid.blockColumns N)) : (ReaderIO.forward M N K BM plan ⊨ fun xs m n k => xs m n k - Real.log (∑ j : Fin N, Real.exp (xs m j k))) :=
  (forward_iff M N K BM plan).mpr (PublicIO.forward M N K BM retained plan)

theorem backward_correct (M N K BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan N (Grid.blockColumns N)) : (ReaderIO.backward M N K BM plan ⊨ fun xs m n k => xs.upstream m n k - Real.exp (xs.saved m n k) * ∑ j : Fin N, xs.upstream m j k) :=
  (backward_iff M N K BM plan).mpr (PublicIO.backward M N K BM retained plan)
end
end VeriTile.Bench.Repairs.LogSoftmax.ReaderProof
