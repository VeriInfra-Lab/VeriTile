import bench.tritonbench_g.embedding_triton_kernel.Proofs.IOImplements
import bench.tritonbench_g.embedding_triton_kernel.Proofs.HeadlineLookup

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.LogicalIO
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem judgment_iff (rest : Algebra ℝ) (p : Parameters) (pid : Nat) :
    (view rest p pid ⊨{Signed p pid} (fun xs row col =>
      if p.first ≤ xs.ids row ∧ xs.ids row<p.last then xs.weightAt (xs.ids row-p.first).toNat col else 0)) ↔
    (IO.embeddingIO rest 0 Bool.false p pid).ImplementsOn (IO.signed p pid)
      (IO.outputValues p (fun ids weights row col => if valid : p.first ≤ ids row ∧ ids row<p.last then
        weights ⟨(ids row-p.first).toNat,HeadlineLookup.vocabulary_index p (ids row) valid⟩ col else 0)) := by
  have eq : (view rest p pid).lift (fun xs row col =>
      if p.first ≤ xs.ids row ∧ xs.ids row<p.last then xs.weightAt (xs.ids row-p.first).toNat col else 0) =
      IO.outputValues p (fun ids weights row col => if valid : p.first ≤ ids row ∧ ids row<p.last then
        weights ⟨(ids row-p.first).toNat,HeadlineLookup.vocabulary_index p (ids row) valid⟩ col else 0) := by
    funext raw port j
    fin_cases port
    change (if p.first ≤ IO.ids p raw j.1 ∧ IO.ids p raw j.1<p.last then
      (if h : (IO.ids p raw j.1-p.first).toNat<IO.vocabulary p then
        IO.weights p raw ⟨(IO.ids p raw j.1-p.first).toNat,h⟩ j.2.1 else 0) else 0) = _
    by_cases h : p.first ≤ IO.ids p raw j.1 ∧ IO.ids p raw j.1<p.last
    · simp [h,HeadlineLookup.vocabulary_index p (IO.ids p raw j.1) h,IO.outputValues]
    · simp [h,IO.outputValues]
  change (IO.embeddingIO rest 0 Bool.false p pid).ImplementsOn (IO.signed p pid) _ ↔ _
  rw [eq]
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.LogicalIO
