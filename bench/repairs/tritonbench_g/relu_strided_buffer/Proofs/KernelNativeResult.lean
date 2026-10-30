import bench.repairs.tritonbench_g.relu_strided_buffer.Kernels
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.RawKernel
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeFP32

namespace VeriTile.Bench.Repairs.ReluStridedBuffer.KernelNativeResult
open VeriTile.Triton FP.Structural
open TritonBenchG.ReluStridedBuffer HostPlan SignedView Whole
noncomputable section
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

/-- Exact kernel equivalence for every FP32 word and both original branches.
The actual candidate constant-folds the unused integer tile-count computation.
Both kernels succeed and every typed memory cell agrees. No floating rewrite
law, finiteness condition or numerical experiment is required. -/
theorem relu_strided_buffer_folded_equiv (rest : Algebra (FP.Value .fp32))
    (c : Parameters) (s : State (FP.Value .fp32)) :
    ∃ a b, FP.Structural.exec (fp32Backend rest) (kernel c) s = some a ∧
      FP.Structural.exec (fp32Backend rest) (foldedKernel c) s = some b ∧ a.mem = b.mem := by
  obtain ⟨final,_,ran,_,_⟩ := kernel_execution (fp32Backend rest) WordTile.orderedLt
    (WordTile.backend_compare (zeroLiterals rest) .ieeeValue) c s
  exact ⟨final,final,ran,(folded_execution _ c s).symm.trans ran,rfl⟩

/-- Complete original/folded FP32 grids over raw signed views, independent
ordered-ReLU words, input/exterior frames and all active bounds. -/
theorem relu_strided_buffer_grid_equiv (rest : Algebra (FP.Value .fp32))
    (x y : SignedView.View) (n inputCapacity outputCapacity : Nat) (hn : 0<n)
    (hin : x.Valid n inputCapacity) (hout : y.Valid n outputCapacity) (hstride : y.stride≠0)
    (hsep : SignedGrid.FootprintsDisjoint x y n)
    (s : State (FP.Value .fp32)) (xs : Nat → FP.Value .fp32)
    (values : ∀ i, i < n → s.mem x.region (x.address i) = .mk .fp32 (xs i)) :
    ∃ original folded,
      runGrid (fp32Backend rest) (Raw.arguments x y n) s = some original ∧
      runFoldedGrid (fp32Backend rest) (Raw.arguments x y n) s = some folded ∧
      original.mem = folded.mem ∧
      (∀ i, i < n → original.mem y.region (y.address i) =
        Cell.mk .fp32 (fp32Result xs i)) ∧
      (∀ i, i < n → original.mem x.region (x.address i) =
        s.mem x.region (x.address i)) ∧
      (∀ r a, (r ≠ y.region ∨ ∀ i, i < n → a ≠ y.address i) →
        original.mem r a = s.mem r a) ∧
      (∀ i, i < n → x.address i < inputCapacity ∧
        y.address i < outputCapacity) := by
  obtain ⟨final,ran,result,inputFrame,frame,bounds⟩ := Raw.grid (fp32Backend rest)
    WordTile.orderedLt (WordTile.backend_compare (zeroLiterals rest) .ieeeValue)
    x y hn hin hout hstride hsep s
  refine ⟨final,final,ran,(grid_folded_execution _ _ s).symm.trans ran,rfl,?_,inputFrame,frame,bounds⟩
  intro i hi
  rw [result i hi,values i hi]
  simp [Cell.read,fp32Backend,
    WordTile.scalarResult_word _ _ (zero_literal_exact rest),fp32_result_classification,fp32Result]

#axiomsClean relu_strided_buffer_folded_equiv
#axiomsClean relu_strided_buffer_grid_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReluStridedBuffer.KernelNativeResult
