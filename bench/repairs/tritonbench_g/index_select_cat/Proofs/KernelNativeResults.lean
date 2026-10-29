import bench.repairs.tritonbench_g.index_select_cat.Kernels
import bench.repairs.tritonbench_g.index_select_cat.Proofs.RawKernel
import bench.repairs.tritonbench_g.index_select_cat.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.IndexSelectCat.KernelNativeResults
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
open scoped VeriTile.Spec
noncomputable section

abbrev assumptions := Rewrite.assumptions
abbrev original := Rewrite.original
abbrev indexCommuted := Rewrite.indexCommuted

theorem index_select_index_commuted_equiv (p : Layout) :
    original p ≡[assumptions] indexCommuted p := Rewrite.equivalent p

/-- Both exact source variants succeed and agree on every typed cell; active
outputs gather the initial source memory and every exterior cell is preserved. -/
theorem kernel_equiv (rest : Algebra (BitVec 32)) (p : Layout) (s : State (BitVec 32))
    (unique : ∀ a b,TypedGather.active p s a → TypedGather.active p s b →
      TypedGather.outputOffset p s a=TypedGather.outputOffset p s b → a=b) :
    ∃ left right,FP.Structural.exec (copyBackend rest) (original p).kernel s=some left ∧
      FP.Structural.exec (copyBackend rest) (indexCommuted p).kernel s=some right ∧ left.mem=right.mem ∧
      (∀ i,TypedGather.active p s i → left.mem p.output (TypedGather.outputOffset p s i)=TypedGather.loaded p s i) ∧
      (∀ region off,region≠p.output ∨ (∀ i,TypedGather.active p s i → off≠TypedGather.outputOffset p s i) →
        left.mem region off=s.mem region off) := by
  obtain ⟨t,run,values,frame⟩ := RawKernel.complete rest p s unique
  refine ⟨t,t,run,?_,rfl,values,frame⟩
  change FP.Structural.exec (copyBackend rest) (indexCommutedKernel p) s=some t
  rw [←Rewrite.execution_eq]
  exact run

#axiomsClean kernel_equiv

theorem original_source (p : Layout) : (original p).kernel = kernel p := rfl
theorem candidate_source (p : Layout) : (indexCommuted p).kernel = indexCommutedKernel p := rfl

#print_fp_assumptions index_select_index_commuted_equiv
#axiomsClean original_source
#axiomsClean candidate_source
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectCat.KernelNativeResults
