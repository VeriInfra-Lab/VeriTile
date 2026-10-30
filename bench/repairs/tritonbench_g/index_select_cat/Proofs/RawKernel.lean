import bench.repairs.tritonbench_g.index_select_cat.Proofs.TypedGather

namespace VeriTile.Bench.Repairs.IndexSelectCat.RawKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section

/-- The initial index/source-memory gather formula, successful shared source
execution and full frame, under raw output-address injectivity. -/
theorem complete {α : Type} [Inhabited α] (rest : Algebra α) (p : Layout) (s : State α)
    (unique : ∀ a b,TypedGather.active p s a → TypedGather.active p s b →
      TypedGather.outputOffset p s a=TypedGather.outputOffset p s b → a=b) :
    ∃ t,FP.Structural.exec (copyBackend rest) (kernel p) s=some t ∧
      (∀ i,TypedGather.active p s i → t.mem p.output (TypedGather.outputOffset p s i)=TypedGather.loaded p s i) ∧
      (∀ region off,region≠p.output ∨ (∀ i,TypedGather.active p s i → off≠TypedGather.outputOffset p s i) →
        t.mem region off=s.mem region off) := by
  let t := TypedGather.execution rest p s
  refine ⟨t.val,t.property,?_,?_⟩
  · intro i active
    rw [TypedGather.execution_memory]
    exact masked_readback_active _ _ _ _ _ i active unique
  · intro region off outside
    exact TypedGather.frame rest p s region off outside

#axiomsClean complete
#specNonCircular TypedGather.loaded avoiding [kernel,indexCommutedKernel,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.IndexSelectCat.RawKernel
