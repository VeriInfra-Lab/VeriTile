import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.HostABI
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.RealReference
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.WholeVJP
import bench.repairs.tritonbench_g.chunk_retention.Whole001.Proofs.Numeric

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

/-- Complete logical tensor observations, including every admitted active index. -/
structure ReferencePost (ops : Ops α) (c : Shape) (xs : Inputs α) (scale : α)
    (tree : SumTree) (flag : Bool) (after : Heap α) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    after ⟨.output,h,0,t,j⟩=Reference.output ops c xs scale h t j
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    after ⟨.dq,h,0,t,e⟩=Reference.dq ops c xs scale h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    after ⟨.dk,h,0,t,e⟩=Reference.dk ops c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    after ⟨.dv,h,0,t,j⟩=Reference.dv ops c xs scale tree h t j
  finalState : flag=Bool.true → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    after ⟨.finalState,h,0,e,j⟩=Reference.finalState ops c xs h e j

structure RealPost (c : Shape) (xs : Inputs ℝ) (scale : ℝ) (flag : Bool) (after : Heap ℝ) : Prop where
  output : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    after ⟨.output,h,0,t,j⟩=MathTarget.output c xs scale Bool.true h t j
  query : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    after ⟨.dq,h,0,t,e⟩=MathTarget.queryGradient c xs scale Bool.true h t e
  key : ∀ h<c.heads,∀ t<c.T,∀ e<c.K,
    after ⟨.dk,h,0,t,e⟩=MathTarget.keyGradient c xs scale h t e
  value : ∀ h<c.heads,∀ t<c.T,∀ j<c.V,
    after ⟨.dv,h,0,t,j⟩=MathTarget.valueGradient c xs scale h t j
  finalState : flag=Bool.true → ∀ h<c.heads,∀ e<c.K,∀ j<c.V,
    after ⟨.finalState,h,0,e,j⟩=MathTarget.finalState c xs Bool.true h e j

theorem source_reference (ops : Ops α) (c : Shape) (scale : α) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (flag : Bool) (entry : Heap α) :
    ReferencePost ops c (inputsOfHeap entry) scale tree flag (Source.run ops c scale tree flag entry) := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro h hh t ht j hj
    exact Source.run_output ops c scale tree flag entry h t j hh ht hj
  · intro h hh t ht e he
    exact Source.run_dq ops c scale tree flag entry h t e hh ht he
  · intro h hh t ht e he
    exact Source.run_dk ops c scale tree flag entry h t e hh ht he
  · intro h hh t ht j hj
    exact Source.run_dv ops c scale tree flag entry complete h t j hh ht hj
  · intro hf h hh e he j hj
    subst flag
    exact Source.run_final ops c scale tree entry h e j hh he hj

theorem source_real (c : Shape) (scale : ℝ) (tree : SumTree)
    (complete : tree.Complete c.keyBlocks) (flag : Bool) (entry : Heap ℝ) :
    RealPost c (inputsOfHeap entry) scale flag (Source.run realOps c scale tree flag entry) := by
  have post:=source_reference realOps c scale tree complete flag entry
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro h hh t ht j hj
    exact (post.output h hh t ht j hj).trans (RealScalar.reference_output c _ scale h t j ht hj)
  · intro h hh t ht e he
    exact (post.query h hh t ht e he).trans (RealScalar.reference_query c _ scale h t e ht he)
  · intro h hh t ht e he
    exact (post.key h hh t ht e he).trans (RealScalar.reference_key c _ scale h t e ht he)
  · intro h hh t ht j hj
    exact (post.value h hh t ht j hj).trans (RealScalar.reference_value c _ scale tree complete h t j ht hj)
  · intro hf h hh e he j hj
    exact (post.finalState hf h hh e he j hj).trans (RealScalar.reference_final c _ h e j he hj)

/-- No seed gradient is exposed by this public wrapper. -/
def detachedVariation (d : MathTarget.Variation) : MathTarget.Variation :=
  {d with seed:=fun _ _ _=>0}

theorem detached_vjp (c : Shape) (xs : Inputs ℝ) (d : MathTarget.Variation) (scale : ℝ) :
    MathTarget.fullVariation c xs (detachedVariation d) scale Bool.true=
      MathTarget.gradientPairing c xs (detachedVariation d) scale Bool.true :=
  MathTarget.whole_vjp c xs (detachedVariation d) scale Bool.true

#axiomsClean source_reference
#axiomsClean source_real
#axiomsClean detached_vjp
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
