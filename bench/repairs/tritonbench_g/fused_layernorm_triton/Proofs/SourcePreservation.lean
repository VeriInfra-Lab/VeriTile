import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Kernel

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePreservation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

/-- Complete values and the exterior frame determine all regional memory. -/
theorem memory_eq (M : Algebra α) (eq32 : α → α → Bool) (s t u : State α)
    (X W Bias Y Mean Rstd : RegionName) (D B : Nat) (tree : ReductionTree B) (rows : List Nat)
    (ht : GridEffect M eq32 s t X W Bias Y Mean Rstd D B tree rows)
    (hu : GridEffect M eq32 s u X W Bias Y Mean Rstd D B tree rows) : t.mem=u.mem := by
  funext q off
  by_cases hm : ∃ row ∈ rows,q=Mean ∧ off=row
  · rcases hm with ⟨row,hr,hq,ho⟩
    subst q
    rw [ho,ht.mean row hr,hu.mean row hr]
  by_cases hr : ∃ row ∈ rows,q=Rstd ∧ off=row
  · rcases hr with ⟨row,hrow,hq,ho⟩
    subst q
    rw [ho,ht.rstd row hrow,hu.rstd row hrow]
  by_cases hy : ∃ row ∈ rows,∃ col,col<D ∧ q=Y ∧ off=col+D*row
  · rcases hy with ⟨row,hrow,col,hcol,rfl,rfl⟩
    rw [ht.output row hrow col hcol,hu.output row hrow col hcol]
  have miss : ∀ row ∈ rows,rowMiss Y Mean Rstd D row q off := by
    intro row hrow
    refine ⟨?_,?_,?_⟩
    · by_cases hq : q=Mean
      · exact .inr (fun ho => hm ⟨row,hrow,hq,ho⟩)
      · exact .inl hq
    · by_cases hq : q=Rstd
      · exact .inr (fun ho => hr ⟨row,hrow,hq,ho⟩)
      · exact .inl hq
    · by_cases hq : q=Y
      · exact .inr (fun col hc ho => hy ⟨row,hrow,col,hc,hq,ho⟩)
      · exact .inl hq
  rw [ht.frame q off miss,hu.frame q off miss]

/-- The readable full source preserves the entire regional-memory behavior
of the previously retained, proof-assembled source, for every primitive algebra.
Local-register spelling is intentionally outside this observable statement. -/
theorem original (M : Algebra α) (eq32 : α → α → Bool) (s : State α)
    (X W Bias Y Mean Rstd : RegionName) (rows D : Nat) (c : Contract.Configuration)
    (sep : Separation X W Bias Y Mean Rstd) :
    ∃ t u,KernelProof.launch M eq32 X W Bias Y Mean Rstd rows D c s=some t ∧
      runRows (withEquality M eq32)
        (rowKernel X W Bias Y Mean Rstd rows D c.block (KernelProof.legacyTree c.tree)) (List.range rows) s=some u ∧
      t.mem=u.mem := by
  obtain ⟨t,ht,he⟩ := KernelProof.succeeds M eq32 X W Bias Y Mean Rstd rows D c s sep
  obtain ⟨u,hu,hf⟩ := Whole.whole_grid M eq32 s X W Bias Y Mean Rstd rows D c.block
    (KernelProof.legacyTree c.tree) c.positive sep (List.range rows)
  exact ⟨t,u,ht,hu,memory_eq M eq32 s t u X W Bias Y Mean Rstd D c.block _ _ he hf⟩

#axiomsClean original
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.SourcePreservation
