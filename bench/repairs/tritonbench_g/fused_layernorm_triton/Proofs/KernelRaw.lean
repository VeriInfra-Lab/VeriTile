import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.KernelValues

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.KernelRaw
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
noncomputable section

/-- Full shared source grid, independent normalization/statistics and the
complete exterior frame over raw regional memory. -/
theorem raw_real (rest : Algebra ℝ) (s : State ℝ)
    (X W B Y Mean Rstd : RegionName) (rows D : Nat) (positive : 0<D)
    (sep : Separation X W B Y Mean Rstd) (c : Contract.Configuration) (complete : c.tree.Complete) :
    ∃ final,KernelProof.launch (ideal rest) realEqual X W B Y Mean Rstd rows D c s=some final ∧
      (∀ row,row<rows → final.mem Mean row=.mk .real (logicalMean s X D row)) ∧
      (∀ row,row<rows → final.mem Rstd row=.mk .real (logicalInverse s X D row)) ∧
      (∀ row,row<rows → ∀ col,col<D → final.mem Y (col+D*row)=.mk .bf16 (logicalOutput s X W B D row col)) ∧
      (∀ q off,(∀ row,row<rows → rowMiss Y Mean Rstd D row q off) → final.mem q off=s.mem q off) := by
  obtain ⟨final,run,effect⟩ := KernelProof.succeeds (ideal rest) realEqual X W B Y Mean Rstd rows D c s sep
  refine ⟨final,run,?_,?_,?_,?_⟩
  · intro row hr
    rw [effect.mean row (List.mem_range.mpr hr),
      (row_real_mean_m2 rest s X D c.block row positive c.positive _ (KernelProof.complete _ complete)).1]
  · intro row hr
    rw [effect.rstd row (List.mem_range.mpr hr),
      row_real_inverse rest s X D c.block row positive c.positive _ (KernelProof.complete _ complete)]
  · intro row hr col hc
    rw [effect.output row (List.mem_range.mpr hr) col hc,
      row_real_output rest s X W B D c.block row col positive c.positive _ (KernelProof.complete _ complete)]
  · intro q off miss
    exact effect.frame q off (fun row hr => miss row (List.mem_range.mp hr))



#axiomsClean raw_real
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.KernelRaw
