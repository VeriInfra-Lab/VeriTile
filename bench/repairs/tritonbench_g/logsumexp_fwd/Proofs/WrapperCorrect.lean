import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.WholeReal

namespace VeriTile.Bench.Repairs.LogsumexpFwd.WrapperCorrect
open VeriTile.Triton FP.Structural
noncomputable section
open Classical

def realProfile (warps : Nat) (choice : warps∈Host.warpChoices) : Host.Profile (WithBot ℝ) :=
  ⟨warps,choice,fun _ => RealModel.engine⟩

/-- Whole checked source call and independent mathematical row formula,
including every tuning probe, all returned cells, output metadata and the
complete input/exterior frame. The Real interpretation is a separate model;
native floating arithmetic is covered by FPEquiv, not assumed exact here. -/
theorem logsumexp_real (a : Arguments) (h : InputContract a) (scale : ℝ)
    (probes : List (Host.Profile (WithBot ℝ))) (warps : Nat) (choice : warps∈Host.warpChoices)
    (input : Nat → ℝ) (s : State (WithBot ℝ))
    (binding : ∀ i,i<a.input.numel → (s.mem "input" i).read .real=(input i : WithBot ℝ)) :
    ∃ t out,Host.run RealModel.library a (scale : WithBot ℝ) probes (realProfile warps choice) s=some (t,out) ∧
      out.shape=a.input.shape.dropLast ∧ out.dtype=a.outputDType ∧ out.device=a.input.device ∧
      (∀ row,row<a.rows → out.value row=Host.castOutput RealModel.library a
        (RealModel.formula input row a.columns a.hasScale scale)) ∧
      (∀ (r : RegionName) (i : Nat),(r≠"scratch" ∨ a.rows*a.blocks ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,out,he,hs,hd,hdev,hv,hf⟩ := Host.complete RealModel.library a h (scale : WithBot ℝ)
    probes (realProfile warps choice) s
  refine ⟨t,out,he,hs,hd,hdev,?_,hf⟩
  intro row hr
  rw [hv row hr,Host.target_local RealModel.library a h _ _ _ _ row hr binding]
  change Host.castOutput RealModel.library a (Host.target RealModel.library a (fun _ => RealModel.engine) _ _ _)=_
  rw [RealModel.target_formula]

#axiomsClean logsumexp_real
end
end VeriTile.Bench.Repairs.LogsumexpFwd.WrapperCorrect
