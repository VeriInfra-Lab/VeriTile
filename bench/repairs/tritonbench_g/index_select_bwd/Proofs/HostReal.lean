import bench.repairs.tritonbench_g.index_select_bwd.Proofs.Host

/-! Optional historical whole-call API results. Kernel acceptance is exposed
in the canonical Correct/FPEquiv entries; these results impose no new required
host-lifecycle proof or numerical-conformance experiment. -/

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Correct
open VeriTile.Triton FP.Structural Logical
noncomputable section

/-- Complete direct source and host composition over arbitrary Real
payloads, including nonzero initial gradients and repeated signed indices.
This theorem uses no finite encoding or payload equality premise. -/
theorem whole_mathematical_real (rest : Algebra ℝ) (r : Request)
    (checked : checkInputs r=Bool.true) (g h : Nat → ℝ) (object : Nat)
    (device : VeriTile.Torch.Device) :
    let guard := (Native.checkInputs_iff r).mp checked
    let result := Host.run (Host.backend rest (·+·)) r guard g h object device
    (∀ q : Coord r.R r.C,result.destination (Host.flat q)=Host.target r g h q) ∧
    (∀ i,r.R*r.C ≤ i → result.destination i=g i) ∧
    (∀ q : Coord r.R r.C,(∀ i : Fin r.N,r.indices i≠(q.1.val : Int)) →
      result.destination (Host.flat q)=g (Host.flat q)) ∧
    result.incoming=h ∧ result.returned=none ∧ result.object=object ∧ result.device=device := by
  dsimp only
  refine ⟨?_,?_,?_,rfl,rfl,rfl,rfl⟩
  · intro q
    exact Host.mathematical_real rest r _ g h object device q
  · intro i hi
    exact Host.exterior_frame _ r _ g h object device i hi
  · intro q miss
    exact Host.unselected_frame _ r _ g h object device q miss

#axiomsClean whole_mathematical_real
#specNonCircular Host.target avoiding [Host.run,Program.run,Source.execution]
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Correct
