import bench.repairs.tritonbench_g.index_select_cat.Proofs.Host

namespace VeriTile.Bench.Repairs.IndexSelectCat.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete mathematical source/host result for arbitrary Real payloads,
including repeated indices and N>M. No encoding or finiteness premise. -/
theorem whole_mathematical_real (rest : Algebra ℝ) (r : Request)
    (checked : checkInputs r=Bool.true) (source output : Nat → ℝ) (object : Nat)
    (incoming : VeriTile.Torch.Device) :
    let result := Host.run rest r source output object incoming
    (∀ i j,i<r.N → j<r.D → result.output (i*r.D+j)=
      Host.target r.indices (fun row col=>source (row*r.D+col)) i j) ∧
    (∀ q,r.N*r.D ≤ q → result.output q=output q) ∧
    (∀ i j,i<r.N → j<r.D → r.row i<r.M) ∧
    result.returnedObject=object ∧ result.device=incoming :=
  Host.whole_call rest r checked source output object incoming

#axiomsClean whole_mathematical_real
#specNonCircular Host.target avoiding [Host.run,Program.run,TypedGather.execution]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectCat.Correct
