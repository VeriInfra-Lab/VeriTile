import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.RealResult

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open Checked Physical Wrapper
noncomputable section

/-- Complete guarded arbitrary-Real wrapper result. The derivative includes
both returned tensors; the original initial state remains detached. -/
theorem fused_recurrent_hgrn_correct {M β : Type} (api : API M ℝ β) (c : Shape) (U S V hasDo : Bool)
    (runtime : Runtime c U S V) (accepted : check c U S V runtime=Bool.true) (BDF BDB : Nat)
    (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true) (initial incoming : M)
    (saved : SavedStorage api runtime (mathematicalForward api c U S V runtime BDF initial).memory incoming)
    (zero : MissingZero api 0 c U S V hasDo runtime incoming) :
    Public.RealClaim api c U S V hasDo runtime BDF BDB initial incoming
      (mathematicalForward api c U S V runtime BDF initial)
      (mathematicalBackward api c U S V hasDo runtime BDF BDB initial incoming) :=
  Public.real_complete api c U S V hasDo runtime accepted BDF BDB tileF tileB initial incoming saved zero

#axiomsClean fused_recurrent_hgrn_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
