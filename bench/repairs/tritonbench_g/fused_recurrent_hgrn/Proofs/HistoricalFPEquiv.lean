import bench.repairs.tritonbench_g.fused_recurrent_hgrn.Proofs.FPResult

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn
open Checked Physical Wrapper Public
noncomputable section

/-- Complete FP32 word/byte wrapper result under local add/mul/FMA laws and
value-independent compiler choices. Structural obligations follow from guards. -/
theorem fused_recurrent_hgrn_fp32 {M : Type} (api : API M Word Byte) (codec : Codec api)
    (p : Native.Instructions) (laws : Native.LocalConformance p) (schedule : Native.Schedule)
    (c : Shape) (U S V hasDo : Bool) (runtime : Runtime c U S V)
    (accepted : check c U S V runtime=Bool.true) (BDF BDB : Nat)
    (tileF : tileCheck BDF=Bool.true) (tileB : tileCheck BDB=Bool.true) (initial incoming : M)
    (saved : SavedStorage api runtime (nativeForward api p schedule c U S V runtime BDF initial).memory incoming)
    (zero : MissingZero api 0 c U S V hasDo runtime incoming) :
    FPClaim api p schedule c U S V hasDo runtime BDF BDB initial incoming
      (nativeForward api p schedule c U S V runtime BDF initial)
      (nativeBackward api p schedule c U S V hasDo runtime BDF BDB initial incoming) :=
  fp_complete api codec p laws schedule c U S V hasDo runtime accepted BDF BDB tileF tileB initial incoming saved zero

#axiomsClean fused_recurrent_hgrn_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn
