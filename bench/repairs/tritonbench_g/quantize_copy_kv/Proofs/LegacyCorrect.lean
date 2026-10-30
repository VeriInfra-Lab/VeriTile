import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.RealGrid
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.Correct
noncomputable section

theorem python_checks_iff (m : Option Inputs) (w : Nat → MetadataWord) :
    checkArguments m w=.ok () ↔ Admitted m w := checkArguments_iff m w

/-- Arbitrary independent Real source/scale cells and exact signed-byte codes.
Neither the input nor the scale target passes through a finite byte decoder. -/
specification quantize_copy_kv_real (m : Inputs) (w : Nat → MetadataWord) (h : InputContract m w)
    (objects : Role → Nat) (plan : Plan (request m w)) (before : Mathematical.State) :
    Mathematical.Quantized (request m w) (directLayout m objects) before
      (Mathematical.run (request m w) plan (directLayout m objects) before) :=
  Mathematical.correct (request m w) (checked_domain m w h) plan (directLayout m objects) before

#axiomsClean python_checks_iff
#axiomsClean quantize_copy_kv_real
#specNonCircular Mathematical.Quantized avoiding [Mathematical.run,sourceRegisters,runGrid]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.Correct
