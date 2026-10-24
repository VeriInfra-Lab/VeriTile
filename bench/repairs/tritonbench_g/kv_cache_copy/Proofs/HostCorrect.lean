import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Direct
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KvCacheCopy.LegacyHost
noncomputable section

/-- Exact raw FP32 payloads for the complete original signed-index kernel,
initial metadata, both caller objects, None return and full backing-byte frame.
The copy performs no floating arithmetic and needs no numerical premise. -/
specification kv_cache_copy_fp32 (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (keyObject valueObject : Nat) (before : Memory Word)
    (binding : TensorBinding m b l t id before) :
    ∃ out,Direct.runWrapper m b l t keyObject valueObject before=.ok out ∧ out.returned=none ∧
      out.keyObject=keyObject ∧ out.valueObject=valueObject ∧ out.keyMetadata=m .keyCache ∧
      out.valueMetadata=m .valueCache ∧ CacheCopy m (config m b) l t before out.memory ∧
      (∀ byte,¬Touched m (config m b) l t byte → observeByte out.memory byte=observeByte before byte) :=
  Direct.correct m b l t h keyObject valueObject before binding

#axiomsClean kv_cache_copy_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.LegacyHost
