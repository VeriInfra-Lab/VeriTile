import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Direct
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KcacheCopyTriton.LegacyHost
noncomputable section

/-- Exact raw FP32 words from the original reshape inputs through the complete
3D signed-index source launch to the original caller cache and byte frame.
There is no floating arithmetic and no numerical correspondence premise. -/
specification kcache_copy_triton_fp32 (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (cacheObject : Nat) (before : Memory Word)
    (binding : TensorBinding m n b l t id before) :
    ∃ out,Direct.runWrapper m n b l t cacheObject before=.ok out ∧ out.returned=none ∧
      out.cacheObject=cacheObject ∧ out.cacheMetadata=m .cache ∧ CacheCopy m (config m n b) l t before out.memory ∧
      (∀ byte,¬Touched m (config m n b) l t byte → observeByte out.memory byte=observeByte before byte) :=
  Direct.correct m n b l t h cacheObject before binding

#axiomsClean kcache_copy_triton_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.LegacyHost
