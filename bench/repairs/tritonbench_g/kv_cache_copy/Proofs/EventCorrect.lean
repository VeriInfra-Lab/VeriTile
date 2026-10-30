import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.Execution
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.KvCacheCopy.EventCorrect
noncomputable section

theorem python_checks_iff (m : Option Inputs) (b : Option Bool) (l : Lengths) (t : Tables) :
    checkArguments m b l t=.ok () ↔ Admitted m b l t := checkArguments_iff m b l t

/-- Arbitrary Real cells, the original split loop and both output layouts.
The independent target reads the initial caller values; no finite decoding. -/
specification kv_cache_copy_real (m : Inputs) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m b l t) (before : Memory ℝ) :
    CacheCopy m (config m b) l t before (launch m (config m b) l t before) :=
  launch_correct m b l t h before

#axiomsClean python_checks_iff
#axiomsClean kv_cache_copy_real
#specNonCircular CacheCopy avoiding [launch,tile,Native.launch,Native.program]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheCopy.EventCorrect
