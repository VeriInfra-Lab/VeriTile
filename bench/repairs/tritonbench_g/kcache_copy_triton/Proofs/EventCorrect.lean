import bench.repairs.tritonbench_g.kcache_copy_triton.Proofs.Execution
import VeriTile.Meta.Specification
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.EventCorrect
noncomputable section

theorem python_checks_iff (m : Option Inputs) (n : Option Int) (b : Option Bool) (l : Lengths) (t : Tables) :
    checkArguments m n b l t=.ok () ↔ Admitted m n b l t := checkArguments_iff m n b l t

/-- Arbitrary Real cells, the original three-dimensional grid and both output layouts.
The independent target reads the initial caller values; no finite decoding. -/
specification kcache_copy_triton_real (m : Inputs) (n : Nat) (b : Bool) (l : Lengths) (t : Tables)
    (h : InputContract m n b l t) (before : Memory ℝ) :
    CacheCopy m (config m n b) l t before (launch m (config m n b) l t before) :=
  launch_correct m n b l t h before

#axiomsClean python_checks_iff
#axiomsClean kcache_copy_triton_real
#specNonCircular CacheCopy avoiding [launch,tile,Native.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KcacheCopyTriton.EventCorrect
