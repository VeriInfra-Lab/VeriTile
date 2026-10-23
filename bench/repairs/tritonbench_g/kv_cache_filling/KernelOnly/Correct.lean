import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.IOReal

/-! Public kernel IO and independent complete-head affine quantization. This
is the accepted isolated minimum generation; IO.lean lists all finite windows.
The complete previous raw family and Tensor claims are auxiliary Proofs/ and
TensorBridge results, with the same source execution foundation. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling.Correct
open VeriTile Triton FP.Structural
noncomputable section

def cacheIO (r : Request) (j : Job) (version : Bool) (pid : Nat) : ProgramIO ℝ ℝ :=
  Interface.programIO QuantReal.ops r j version pid

/-- Independent copy / full-head affine quantization, including paired INT4
packing and both scale/zero metadata lanes. Extrema range over all input
coordinates, before padding. -/
def formula (r : Request) (values : Interface.Inputs ℝ r) : Interface.Outputs ℝ r :=
  Interface.result r (SharedTarget.idealValue r
    (Interface.keyRow r values) (Interface.valueRow r values))

/-- Every scheduled job and both source versions: backed source accesses,
successful execution, every cache/metadata output, and exact exterior frame.
The logical input domain fixes only the two decoded scheduling integers. -/
specification whole_shared_real (r : Request) (d : Domain r) :
    Spec.Real (∀ j ∈ jobs r, ∀ version pid,
      (cacheIO r j version pid).ImplementsOn (Interface.inputDomain r j) (formula r)) := by
  intro j _ version pid
  exact Interface.real_implements r d j version pid

#specNonCircular formula avoiding [FP.Structural.exec, Shared.original, Shared.candidate,
  SharedTarget.value, QuantProfile.row4, QuantProfile.row8]
#axiomsClean whole_shared_real
#stmtSurfaceSubset whole_shared_real ⊆
  [cacheIO, formula, Domain, Interface.inputDomain, Job, Request, jobs,
   Spec.Real, ProgramIO.ImplementsOn]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.Correct
