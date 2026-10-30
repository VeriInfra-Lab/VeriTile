import bench.tritonbench_g.kv_cache_filling.Proofs.QuantNative
import bench.tritonbench_g.kv_cache_filling.Proofs.IONative

namespace VeriTile.Bench.Repairs.KvCacheFilling.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section
local instance pairWordInhabited : Inhabited NativeProfile.Word := ⟨FP.Value.zero .fp32⟩

/-- Every scheduled copy/INT4/INT8 job compares the two actual complete source
versions. Only the existing integer address product changes; FP32 primitives,
ordered reductions, quantization/packing, decoded inputs and all outputs stay
fixed. Both source runs are safe and successful, with exact exterior frames. -/
specification kernel_native (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (r : Request) (d : Domain r) :
    ∀ j ∈ jobs r, ∀ pid,
      Interface.original (QuantNative.ops a reductions) r j pid ≡[[]]
        Interface.indexCommuted (QuantNative.ops a reductions) r j pid := by
  intro j _ pid
  exact ProgramIO.Implementation.of_equivalentOn
    (Interface.candidate (QuantNative.ops a reductions) r j pid)
    (Interface.equivalent (QuantNative.ops a reductions) r d j pid)

#print_fp_assumptions kernel_native
#axiomsClean kernel_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.FPEquiv
