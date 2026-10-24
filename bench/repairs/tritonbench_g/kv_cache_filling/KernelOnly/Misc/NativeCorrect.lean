import bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Proofs.IONative

namespace VeriTile.Bench.Repairs.KvCacheFilling.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited NativeProfile.Word := ⟨FP.Value.zero .fp32⟩

/-- The original source with the admitted ordered FP32 primitive models. -/
def nativeIO (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (r : Request) (j : Job) (pid : Nat) : ProgramIO NativeProfile.Word NativeProfile.Word :=
  Interface.programIO (QuantNative.ops a reductions) r j Bool.false pid

/-- The candidate shares the original's finite IO windows and input encoding. -/
def nativeCandidate (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (r : Request) (j : Job) (pid : Nat) : ProgramIO.Variant (nativeIO a reductions r j pid) :=
  Interface.candidate (QuantNative.ops a reductions) r j pid

/-- Both actual sources succeed safely and agree on every active cache byte,
FP32 copy and scale/zero word for every scheduled job, with exact frames.
All FP32 words and ordered-reduction behavior remain admitted. -/
theorem kernel_native_io (a : NativeProfile.Arithmetic) (reductions : QuantNative.Reductions)
    (r : Request) (d : Domain r) :
    ∀ j ∈ jobs r, ∀ pid,
      (nativeIO a reductions r j pid).EquivalentOn
        (nativeCandidate a reductions r j pid) (Interface.inputDomain r j) := by
  intro j _ pid
  exact Interface.equivalent (QuantNative.ops a reductions) r d j pid

#specNonCircular Interface.target avoiding [FP.Structural.exec, Shared.original, Shared.candidate]
#axiomsClean kernel_native_io
#stmtSurfaceSubset kernel_native_io ⊆
  [Domain, instInhabitedWord, nativeCandidate, nativeIO, Interface.inputDomain, Job,
   NativeProfile.Arithmetic, NativeProfile.Word, QuantNative.Reductions, Request, jobs,
   ProgramIO.EquivalentOn]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KvCacheFilling.FPEquiv
