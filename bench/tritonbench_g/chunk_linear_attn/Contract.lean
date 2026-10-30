import bench.tritonbench_g.chunk_linear_attn.Proofs.Admission

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Contract
open VeriTile Triton

/-- The unchanged Python wrapper's public guards. Raw Python classification
flags are API observations; the typed descriptor API supplies shape, strides,
device, dtype, views and scale origin. No computation result is admitted here. -/
abbrev InputContract := Admission.Guards
abbrev check := Admission.check

theorem check_iff (c : Host.Invocation) (flags : Admission.RuntimeFlags) :
    check c flags=Bool.true ↔ InputContract c flags := Admission.check_iff c flags

/-- Pointer assertions are observed after the wrapper prepares kernel buffers.
They apply to actual addresses, including the dense cotangent buffers. -/
abbrev PointerContract := Admission.PointerGuards
abbrev checkPointers := Admission.checkPointers

theorem pointers_iff (c : Host.Invocation) (a : Host.Allocations)
    (p : Admission.PointerObservation c a) :
    checkPointers c a p=Bool.true ↔ PointerContract c a p := by
  simp [checkPointers,Admission.checkPointers,PointerContract]

theorem admitted (c : Host.Invocation) (flags : Admission.RuntimeFlags)
    (h : InputContract c flags) (api : Admission.TensorAPI c) : Host.Contract c :=
  Admission.contract c flags ((check_iff c flags).mpr h) api

#axiomsClean check_iff
#axiomsClean pointers_iff
#axiomsClean admitted
#specNonCircular InputContract avoiding [Host.execute,executeFour,gridMemory,Real.Call.result]
end VeriTile.Bench.Repairs.ChunkLinearAttn.Contract
