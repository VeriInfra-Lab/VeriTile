import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.RealPhysical

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Correct
open VeriTile Triton FP.Structural
open Checked
noncomputable section
set_option maxHeartbeats 8000000

structure Result (r : Request) (a : Allocations r) (before : RealPhysical.Physical)
    (out : HostRun.Returned) : Prop where
  objects : out.tensors=(a.history,a.corrected)
  source : runGrid Real.algebra Real.dot r.params (grid r.params) (RealPhysical.input r a before)=
    some (RealPhysical.source r a before)
  history : ∀ head,head<r.B*r.H → ∀ chunk,chunk<r.dims.NT → ∀ key,key<r.K → ∀ value,value<r.V →
    RealPhysical.returned r a before (out.tensors.1.byteBase+4*(((head*r.dims.NT+chunk)*r.K+key)*r.V+value))=
      RealPhysical.history r a before head chunk key value
  corrected : ∀ head,head<r.B*r.H → ∀ time,time<r.T → ∀ value,value<r.V →
    RealPhysical.returned r a before (out.tensors.2.byteBase+4*((head*r.T+time)*r.V+value))=
      RealPhysical.corrected r a before head time value
  final : ∀ f,r.final=some f → ∀ head,head<r.B*r.H → ∀ key,key<r.K → ∀ value,value<r.V →
    RealPhysical.returned r a before (f.byteBase+4*((head*r.K+key)*r.V+value))=
      RealPhysical.history r a before head r.dims.NT key value
  exterior : ∀ address,Bytes.Outside out.tensors.1 address → Bytes.Outside out.tensors.2 address →
    (∀ f∈r.final.toList,Bytes.Outside f address) → RealPhysical.returned r a before address=before address

/-- Independent ideal Real interpretation at the same checked returned tensor
identities. It does not equate Real arithmetic with concrete FP32 bit payloads. -/
theorem returned (I : Primitives) (r : Request) (bytes : Bytes.Memory) (rt : HostRun.Runtime r)
    (caller : Nat) (out : HostRun.Returned) (success : HostRun.run I r bytes rt caller=.ok out)
    (before : RealPhysical.Physical) :
    ∃ (a : Allocations r),rt.allocate=.ok a ∧ Result r a before out := by
  have host:=HostRun.returned I r bytes rt caller out success
  obtain ⟨a,allocated,pointers,setup,identity⟩:=host.allocation
  subst out
  exact ⟨a,allocated,rfl,(RealPhysical.source_complete r host.checked a before).1,
    RealPhysical.history_read r host.checked a before,RealPhysical.corrected_read r host.checked a before,
    RealPhysical.final_read r host.checked a before,RealPhysical.exterior r a before⟩

abbrev checked_host := @HostRun.complete
abbrev full_source := @RealPhysical.source_complete
#axiomsClean returned
#axiomsClean checked_host
#axiomsClean full_source
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Correct
