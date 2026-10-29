import bench.repairs.tritonbench_g.chunk_delta_fwd.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.FPEquiv
open VeriTile Triton FP.Structural
open Checked Checked.NativePointers
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
noncomputable section
set_option maxHeartbeats 8000000

/-- A checked return, the complete typed source on original physical input
words, derived finite pointers, every returned FP32 word, and caller bytes. -/
structure Result (I : Primitives) (r : Request) (checked : r.check=Bool.true)
    (a : Allocations r) (before : Bytes.Memory) (caller : Nat) (out : HostRun.Returned) : Prop where
  objects : out.tensors=(a.history,a.corrected)
  restored : out.currentDevice=caller
  owner : out.launchDevice=(if r.B*r.H*r.V=0 then none else some r.key.device)
  source : runGrid Native.algebra Native.dot r.params (grid r.params) (input r a before)=some (Checked.source r a before)
  inputs : ∀ region t,descriptor r a region=some t → ∀ i,
    (input r a before region i).read .real=.bits (Bytes.read before (t.byteBase+4*i))
  pointers : ∀ (x : Coordinates r.dims) site t,descriptor r a (region site)=some t →
    (baseWord r.dims x site).toInt=(base r.dims x site : Int) ∧
    (blockWord r.dims x site).toInt=(blockOffset r.dims x site : Int) ∧
    (word64 t.byteBase+((baseWord r.dims x site).signExtend 64+blockWord r.dims x site)*word64 4).toNat=
      t.byteBase+4*offset r.dims x site ∧
    (t.byteBase+4*offset r.dims x site)%4=0 ∧ t.byteBase+4*offset r.dims x site+4<2^63
  history : ∀ head,head<r.B*r.H → ∀ chunk,chunk<r.dims.NT → ∀ key (hk : key<r.K),∀ value,value<r.V →
    Bytes.read out.memory (out.tensors.1.byteBase+4*(((head*r.dims.NT+chunk)*r.K+key)*r.V+value))=
      (Native.historyGraph r.params (r.shape checked) (bits r a before) head chunk key value hk).eval I.isa
  corrected : ∀ head,head<r.B*r.H → ∀ time,time<r.T → ∀ value,value<r.V →
    Bytes.read out.memory (out.tensors.2.byteBase+4*((head*r.T+time)*r.V+value))=
      (Native.correctedGraph r.params (r.shape checked) (bits r a before) head time value).eval I.isa
  final : ∀ f,r.final=some f → ∀ head,head<r.B*r.H → ∀ key (hk : key<r.K),∀ value,value<r.V →
    Bytes.read out.memory (f.byteBase+4*((head*r.K+key)*r.V+value))=
      (Native.historyGraph r.params (r.shape checked) (bits r a before) head r.dims.NT key value hk).eval I.isa
  callerFrame : ∀ t∈r.inputs,∀ byte,byte<4*t.count → out.memory (t.byteBase+byte)=before (t.byteBase+byte)
  exterior : ∀ address,Bytes.Outside out.tensors.1 address → Bytes.Outside out.tensors.2 address →
    (∀ f∈r.final.toList,Bytes.Outside f address) → out.memory address=before address

theorem returned (I : Primitives) (r : Request) (before : Bytes.Memory) (rt : HostRun.Runtime r)
    (caller : Nat) (out : HostRun.Returned) (success : HostRun.run I r before rt caller=.ok out) :
    ∃ (checked : r.check=Bool.true) (a : Allocations r),rt.allocate=.ok a ∧ a.check=Bool.true ∧
      (r.B*r.H*r.V≠0 → rt.setup a r.key.device rt.warps=.ok ()) ∧ Result I r checked a before caller out := by
  have host:=HostRun.returned I r before rt caller out success
  obtain ⟨a,allocated,pointers,setup,identity⟩:=host.allocation
  subst out
  refine ⟨host.checked,a,allocated,pointers,setup,rfl,rfl,rfl,
    (source_complete r host.checked a before).1,input_word r a before,
    HostRun.native_addresses r host.checked a pointers,?_,?_,?_,?_,?_⟩
  · exact history_word I r host.checked a before
  · exact corrected_word I r host.checked a before
  · exact final_word I r host.checked a before
  · exact caller_frame I r host.checked a before (Checked.source r a before)
  · exact Checked.exterior I r a before (Checked.source r a before)

/-- Generic representation transport is separate from concrete word loads,
which are identity leaves. The numerical families are fixed scalar operations. -/
theorem local_numerical_transport {α β : Type} (f : α → β)
    (A : VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph.Primitives α)
    (B : VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph.Primitives β)
    (ports : VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph.ScalarTransport f A B)
    (graph : VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.Graph) :
    f (graph.eval A.isa)=graph.eval B.isa := by
  apply VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.transport
  exact ⟨ports.bits,ports.literal,ports.binary,ports.unary,ports.cast,ports.fma,fun _ _ _ _ _=>rfl⟩

abbrev checked_host := @HostRun.complete
abbrev source_time := @NativePointers.time_native
abbrev source_loops := @NativePointers.loop_bounds
abbrev signed_range := @NativePointers.range
abbrev state_coordinates := @NativePointers.stateCoordinates
abbrev inner_coordinates := @NativePointers.innerCoordinates
abbrev scalar_byte_stores := @VeriTile.Bench.Repairs.ChunkGlaSimple.ByteMemory.stores_render
#axiomsClean returned
#axiomsClean local_numerical_transport
#axiomsClean source_time
#axiomsClean source_loops
#axiomsClean scalar_byte_stores
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.FPEquiv
