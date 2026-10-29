import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.HostReturned
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.SnapshotReal
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeFinalAdvance

namespace VeriTile.Bench.Repairs.ContextAttnFwd
open VeriTile.Triton FP.Structural NativeGraph
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula
noncomputable section

/-- Concrete fixed-plan connection: actual source sites and headers at every
current-memory prefix; complete direct signed loop including final test;
metadata and masked data byte events; formed relative offsets. No floating
arithmetic or whole-output equality is a premise of these structural facts. -/
structure MachineFacts (c : Host.Call) (B : Native.Bytes extent) (before : Physical.Memory)
    (abi : Native.ABI c) : Prop where
  source : Native.GridBridge c abi (backend (NativeSM90.plan c.layout.BM c.layout.D)) (Snapshot.initial c B before)
  loop : ∀ idx (s : State Expr),
    Native.bitLoop ((backend (NativeSM90.plan c.layout.BM c.layout.D)).withDefaultPrecision .fp32)
      (FPSuccess.parts c.layout c.regions (Host.scale c)).body c.layout.BN
      (Native.stopBits (c.launchAt idx).block c.layout.BM (Native.seqAt c idx) (c.prompt (Native.batchAt c idx)))
      (Native.countAt c idx+1) (Native.word 0) s =
    rangeInt ((backend (NativeSM90.plan c.layout.BM c.layout.D)).withDefaultPrecision .fp32)
      "start_n" 0 (Native.stopAt c idx : Int) c.layout.BN
      (FPSuccess.parts c.layout c.regions (Host.scale c)).body s
  metadata : ∀ rg,rg∈[c.regions.start,c.regions.sequence,c.regions.prompt] → ∀ idx,
    Native.metadataEvent c B rg idx=some (B.base rg+4*Native.batchAt c idx)
  query : ∀ role,role=Native.DataRole.query ∨ role=.output → ∀ idx (i : Fin c.layout.BM) (d : Fin c.layout.D),
    Native.queryEvent c B role idx i d=if queryRow c.layout (c.launchAt idx) i<c.length (Native.batchAt c idx)
      then some (B.base (role.region c)+4*Native.queryOffset c role idx i d) else none
  cache : ∀ role,role=Native.DataRole.key ∨ role=.value → ∀ idx step (j : Fin c.layout.BN) (d : Fin c.layout.D),
    step<Native.countAt c idx → Native.cacheEvent c B role idx step j d=
      if step*c.layout.BN+j.val<Native.endAt c idx
      then some (B.base (role.region c)+4*Native.cacheOffset c role idx step j d) else none
  formedQuery : ∀ idx (i : Fin c.layout.BM) (d : Fin c.layout.D),
    Native.queryOffset c .query idx i d<2^61 ∧ Native.queryOffset c .output idx i d<2^61
  formedCache : ∀ idx step (j : Fin c.layout.BN) (d : Fin c.layout.D),step<Native.countAt c idx →
    Native.cacheOffset c .key idx step j d<2^61 ∧ Native.cacheOffset c .value idx step j d<2^61
  finalCounter : ∀ idx,
    (Native.word (Native.countAt c idx*c.layout.BN)).toInt=((Native.countAt c idx*c.layout.BN : Nat) : Int) ∧
    ¬(Native.word (Native.countAt c idx*c.layout.BN)).slt (Native.word (Native.stopAt c idx))=Bool.true
  pointerAdvance : ∀ base offset stride,
    Native.ptr base 4 (Native.word offset)+Native.word (4*(c.layout.BN*stride))=
      Native.ptr base 4 (Native.word (offset+c.layout.BN*stride))
  inputs : ∀ rg,rg∈[c.regions.query,c.regions.key,c.regions.value,c.regions.out] → ∀ off,
    ((Snapshot.initial c B before).mem rg off).read .real=.leaf (Snapshot.load32 before (B.base rg+4*off))

 theorem machine_facts (c : Host.Call) (B : Native.Bytes extent) (before : Physical.Memory)
    (hc : Geometry.Checked c extent) (hm : Snapshot.Metadata c B before)
    (roles : Snapshot.LogicalRoles c) (sep : FPSuccess.Separate c)
    (abi : Native.ABI c) (formed : FormedBounds.Checked c) : MachineFacts c B before abi := by
  refine ⟨Returned.source_native c B before abi hm sep,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩
  · intro idx s
    exact Native.native_loop c abi idx _ s
  · exact Native.metadata_event c B hc abi
  · exact Native.query_event c B hc abi
  · exact Native.cache_event c B hc abi
  · exact FormedBounds.query_formed c hc formed
  · exact FormedBounds.cache_formed c hc formed
  · exact Native.final_counter c abi
  · intro base offset stride
    exact Native.pointer_advance base offset stride c.layout.BN
  · intro rg hr off
    exact Snapshot.floating c B before rg off (roles rg hr)

#print axioms machine_facts
end
end VeriTile.Bench.Repairs.ContextAttnFwd
