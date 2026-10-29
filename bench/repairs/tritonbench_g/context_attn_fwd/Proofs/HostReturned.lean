import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Snapshot
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeVocabularyRun
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeProgramBridge
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeMetadata
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FormedBounds
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeStructure

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Returned
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section
set_option maxHeartbeats 4000000

/-- Checked host branch: zero batches and all-zero query lengths do not launch. -/
def Needed (c : Host.Call) : Prop := ∃ b,b<c.batches ∧ 0<c.length b

local instance (c : Host.Call) : Decidable (Needed c) := Classical.propDecidable _

def order (c : Host.Call) : List (GridIndex c.grid) :=
  if Needed c then FPSuccess.allPrograms c else []

theorem complete_order (c : Host.Call) (h : Needed c) (idx : GridIndex c.grid) : idx∈order c := by
  simpa [order,h] using FPSuccess.allPrograms_complete c idx

theorem empty_lengths (c : Host.Call) (h : ¬Needed c) : ∀ b,b<c.batches → c.length b=0 := by
  intro b hb
  by_contra he
  exact h ⟨b,hb,Nat.pos_of_ne_zero he⟩

/-- Handles and devices are host observations. The supplied output handle is
retained; ordinary torch.cuda.device enter/exit restores callerDevice. -/
structure Context where
  suppliedOutput : Nat
  ownerDevice : Nat
  callerDevice : Nat

structure Result where
  graph : State Expr
  bytes : Physical.Memory
  returned : Option Unit
  suppliedOutput : Nat
  currentDevice : Nat
  launchDevice : Option Nat

def invoke (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (ctx : Context) (before : Physical.Memory) : Option Result := do
  let t ← FPSuccess.runGrid c (backend (NativeSM90.plan c.layout.BM c.layout.D))
    (order c) (Snapshot.initial c B before)
  return ⟨t,Physical.stores c B ops t before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩

/-- Normal completed host call, full dynamic source grid, supported graph,
actual output bytes, and complete byte/cell frame. Metadata observations come
from the same physical byte snapshot; readonly role aliases are unrestricted. -/
theorem whole (c : Host.Call) (A : FlatAlloc) (s : BlockState)
    (B : Native.Bytes A.extent) (ops : ScalarOps) (ctx : Context) (before : Physical.Memory)
    (hc : Geometry.Checked c A.extent) (hm : Geometry.Metadata c s)
    (hbytes : Snapshot.Metadata c B before) (sep : FPSuccess.Separate c)
    (physicalSep : Physical.Separate c B) :
    ∃ result,invoke c B ops ctx before=some result ∧
      result.returned=none ∧ result.suppliedOutput=ctx.suppliedOutput ∧
      result.currentDevice=ctx.callerDevice ∧
      result.launchDevice=(if Needed c then some ctx.ownerDevice else none) ∧
      FPSuccess.runGrid c (backend (NativeSM90.plan c.layout.BM c.layout.D)) (order c)
        (Snapshot.initial c B before)=some result.graph ∧
      GoodMemory result.graph ∧ FPSuccess.Metadata c result.graph ∧
      (∀ i : Physical.Lane c,Physical.active c i →
        result.bytes (Physical.address c B i)=Physical.byte (Physical.outputWord c ops result.graph i) i.2.2) ∧
      (∀ a,(¬∃ i : Physical.Lane c,Physical.active c i ∧ Physical.address c B i=a) → result.bytes a=before a) ∧
      (∀ rg,rg∈Physical.reads c → ∀ off,off<A.extent rg → ∀ j : Fin 4,
        result.bytes (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val)) ∧
      (∀ rg off,¬FPSuccess.writes c (order c) rg off →
        result.graph.mem rg off=(Snapshot.initial c B before).mem rg off) ∧
      (¬Needed c → result.bytes=before) := by
  let plan := NativeSM90.plan c.layout.BM c.layout.D
  obtain ⟨t,ht,hmeta,hframe⟩ := FPSuccess.grid_success_frame c (backend plan)
    (dots.support (scalar plan) c.layout) sep (order c) (Snapshot.initial c B before)
    (Snapshot.metadata c B before hbytes)
  obtain ⟨u,hu,hgood⟩ := grid_good c plan (order c) (Snapshot.initial c B before)
    (Snapshot.good c B before)
  have he : t=u := Option.some.inj (ht.symm.trans hu)
  subst u
  dsimp only [plan] at ht
  refine ⟨⟨t,Physical.stores c B ops t before,none,ctx.suppliedOutput,ctx.callerDevice,
    if Needed c then some ctx.ownerDevice else none⟩,?_,rfl,rfl,rfl,rfl,ht,hgood,hmeta,?_,?_,?_,hframe,?_⟩
  · simp only [invoke,ht]
    rfl
  · intro i hi
    exact Physical.readback c B A s hc hm sep ops t before i hi
  · exact Physical.frame c B ops t before
  · exact Physical.input_frame c B hc physicalSep ops t before
  · intro h
    exact Physical.empty_stores c B ops t before (empty_lengths c h)

/-- The source bridge is valid before and after every finite current-memory
prefix, not merely for independent blocks fed the initial memory. -/
theorem source_native (c : Host.Call) (B : Native.Bytes extent) (before : Physical.Memory)
    (abi : Native.ABI c) (hm : Snapshot.Metadata c B before) (sep : FPSuccess.Separate c) :
    Native.GridBridge c abi (backend (NativeSM90.plan c.layout.BM c.layout.D))
      (Snapshot.initial c B before) :=
  Native.grid_bridge c abi (backend (NativeSM90.plan c.layout.BM c.layout.D))
    (dots.support (scalar (NativeSM90.plan c.layout.BM c.layout.D)) c.layout)
    (Snapshot.initial c B before) (Snapshot.metadata c B before hm) sep

#print axioms whole
#print axioms source_native
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Returned
