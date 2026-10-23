import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.CheckedReal
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.CheckedNative
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPReferenceGrid
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeInstructions
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Bytes
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open NativePhysical (Expr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
abbrev Memory := VeriTile.Bench.Repairs.ContextAttnFwd.Physical.Memory
abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev Windows := VeriTile.Bench.Repairs.ContextAttnFwd.Native.Bytes
abbrev load32 := VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot.load32
abbrev byte := VeriTile.Bench.Repairs.ContextAttnFwd.Physical.byte

def metadataRegions (r : Regions) := [r.start,r.sequence]
def readRegions (r : Regions) := [r.query,r.key,r.value,r.start,r.sequence]

def initial (c : Call) (r : Regions) (B : Windows extent) (before : Memory) : State Expr where
  mem := fun rg off => if rg∈metadataRegions r then .mk .int (load32 before (B.base rg+4*off)).toInt
    else .mk .real (.leaf .fp32 (load32 before (B.base rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := c.grid.toNumPids
  undef := fun d _ _ => defaultValue d

def layoutState (B : Windows extent) (before : Memory) : BlockState where
  mem := fun rg off => ⟨.int,(load32 before (B.base rg+4*off)).toInt⟩
  regs := fun _ _ _ => none
  pids := fun _ => 0

structure Observed (c : Call) (r : Regions) (B : Windows extent) (before : Memory)
    (start length : Nat → Int) : Prop where
  start : ∀ b,b < c.batches → (load32 before (B.base r.start+4*b)).toInt=start b
  length : ∀ b,b < c.batches → (load32 before (B.base r.sequence+4*b)).toInt=length b

def observation (c : Call) (r : Regions) (B : Windows extent) (before : Memory)
    (start length : Nat → Int) (h : Observed c r B before start length) :
    Checked.MetadataObservation c r (layoutState B before) start length where
  startWords := fun b => load32 before (B.base r.start+4*b)
  lengthWords := fun b => load32 before (B.base r.sequence+4*b)
  startCPU := fun b hb => (h.start b hb).symm
  lengthCPU := fun b hb => (h.length b hb).symm
  startMemory := fun b hb => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.start b hb
  lengthMemory := fun b hb => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.length b hb

theorem metadata (c : Call) (r : Regions) (B : Windows extent) (before : Memory)
    (start length : Nat → Int) (h : Observed c r B before start length)
    (guard : Checked.metadataChecks c start length) : FPHostSource.Metadata c r (initial c r B before) := by
  constructor
  · intro b hb
    simpa [initial,metadataRegions] using (h.start b hb).trans (guard.1 ⟨b,hb⟩).1
  · intro b hb
    simpa [initial,metadataRegions] using (h.length b hb).trans (guard.1 ⟨b,hb⟩).2

def Needed (c : Call) : Prop := ∃ b,b < c.batches ∧ 0 < c.length b
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _

def order (c : Call) : List (GridIndex c.grid) := if Needed c then FPHost.jobs c else []
def backend (c : Call) := NativePhysical.backend (NativeProfiles.plan c.q.dims)

def reference (c : Call) (r : Regions) (B : Windows extent) (before : Memory) : State Expr :=
  FPReference.grid c r (backend c) (order c) (initial c r B before)

abbrev Lane (c : Call) := GridIndex c.grid × TileIndex [c.layout.BM,c.layout.D] × Fin 4

def active (c : Call) (i : Lane c) := FPReference.laneActive c i.1 i.2.1

def address (c : Call) (r : Regions) (B : Windows extent) (i : Lane c) :=
  B.base r.out+4*FPReference.laneOffset c i.1 i.2.1+i.2.2.val

def outputWord (c : Call) (r : Regions) (ops : NativeInstructions.ScalarOps) (s : State Expr) (i : Lane c) : Word :=
  let expression := (s.mem r.out (FPReference.laneOffset c i.1 i.2.1)).read .real
  NativeInstructions.evaluate ops (NativeProfiles.plan c.q.dims) expression (NativeProfiles.ownerOf c.q.dims expression)

def referenceWord (c : Call) (r : Regions) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (before : Memory) (i : Lane c) : Word := outputWord c r ops (reference c r B before) i

def stores (c : Call) (r : Regions) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (s : State Expr) (before : Memory) (a : Nat) : Byte := by
  classical
  exact if h : ∃ i : Lane c,active c i ∧ address c r B i=a then byte (outputWord c r ops s (Classical.choose h)) (Classical.choose h).2.2
  else before a

theorem address_unique (c : Call) (r : Regions) (B : Windows A.extent) (s : BlockState)
    (hm : Host.Memory c r A s) (i j : Lane c) (hi : active c i) (hj : active c j)
    (he : address c r B i=address c r B j) : i=j := by
  have il := i.2.2.isLt
  have jl := j.2.2.isLt
  have offsets : FPReference.laneOffset c i.1 i.2.1=FPReference.laneOffset c j.1 j.2.1 := by
    unfold address at he; omega
  have ia : RealIO.active c.layout (c.launch i.1) (Program.inputs s r) i.2.1 :=
    (Host.active_iff hm i.1 i.2.1).mpr hi
  have ja : RealIO.active c.layout (c.launch j.1) (Program.inputs s r) j.2.1 :=
    (Host.active_iff hm j.1 j.2.1).mpr hj
  have equal : RealIO.write c.layout (c.launch i.1) (Program.inputs s r) i.2.1=
      RealIO.write c.layout (c.launch j.1) (Program.inputs s r) j.2.1 := by
    rw [Host.write_eq hm,Host.write_eq hm]
    exact offsets
  obtain ⟨sameProgram,sameLane⟩ := Host.unique_writer hm i.1 j.1 i.2.1 j.2.1 ia ja equal
  have sameByte : i.2.2=j.2.2 := Fin.ext (by unfold address at he; omega)
  exact Prod.ext sameProgram (Prod.ext sameLane sameByte)

theorem readback (c : Call) (r : Regions) (B : Windows A.extent) (layout : BlockState)
    (hm : Host.Memory c r A layout) (ops : NativeInstructions.ScalarOps) (s : State Expr)
    (before : Memory) (i : Lane c) (hi : active c i) :
    stores c r B ops s before (address c r B i)=byte (outputWord c r ops s i) i.2.2 := by
  have hex : ∃ j : Lane c,active c j ∧ address c r B j=address c r B i := ⟨i,hi,rfl⟩
  rw [stores,dif_pos hex]
  have witness := Classical.choose_spec hex
  rw [address_unique c r B layout hm (Classical.choose hex) i witness.1 hi witness.2]

theorem frame (c : Call) (r : Regions) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (s : State Expr) (before : Memory) (a : Nat)
    (outside : ¬∃ i : Lane c,active c i ∧ address c r B i=a) : stores c r B ops s before a=before a := by
  rw [stores,dif_neg outside]

def Separate (c : Call) (r : Regions) (B : Windows extent) : Prop :=
  ∀ rg∈readRegions r,B.base r.out+4*extent r.out≤B.base rg ∨ B.base rg+4*extent rg≤B.base r.out

theorem active_bound (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState) (hm : Host.Memory c r A s)
    (i : Lane c) (hi : active c i) : FPReference.laneOffset c i.1 i.2.1 < A.extent r.out := by
  have on : RealIO.active c.layout (c.launch i.1) (Program.inputs s r) i.2.1 :=
    (Host.active_iff hm i.1 i.2.1).mpr hi
  exact hm.oBounded _ _ _ (Host.logical_row_lt hm i.1 i.2.1 on)
    (lt_of_lt_of_le (launch_head_lt c i.1) hm.shape.oHeads) (lt_of_lt_of_le i.2.1.2.1.isLt hm.shape.oDims)

theorem input_frame (c : Call) (r : Regions) (A : FlatAlloc) (layout : BlockState)
    (B : Windows A.extent) (hm : Host.Memory c r A layout) (separated : Separate c r B)
    (ops : NativeInstructions.ScalarOps) (s : State Expr) (before : Memory)
    (rg : RegionName) (read : rg∈readRegions r) (off : Nat) (bound : off < A.extent rg) (j : Fin 4) :
    stores c r B ops s before (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val) := by
  apply frame
  rintro ⟨i,hi,he⟩
  have within := active_bound c r A layout hm i hi
  have il := i.2.2.isLt
  have jl := j.isLt
  unfold address at he
  rcases separated rg read with h|h <;> omega

#specNonCircular referenceWord avoiding [FP.Structural.exec,FP.Structural.run,FPReference.runGrid,kernel]
#axiomsClean readback
#axiomsClean input_frame
end
end VeriTile.Bench.Repairs.ContextAttnNopad.Bytes
