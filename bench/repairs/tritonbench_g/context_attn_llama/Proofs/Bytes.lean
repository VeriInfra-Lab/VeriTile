import bench.repairs.tritonbench_g.context_attn_llama.Proofs.CheckedReal
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.FPReferenceGrid
import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeInstructions
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Bytes
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
open NativeInstructions (Expr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
abbrev Memory := VeriTile.Bench.Repairs.ContextAttnFwd.Physical.Memory
abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev Windows := VeriTile.Bench.Repairs.ContextAttnFwd.Native.Bytes
abbrev load32 := VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot.load32
abbrev byte := VeriTile.Bench.Repairs.ContextAttnFwd.Physical.byte

def metadataRegions (c : Call) := [c.regions.start,c.regions.sequence,c.regions.prompt,c.regions.request,c.regions.pages]

def initial (c : Call) (B : Windows extent) (before : Memory) : State Expr where
  mem := fun rg off => if rg∈metadataRegions c then .mk .int (load32 before (B.base rg+4*off)).toInt
    else .mk .real (.leaf (load32 before (B.base rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := c.grid.toNumPids
  undef := fun dtype _ _ => defaultValue dtype

def layoutState (B : Windows extent) (before : Memory) : BlockState where
  mem := fun rg off => ⟨.int,(load32 before (B.base rg+4*off)).toInt⟩
  regs := fun _ _ _ => none
  pids := fun _ => 0

structure Observed (c : Call) (B : Windows extent) (before : Memory)
    (start sequence prompt request pages : Nat → Int) : Prop where
  start : ∀ b,b < c.batches → (load32 before (B.base c.regions.start+4*b)).toInt=start b
  sequence : ∀ b,b < c.batches → (load32 before (B.base c.regions.sequence+4*b)).toInt=sequence b
  prompt : ∀ b,b < c.batches → (load32 before (B.base c.regions.prompt+4*b)).toInt=prompt b
  request : ∀ b,b < c.batches → (load32 before (B.base c.regions.request+4*b)).toInt=request b
  pages : ∀ off,(load32 before (B.base c.regions.pages+4*off)).toInt=pages off

def observation (c : Call) (B : Windows extent) (before : Memory)
    (start sequence prompt request pages : Nat → Int) (h : Observed c B before start sequence prompt request pages) :
    Checked.MetadataObservation c (layoutState B before) start sequence prompt request pages where
  startWords := fun b => load32 before (B.base c.regions.start+4*b)
  sequenceWords := fun b => load32 before (B.base c.regions.sequence+4*b)
  promptWords := fun b => load32 before (B.base c.regions.prompt+4*b)
  requestWords := fun b => load32 before (B.base c.regions.request+4*b)
  pageWords := fun b => load32 before (B.base c.regions.pages+4*b)
  startCPU := fun b hb => (h.start b hb).symm
  sequenceCPU := fun b hb => (h.sequence b hb).symm
  promptCPU := fun b hb => (h.prompt b hb).symm
  requestCPU := fun b hb => (h.request b hb).symm
  pagesCPU := fun off => (h.pages off).symm
  startMemory := fun b hb => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.start b hb
  sequenceMemory := fun b hb => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.sequence b hb
  promptMemory := fun b hb => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.prompt b hb
  requestMemory := fun b hb => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.request b hb
  pagesMemory := fun off => by simpa [layoutState,Program.inputs,BlockState.readMemValue] using h.pages off

theorem metadata (c : Call) (B : Windows extent) (before : Memory)
    (start sequence prompt request pages : Nat → Int) (h : Observed c B before start sequence prompt request pages)
    (guard : Checked.metadataChecks c start sequence prompt request pages) : FPSuccess.Metadata c (initial c B before) := by
  refine ⟨?_,?_,?_⟩
  · intro b hb; simpa [initial,metadataRegions] using (h.start b hb).trans (guard.1 ⟨b,hb⟩).1
  · intro b hb; simpa [initial,metadataRegions] using (h.sequence b hb).trans (guard.1 ⟨b,hb⟩).2.1
  · intro b hb; simpa [initial,metadataRegions] using (h.prompt b hb).trans (guard.1 ⟨b,hb⟩).2.2.1

def Needed (c : Call) : Prop := ∃ b,b < c.batches ∧ 0 < c.length b
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _
def order (c : Call) : List (GridIndex c.grid) := if Needed c then FPSuccess.allPrograms c else []
abbrev backend := NativeInstructions.backend
abbrev support := NativeInstructions.support

def reference (c : Call) (B : Windows extent) (before : Memory) : State Expr :=
  FPReference.grid c (backend c) (support c) (order c) (initial c B before)

abbrev Lane (c : Call) := GridIndex c.grid × TileIndex [c.layout.BM,c.layout.D] × Fin 4
def active (c : Call) (i : Lane c) := c.active i.1 i.2.1
def address (c : Call) (B : Windows extent) (i : Lane c) := B.base c.regions.out+4*c.offset i.1 i.2.1+i.2.2.val
def outputWord (c : Call) (ops : NativeInstructions.ScalarOps) (s : State Expr) (i : Lane c) : Word :=
  NativeInstructions.evaluate ops (NativeInstructions.plan c.layout.BM c.layout.D)
    ((s.mem c.regions.out (c.offset i.1 i.2.1)).read .real)
    (NativeInstructions.owner c.layout.BM c.layout.D i.2.1.1.val i.2.1.2.1.val)
def referenceWord (c : Call) (B : Windows extent) (ops : NativeInstructions.ScalarOps) (before : Memory) (i : Lane c) : Word :=
  outputWord c ops (reference c B before) i

def stores (c : Call) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (s : State Expr) (before : Memory) (a : Nat) : Byte := by
  classical
  exact if h : ∃ i : Lane c,active c i ∧ address c B i=a then
    byte (outputWord c ops s (Classical.choose h)) (Classical.choose h).2.2 else before a

theorem address_unique (c : Call) (B : Windows A.extent) (s : BlockState) (hv : Host.Valid c A s)
    (i j : Lane c) (hi : active c i) (hj : active c j) (he : address c B i=address c B j) : i=j := by
  have il := i.2.2.isLt
  have jl := j.2.2.isLt
  have offsets : c.offset i.1 i.2.1=c.offset j.1 j.2.1 := by unfold address at he; omega
  obtain ⟨sameProgram,sameLane⟩ := Host.writer_unique c A s hv i.1 j.1 i.2.1 j.2.1 hi hj offsets
  have sameByte : i.2.2=j.2.2 := Fin.ext (by unfold address at he; omega)
  exact Prod.ext sameProgram (Prod.ext sameLane sameByte)

theorem readback (c : Call) (B : Windows A.extent) (layout : BlockState) (hv : Host.Valid c A layout)
    (ops : NativeInstructions.ScalarOps) (s : State Expr) (before : Memory) (i : Lane c) (hi : active c i) :
    stores c B ops s before (address c B i)=byte (outputWord c ops s i) i.2.2 := by
  have hex : ∃ j : Lane c,active c j ∧ address c B j=address c B i := ⟨i,hi,rfl⟩
  rw [stores,dif_pos hex]
  have witness := Classical.choose_spec hex
  rw [address_unique c B layout hv (Classical.choose hex) i witness.1 hi witness.2]

theorem frame (c : Call) (B : Windows extent) (ops : NativeInstructions.ScalarOps)
    (s : State Expr) (before : Memory) (a : Nat) (outside : ¬∃ i : Lane c,active c i ∧ address c B i=a) :
    stores c B ops s before a=before a := by rw [stores,dif_neg outside]

def Separate (c : Call) (B : Windows extent) : Prop :=
  ∀ rg∈c.inputs,B.base c.regions.out+4*extent c.regions.out≤B.base rg ∨ B.base rg+4*extent rg≤B.base c.regions.out

theorem input_frame (c : Call) (A : FlatAlloc) (layout : BlockState) (B : Windows A.extent)
    (hv : Host.Valid c A layout) (separated : Separate c B)
    (ops : NativeInstructions.ScalarOps) (s : State Expr) (before : Memory)
    (rg : RegionName) (read : rg∈c.inputs) (off : Nat) (bound : off < A.extent rg) (j : Fin 4) :
    stores c B ops s before (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val) := by
  apply frame
  rintro ⟨i,hi,he⟩
  have within := hv.out _ (Host.batch_bound c i.1) _ (Host.head_bound c i.1) _ hi _ i.2.1.2.1.isLt
  have il := i.2.2.isLt
  have jl := j.isLt
  change c.offset i.1 i.2.1 < A.extent c.regions.out at within
  unfold address at he
  rcases separated rg read with h|h <;> omega

#specNonCircular referenceWord avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,kernel]
#axiomsClean readback
#axiomsClean input_frame
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Bytes
