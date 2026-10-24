import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.CheckedReal
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.FPReferenceGrid
import bench.repairs.tritonbench_g.context_attn_bloom.Whole001.NativeStaging
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Snapshot

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Bytes
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
open Native (Expr)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
abbrev Memory := VeriTile.Bench.Repairs.ContextAttnFwd.Physical.Memory
abbrev Byte := BitVec 8
abbrev Word := BitVec 32
abbrev Windows := VeriTile.Bench.Repairs.ContextAttnFwd.Native.Bytes
abbrev load32 := VeriTile.Bench.Repairs.ContextAttnFwd.Snapshot.load32
abbrev byte := VeriTile.Bench.Repairs.ContextAttnFwd.Physical.byte

def initial (c : Call) (B : Windows extent) (before : Memory) : State Expr where
  mem := fun rg off => if rg∈Checked.metadataRegions c.regions then .mk .int (load32 before (B.base rg+4*off)).toInt
    else .mk .real (.bits (load32 before (B.base rg+4*off)))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := (Host.grid c.args).toNumPids
  undef := fun dtype _ _ => defaultValue dtype

/-- Integer observation of the original four-byte storage, before launch. -/
def layoutState (B : Windows extent) (before : Memory) : BlockState where
  mem := fun rg off => ⟨.int,(load32 before (B.base rg+4*off)).toInt⟩
  regs := fun _ _ _ => none
  pids := fun _ => 0

/-- Only initial metadata is consumed from this interface. The Real payload
fields are immaterial to FP execution and to the discrete bounds. -/
def Observed (c : Call) (B : Windows extent) (before : Memory) : Prop :=
  c.input=RealSetup.inputs (layoutState B before) c.regions

theorem metadata (c : Call) (B : Windows extent) (before : Memory) (h : Observed c B before) :
    FPSuccess.Metadata c (initial c B before) := by
  change c.input=RealSetup.inputs (layoutState B before) c.regions at h
  refine ⟨?_,?_,?_,?_,?_⟩ <;> intro i
  all_goals simp [initial,Checked.metadataRegions,h,RealSetup.inputs,layoutState,BlockState.readMemValue,BlockState.readMemTyped,MemCell.readAs]

def Needed (c : Call) : Prop := ∃ b,b < c.args.batch ∧ 0 < c.input.sequence b-c.input.prompt b
local instance (c : Call) : Decidable (Needed c) := Classical.propDecidable _
def order (c : Call) : List (GridIndex (Host.grid c.args)) := if Needed c then FPSuccess.allPrograms c else []

def reference (c : Call) (B : Windows extent) (before : Memory) : State Expr :=
  FPReference.grid c Native.scalar Native.dots (order c) (initial c B before)

abbrev Lane (c : Call) := GridIndex (Host.grid c.args) × RealIO.Lane (Host.layout c.args) × Fin 4
abbrev active (c : Call) (i : Lane c) := FPSuccess.laneActive c i.1 i.2.1
def address (c : Call) (B : Windows extent) (i : Lane c) :=
  B.base c.regions.out+4*FPSuccess.laneOffset c i.1 i.2.1+i.2.2.val

/-- The compiler plan includes the owner used by the physical final store.
It contains discrete routing/tree choices only, and is used by evaluation. -/
structure Profile where
  plan : Native.Plan
  storedOwner : (a : Host.Arguments) → RealIO.Lane (Host.layout a) → Native.Owner

def outputWord (c : Call) (profile : Profile) (ops : Native.WordISA) (s : State Expr) (i : Lane c) : Word :=
  (Native.lower profile.plan ((s.mem c.regions.out (FPSuccess.laneOffset c i.1 i.2.1)).read .real)
    (profile.storedOwner c.args i.2.1)).eval ops.interpret

def referenceWord (c : Call) (B : Windows extent) (profile : Profile) (ops : Native.WordISA)
    (before : Memory) (i : Lane c) : Word := outputWord c profile ops (reference c B before) i

def stores (c : Call) (B : Windows extent) (profile : Profile) (ops : Native.WordISA)
    (s : State Expr) (before : Memory) (a : Nat) : Byte := by
  classical
  exact if h : ∃ i : Lane c,active c i ∧ address c B i=a then
    byte (outputWord c profile ops s (Classical.choose h)) (Classical.choose h).2.2 else before a

theorem address_unique (c : Call) (B : Windows A.extent) (s : BlockState)
    (same : RealSetup.inputs s c.regions=c.input) (hv : Host.Admissible c.args c.regions A s)
    (i j : Lane c) (hi : active c i) (hj : active c j) (he : address c B i=address c B j) : i=j := by
  have il := i.2.2.isLt
  have jl := j.2.2.isLt
  have offsets : FPSuccess.laneOffset c i.1 i.2.1=FPSuccess.laneOffset c j.1 j.2.1 := by unfold address at he; omega
  obtain ⟨sameProgram,sameLane⟩ := Host.unique_writer c.args c.regions A s hv i.1 j.1 i.2.1 j.2.1
    (by simpa only [active,FPSuccess.laneActive,same] using hi) (by simpa only [active,FPSuccess.laneActive,same] using hj) (by simpa only [FPSuccess.laneOffset,same] using offsets)
  have sameByte : i.2.2=j.2.2 := Fin.ext (by unfold address at he; omega)
  exact Prod.ext sameProgram (Prod.ext sameLane sameByte)

theorem readback (c : Call) (B : Windows A.extent) (layout : BlockState)
    (same : RealSetup.inputs layout c.regions=c.input) (hv : Host.Admissible c.args c.regions A layout)
    (profile : Profile) (ops : Native.WordISA) (s : State Expr) (before : Memory) (i : Lane c) (hi : active c i) :
    stores c B profile ops s before (address c B i)=byte (outputWord c profile ops s i) i.2.2 := by
  have hex : ∃ j : Lane c,active c j ∧ address c B j=address c B i := ⟨i,hi,rfl⟩
  rw [stores,dif_pos hex]
  have witness := Classical.choose_spec hex
  rw [address_unique c B layout same hv (Classical.choose hex) i witness.1 hi witness.2]

theorem frame (c : Call) (B : Windows extent) (profile : Profile) (ops : Native.WordISA)
    (s : State Expr) (before : Memory) (a : Nat) (outside : ¬∃ i : Lane c,active c i ∧ address c B i=a) :
    stores c B profile ops s before a=before a := by rw [stores,dif_neg outside]

def Separate (c : Call) (B : Windows extent) : Prop :=
  ∀ rg∈Checked.readRegions c.regions,B.base c.regions.out+4*extent c.regions.out ≤ B.base rg ∨
    B.base rg+4*extent rg ≤ B.base c.regions.out

theorem input_frame (c : Call) (A : FlatAlloc) (layout : BlockState) (B : Windows A.extent)
    (same : RealSetup.inputs layout c.regions=c.input) (hv : Host.Admissible c.args c.regions A layout)
    (separated : Separate c B) (profile : Profile) (ops : Native.WordISA) (s : State Expr) (before : Memory)
    (rg : RegionName) (read : rg∈Checked.readRegions c.regions) (off : Nat) (bound : off < A.extent rg) (j : Fin 4) :
    stores c B profile ops s before (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val) := by
  apply frame
  rintro ⟨i,hi,he⟩
  have inside := (hv.bounds i.1).output i.2.1.1 i.2.1.2.1
    (by simpa only [same] using hi.1) hi.2
  have within := AddressBounds.inside_toNat inside
  have il := i.2.2.isLt
  have jl := j.isLt
  change (AddressBounds.outputAddress _ _ _ _ _).toNat < A.extent c.regions.out at within
  rw [same] at within
  change FPSuccess.laneOffset c i.1 i.2.1 < A.extent c.regions.out at within
  unfold address at he
  rcases separated rg read with h|h <;> omega

#specNonCircular referenceWord avoiding [FP.Structural.exec,FP.Structural.run,FPSuccess.runGrid,kernel]
#axiomsClean readback
#axiomsClean input_frame
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Bytes
