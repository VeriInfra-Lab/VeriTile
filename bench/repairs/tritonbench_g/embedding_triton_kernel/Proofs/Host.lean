import bench.repairs.tritonbench_g.embedding_triton_kernel.Proofs.Native

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.Host
open VeriTile.Triton FP.Structural Physical
noncomputable section

def parameters (c : Request) : Parameters :=
  ⟨"weight","ids","out",c.firstId,c.lastId,c.hidden,c.hidden,c.n,c.hidden,c.hidden.nextPowerOfTwo⟩

def initial {α : Type} [Inhabited α] (zero : α) (interpret : Word → α)
    (c : Request) (m : Memory) : State α where
  mem := fun region i=>if region="ids" then .mk .int (idAt c m i)
    else if region="weight" then .mk .real (interpret (m (address c.weight i)))
    else if region="out" then .mk .real (interpret (m (address c.output i)))
    else .mk .real zero
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun ax=>if ax=0 then grid c.n else 1
  undef := fun d _ _=>defaultValue d

def layout (c : Request) : Execution.Layout (parameters c) :=
  ⟨by change ("ids" : RegionName) ≠ "out"; decide,
    by change ("weight" : RegionName) ≠ "out"; decide,
    Bounds.nextPower_covers c.hidden,by rfl⟩

/-- Independent formula for any payload interpretation. -/
def reference {α : Type} (zero : α) (interpret : Word → α) (c : Request) (m : Memory)
    (r d : Nat) : α :=
  target zero c.first c.last (idAt c m)
    (fun i j=>interpret (m (address c.weight (i*c.hidden+j)))) r d

theorem expected_matches {α : Type} [Inhabited α] (zero : α) (interpret : Word → α)
    (c : Request) (m : Memory) (r d : Nat) :
    Grid.expected zero (parameters c) (initial zero interpret c m) r d=
      reference zero interpret c m r d := by
  simp only [Grid.expected,parameters,initial,Region.mk.injEq,String.reduceEq,↓reduceIte,Cell.read_same,
    reference,target]
  obtain ⟨lo,hi⟩ := id_bounds c m r
  simp only [Bounds.interval c _ lo hi]
  split_ifs with valid
  · rw [Bounds.first_eq c _ lo hi valid]
  · rfl

def typedRun {α : Type} [Inhabited α] (rest : Algebra α) (zero : α) (interpret : Word → α)
    (c : Request) (m : Memory) : Option (State α) :=
  if c.n=0 ∨ c.hidden=0 then some (initial zero interpret c m)
  else Grid.launch rest zero .int32 (parameters c) (initial zero interpret c m)

/-- Host empty-return branches and the full source grid share one theorem. -/
theorem typed_correct {α : Type} [Inhabited α] (rest : Algebra α) (zero : α)
    (interpret : Word → α) (c : Request) (m : Memory) :
    ∃ t,typedRun rest zero interpret c m=some t ∧
      (∀ r,r<c.n → ∀ d,d<c.hidden →
        t.mem "out" (r*c.hidden+d)=.mk .real (reference zero interpret c m r d)) ∧
      (∀ (region : RegionName) o,(region≠"out" ∨ ∀ r,r<c.n → ∀ d,d<c.hidden → o≠r*c.hidden+d) →
        t.mem region o=(initial zero interpret c m).mem region o) := by
  by_cases empty : c.n=0 ∨ c.hidden=0
  · refine ⟨initial zero interpret c m,by simp [typedRun,empty],?_,by intros; rfl⟩
    intro r hr d hd
    rcases empty with hn|hh <;> omega
  · obtain ⟨t,run,values,frame⟩ := Grid.launch_correct rest zero .int32 (parameters c)
      (layout c) (initial zero interpret c m)
    refine ⟨t,by simp [typedRun,empty,run],?_,frame⟩
    intro r hr d hd
    have value := values r hr d hd
    rw [expected_matches] at value
    exact value

structure Returned where
  value : Option Unit
  outputObject : Nat
  outputMetadata : VeriTile.Torch.TensorMetadata
  memory : Memory
  restoredDevice : VeriTile.Torch.Device

def run (c : Request) (m : Memory) (outputObject : Nat) (incoming : VeriTile.Torch.Device) : Returned :=
  ⟨none,outputObject,c.output,
    if c.n=0 ∨ c.hidden=0 then m else Native.launch c m,incoming⟩

/-- The ordinary successful CUDA device context chooses c.output.device and
restores incoming. There are no allocations, transfers, snapshots or copyback. -/
theorem returned_correct (c : Request) (h : InputContract c) (storage : Native.TensorStorage c)
    (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device) :
    (run c m object incoming).value=none ∧
    (run c m object incoming).outputObject=object ∧
    (run c m object incoming).outputMetadata=c.output ∧
    (run c m object incoming).restoredDevice=incoming ∧
    (∀ r,r<c.n → ∀ d,d<c.hidden →
      (run c m object incoming).memory (address c.output (r*c.hidden+d))=desired c m r d) ∧
    (∀ byte,Outside c byte → (run c m object incoming).memory byte=m byte) := by
  refine ⟨rfl,rfl,rfl,rfl,?_,?_⟩
  · intro r hr d hd
    have nonempty : ¬(c.n=0 ∨ c.hidden=0) := by omega
    simp only [run,if_neg nonempty,Native.launch_eq c h storage]
    exact (Physical.launch_correct c h m).1 r hr d hd
  · intro byte miss
    dsimp only [run]
    split_ifs
    · rfl
    · rw [Native.launch_eq c h storage]
      exact (Physical.launch_correct c h m).2 byte miss

theorem interpret_desired {α : Type} (zero : α) (interpret : Word → α)
    (hz : interpret 0=zero) (c : Request) (m : Memory) (r d : Nat) :
    interpret (desired c m r d)=reference zero interpret c m r d := by
  unfold desired reference target
  split_ifs <;> simp_all

/-- The exact returned word and the actual typed source execution agree at
all output coordinates, without any whole-result or execution premise. -/
theorem whole {α : Type} [Inhabited α] (rest : Algebra α) (zero : α)
    (interpret : Word → α) (hz : interpret 0=zero)
    (c : Request) (h : InputContract c) (storage : Native.TensorStorage c)
    (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device) :
    ∃ t,typedRun rest zero interpret c m=some t ∧
      (∀ r,r<c.n → ∀ d,d<c.hidden →
        t.mem "out" (r*c.hidden+d)=.mk .real (reference zero interpret c m r d) ∧
        interpret ((run c m object incoming).memory (address c.output (r*c.hidden+d)))=
          reference zero interpret c m r d) ∧
      (run c m object incoming).value=none ∧
      (run c m object incoming).outputObject=object ∧
      (run c m object incoming).outputMetadata=c.output ∧
      (run c m object incoming).restoredDevice=incoming ∧
      (∀ byte,Outside c byte → (run c m object incoming).memory byte=m byte) := by
  obtain ⟨t,exec,values,_⟩ := typed_correct rest zero interpret c m
  obtain ⟨returned,identity,metadata,context,raw,frame⟩ := returned_correct c h storage m object incoming
  refine ⟨t,exec,?_,returned,identity,metadata,context,frame⟩
  intro r hr d hd
  exact ⟨values r hr d hd,by rw [raw r hr d hd,interpret_desired zero interpret hz]⟩

/-- Byte observation of the aligned 32-bit memory. This exposes the caller
frame even at bytes which are not themselves word-start addresses. -/
def readByte (m : Memory) (byte : Nat) : BitVec 8 :=
  (m (byte/4*4)).extractLsb' (8*(byte%4)) 8

def OutsideByte (c : Request) (byte : Nat) : Prop :=
  byte<c.output.dataPtr.toNat ∨ c.output.dataPtr.toNat+4*c.output.numel≤byte

theorem returned_byte_frame (c : Request) (h : InputContract c) (storage : Native.TensorStorage c)
    (m : Memory) (object : Nat) (incoming : VeriTile.Torch.Device)
    (byte : Nat) (outside : OutsideByte c byte) :
    readByte (run c m object incoming).memory byte=readByte m byte := by
  have miss : Outside c (byte/4*4) := by
    intro r hr d hd eq
    have bound : r*c.hidden+d<c.output.numel := by
      rw [(Bounds.sizes c h).2.2]
      exact Bounds.flat_bound _ _ _ _ hr hd
    have hm := Nat.mod_lt byte (by decide : 0<4)
    have he := Nat.mod_add_div byte 4
    unfold address at eq
    unfold OutsideByte at outside
    rcases outside with outside|outside <;> omega
  have frame := (returned_correct c h storage m object incoming).2.2.2.2.2 _ miss
  simp only [readByte,frame]

#axiomsClean typed_correct
#axiomsClean returned_correct
#axiomsClean whole
#axiomsClean returned_byte_frame
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.Host
