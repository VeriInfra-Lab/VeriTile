import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorComplete
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensorBacking
open VeriTile Triton FP.Structural
open KernelPacket (Call)
open KernelTensor (Role View address Written)
noncomputable section

/-- Full backing separation already required by the selected kernel guards. -/
def Apart (a b : Torch.FP32.Tensor) : Prop :=
  a.storage.base+4*a.storage.elements ≤ b.storage.base ∨
    b.storage.base+4*b.storage.elements ≤ a.storage.base

def OutputUsed (a : Call) (role : Role) : Prop := match a.stage,role with
  | .forwardState,.history | .forwardOutput,.output | .backwardState,.dhistory=>True
  | .forwardState,.finalState | .backwardState,.initialGradient=>a.outputState=Bool.true
  | .backwardGradient,.queryGradient | .backwardGradient,.keyGradient | .backwardGradient,.valuePartials=>True
  | _,_=>False
abbrev OutputIndex (a : Call) := {role : Role // OutputUsed a role}

/-- Only outputs enabled by this standalone kernel are required. Distinct
outputs have separate backing, so their observations share one post-memory. -/
structure Outputs (a : Call) where
  view : (r : OutputIndex a)→View a.shape r.val
  separate : ∀ r s,r≠s → Apart (view r).tensor (view s).tensor

inductive InputRole where
  | query | key | value | history | seed | incoming | dhistory | terminal
  deriving DecidableEq
def InputRole.region : InputRole→RegionName
  | .query=>"q" | .key=>"k" | .value=>"v" | .history=>"h"
  | .seed=>"h0" | .incoming=>"incoming" | .dhistory=>"dhistory" | .terminal=>"terminal"
def InputRole.shapeRole : InputRole→Role
  | .query | .key=>.queryGradient
  | .value | .incoming=>.output
  | .history | .dhistory=>.history
  | .seed | .terminal=>.finalState
def InputUsed (a : Call) (role : InputRole) : Prop := match a.stage,role with
  | .forwardState,.key | .forwardState,.value=>True
  | .forwardState,.seed=>a.inputState=Bool.true
  | .forwardOutput,.query | .forwardOutput,.key | .forwardOutput,.value | .forwardOutput,.history=>True
  | .backwardState,.query | .backwardState,.incoming=>True
  | .backwardState,.terminal=>a.inputState=Bool.true
  | .backwardGradient,.query | .backwardGradient,.key | .backwardGradient,.value
  | .backwardGradient,.history | .backwardGradient,.incoming | .backwardGradient,.dhistory=>True
  | _,_=>False
abbrev InputIndex (a : Call) := {role : InputRole // InputUsed a role}

structure Inputs (a : Call) where
  view : (r : InputIndex a)→View a.shape r.val.shapeRole

/-- Genuine input and output Tensor descriptors. All inputs read one initial
backing; output/input backing separation is structural, never numerical. -/
structure Buffers (a : Call) where
  inputs : Inputs a
  outputs : Outputs a
  separate : ∀ i o,Apart (inputs.view i).tensor (outputs.view o).tensor

theorem input_region_injective : Function.Injective InputRole.region := by
  intro r s h
  cases r <;> cases s <;> simp_all [InputRole.region]

/-- Define the interpreter's input words directly from the actual input data
pointers in one caller backing. Disabled/unused regions receive a fixed filler;
no active input or desired output value is assumed. -/
def snapshot {β : Type} (zero : β) (read : Nat→β) (a : Call) (f : Inputs a)
    (region : RegionName) (off : Nat) : β := by
  classical
  exact if h : ∃ i : InputIndex a,i.val.region=region then
    read (address (f.view (Classical.choose h)) off) else zero

theorem snapshot_input {β : Type} (zero : β) (read : Nat→β) (a : Call) (f : Inputs a)
    (i : InputIndex a) (off : Nat) :
    snapshot zero read a f i.val.region off=read (address (f.view i) off) := by
  classical
  have h : ∃ j : InputIndex a,j.val.region=i.val.region:=⟨i,rfl⟩
  have he : Classical.choose h=i:=Subtype.ext (input_region_injective (Classical.choose_spec h))
  simp only [snapshot,dif_pos h]
  exact congrArg (fun j : InputIndex a=>read (address (f.view j) off)) he

theorem snapshot_logical {β : Type} (zero : β) (read : Nat→β) (a : Call) (f : Inputs a)
    (i : InputIndex a) (off : Nat) (bound : off<(f.view i).tensor.metadata.numel) :
    snapshot zero read a f i.val.region off=KernelTensor.logical read (f.view i) off := by
  rw [snapshot_input,KernelTensor.logical,KernelTensor.coordinates _ _ bound]

theorem snapshot_native (read : Nat→BitVec 32) (a : Call) (f : Inputs a)
    (i : InputIndex a) (off : Nat) (bound : off<(f.view i).tensor.metadata.numel) :
    snapshot 0 read a f i.val.region off=
      (Torch.FP32.logicalValue (fun addr=>⟨read addr⟩) (f.view i).tensor (f.view i).laws off).bits := by
  rw [KernelTensor.native_logical]
  exact snapshot_logical 0 read a f i off bound

theorem written_inside (a : Call) (role : Role) (v : View a.shape role)
    (addr : Nat) (h : Written a role v addr) :
    v.tensor.storage.base≤addr ∧ addr<v.tensor.storage.base+4*v.tensor.storage.elements := by
  let off:=(addr-v.tensor.metadata.dataPtr.toNat)/4
  have bound : off<v.tensor.metadata.numel:=h.2.2.1
  have he : addr=address v off := by
    have hd:=Nat.div_add_mod (addr-v.tensor.metadata.dataPtr.toNat) 4
    dsimp [address,off]
    have hlo:=h.1
    have hmod:=h.2.1
    omega
  have hc:=KernelTensor.coordinates v off bound
  have hb:=v.laws.inBounds off bound
  rw [←hc] at he
  unfold Torch.FP32.Storage.address at he
  omega

def SharedWritten (a : Call) (f : Outputs a) (addr : Nat) : Prop :=
  ∃ r : OutputIndex a,Written a r.val (f.view r) addr

theorem written_unique (a : Call) (f : Outputs a) (r s : OutputIndex a) (addr : Nat)
    (hr : Written a r.val (f.view r) addr) (hs : Written a s.val (f.view s) addr) : r=s := by
  by_contra ne
  have sep:=f.separate r s ne
  have rbound:=written_inside a r.val (f.view r) addr hr
  have sbound:=written_inside a s.val (f.view s) addr hs
  unfold Apart at sep
  omega

/-- One shared physical post-memory for every output of the executed program. -/
def post {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Outputs a) (t : State α) (addr : Nat) : β := by
  classical
  exact if h : SharedWritten a f addr then
    let r:=Classical.choose h
    decode ((t.mem r.val.region ((addr-(f.view r).tensor.metadata.dataPtr.toNat)/4)).read .real)
  else read addr

theorem post_exterior {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Outputs a) (t : State α) (addr : Nat) (h : ¬SharedWritten a f addr) :
    post decode read a f t addr=read addr := by simp [post,h]

theorem post_written {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Outputs a) (t : State α) (r : OutputIndex a) (addr : Nat)
    (hr : Written a r.val (f.view r) addr) :
    post decode read a f t addr=
      decode ((t.mem r.val.region ((addr-(f.view r).tensor.metadata.dataPtr.toNat)/4)).read .real) := by
  classical
  have h : SharedWritten a f addr:=⟨r,hr⟩
  have he : Classical.choose h=r:=written_unique a f _ r addr (Classical.choose_spec h) hr
  simp only [post,dif_pos h]
  exact congrArg (fun j : OutputIndex a=>decode
    ((t.mem j.val.region ((addr-(f.view j).tensor.metadata.dataPtr.toNat)/4)).read .real)) he

theorem post_logical {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Outputs a) (t : State α) (r : OutputIndex a) (off : Nat)
    (bound : off<(f.view r).tensor.metadata.numel) (written : KernelPacket.Footprint a r.val.region off) :
    KernelTensor.logical (post decode read a f t) (f.view r) off=
      decode ((t.mem r.val.region off).read .real) := by
  rw [KernelTensor.logical,KernelTensor.coordinates _ _ bound]
  have hw : Written a r.val (f.view r) (address (f.view r) off) := by
    simp [Written,address,bound,written]
  rw [post_written decode read a f t r _ hw]
  simp [address]

theorem post_input_frame {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Buffers a) (t : State α) (i : InputIndex a) (addr : Nat)
    (inside : (f.inputs.view i).tensor.storage.base≤addr ∧
      addr<(f.inputs.view i).tensor.storage.base+4*(f.inputs.view i).tensor.storage.elements) :
    post decode read a f.outputs t addr=read addr := by
  apply post_exterior
  rintro ⟨r,hr⟩
  have hb:=written_inside a r.val (f.outputs.view r) addr hr
  have sep:=f.separate i r
  unfold Apart at sep
  omega

theorem post_input_logical {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Buffers a) (t : State α) (i : InputIndex a) (off : Nat)
    (bound : off<(f.inputs.view i).tensor.metadata.numel) :
    KernelTensor.logical (post decode read a f.outputs t) (f.inputs.view i) off=
      KernelTensor.logical read (f.inputs.view i) off := by
  unfold KernelTensor.logical
  apply post_input_frame decode read a f t i
  have h:=(f.inputs.view i).laws.inBounds off bound
  unfold Torch.FP32.Storage.address
  omega

/-- Both physical frames concern the single post-memory used by every output. -/
def Frame {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Buffers a) (t : State α) : Prop :=
  (∀ addr,¬SharedWritten a f.outputs addr → post decode read a f.outputs t addr=read addr) ∧
  (∀ i addr,(f.inputs.view i).tensor.storage.base≤addr ∧
      addr<(f.inputs.view i).tensor.storage.base+4*(f.inputs.view i).tensor.storage.elements →
    post decode read a f.outputs t addr=read addr) ∧
  (∀ i off,off<(f.inputs.view i).tensor.metadata.numel →
    KernelTensor.logical (post decode read a f.outputs t) (f.inputs.view i) off=
      KernelTensor.logical read (f.inputs.view i) off)

theorem frame {α β : Type} [Inhabited α] (decode : α→β) (read : Nat→β)
    (a : Call) (f : Buffers a) (t : State α) : Frame decode read a f t :=
  ⟨post_exterior decode read a f.outputs t,post_input_frame decode read a f t,
    post_input_logical decode read a f t⟩

def realInput (read : Nat→ℝ) (a : Call) (f : Inputs a) : KernelPacket.Memory ℝ :=
  fun r off=>.mk .real (snapshot 0 read a f r off)

def realObserved (read : Nat→ℝ) (a : Call) (f : Buffers a) (t : State ℝ)
    (r : OutputIndex a) (off : Nat) : ℝ :=
  KernelTensor.logical (post id read a f.outputs t) (f.outputs.view r) off

theorem real_readback (read : Nat→ℝ) (a : Call) (f : Buffers a) (t : State ℝ)
    (hp : KernelTensor.Valid a) (r : OutputIndex a) (off : Nat)
    (hw : KernelPacket.Footprint a r.val.region off) (v : ℝ)
    (stored : t.mem r.val.region off=.mk .real v) : realObserved read a f t r off=v := by
  unfold realObserved
  rw [post_logical id read a f.outputs t r off
    (KernelTensor.footprint_bound a hp r.val (f.outputs.view r) off hw) hw,stored]
  rfl

def nativeDecode (isa : ISA) (literal : ℝ→BitVec 32) : KernelNative.Term→BitVec 32 :=
  KernelNative.S.lower (KernelNative.sharedOps (wordOps isa)) literal
def nativeObserved (isa : ISA) (literal : ℝ→BitVec 32) (read : Nat→BitVec 32)
    (a : Call) (f : Buffers a) (t : State KernelNative.Term) (r : OutputIndex a) (off : Nat) : BitVec 32 :=
  (Torch.FP32.logicalValue (fun addr=>⟨post (nativeDecode isa literal) read a f.outputs t addr⟩)
    (f.outputs.view r).tensor (f.outputs.view r).laws off).bits

theorem native_readback (isa : ISA) (literal : ℝ→BitVec 32) (read : Nat→BitVec 32)
    (a : Call) (f : Buffers a) (t : State KernelNative.Term) (hp : KernelTensor.Valid a)
    (r : OutputIndex a) (off : Nat) (hw : KernelPacket.Footprint a r.val.region off) (v : BitVec 32)
    (stored : KernelNative.observe isa literal t.mem r.val.region off=v) :
    nativeObserved isa literal read a f t r off=v := by
  rw [nativeObserved,KernelTensor.native_logical]
  exact (post_logical (nativeDecode isa literal) read a f.outputs t r off
    (KernelTensor.footprint_bound a hp r.val (f.outputs.view r) off hw) hw).trans stored

#axiomsClean snapshot_native
#axiomsClean written_unique
#axiomsClean post_logical
#axiomsClean post_input_frame
#axiomsClean frame
#axiomsClean real_readback
#axiomsClean native_readback
#specNonCircular Buffers avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular snapshot avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelTensorBacking
