import bench.tritonbench_g.triton_mul2.Proofs.FiniteScalar

namespace TritonMul2Finite
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
noncomputable section
open Classical
abbrev Codec := VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory.Memory
abbrev ByteMemory := Nat → Option (BitVec 8)

/-- The annotated n_elements is always passed as i32, including large host
numel values. Both index operations wrap before signed comparison. -/
def offset32 (block pid lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 block + BitVec.ofNat 32 lane
def active32 (n block pid lane : Nat) : Bool :=
  decide ((offset32 block pid lane).toInt < (BitVec.ofNat 32 n).toInt)
def pointer64 (base : BitVec 64) (width block pid lane : Nat) : BitVec 64 :=
  base + BitVec.ofInt 64 (offset32 block pid lane).toInt * BitVec.ofNat 64 width
def grid (n block : Nat) : Nat := (n+block-1)/block

structure ByteRegion where
  base : Nat
  capacity : Nat
  readable : Bool
  writable : Bool
  device : Option Nat
  deriving DecidableEq, Repr
def ByteRegion.Contains (r : ByteRegion) (a : Nat) : Prop := r.base≤a ∧ a<r.base+r.capacity
def InRegion (r : ByteRegion) (a width : Nat) : Prop :=
  r.base≤a ∧ a+width≤r.base+r.capacity ∧ a+width≤2^64
/-- Tensor ownership remains metadata. Accessibility and address translation
are independently observed in the actual launch context, byte by byte. -/
structure AccessContext where
  currentDevice : Nat
  readableByte : Nat → Nat → Prop
  writableByte : Nat → Nat → Prop
  backingByte : Nat → Nat → Option Nat
def AccessContext.Readable (ctx : AccessContext) (address width : Nat) : Prop :=
  ∀ k, k<width → ctx.readableByte ctx.currentDevice (address+k)
def AccessContext.Writable (ctx : AccessContext) (address width : Nat) : Prop :=
  ∀ k, k<width → ctx.writableByte ctx.currentDevice (address+k)
def AccessContext.Resolves (ctx : AccessContext) (actual backing width : Nat) : Prop :=
  ∀ k, k<width → ctx.backingByte ctx.currentDevice (actual+k)=some (backing+k)
def deviceMemory (ctx : AccessContext) (memory : ByteMemory) : ByteMemory :=
  fun address => (ctx.backingByte ctx.currentDevice address).bind memory

structure Layout where
  input : ByteRegion
  output : ByteRegion
  inputPointer : BitVec 64
  outputPointer : BitVec 64
  inputDeviceBase : Nat
  outputDeviceBase : Nat
  access : AccessContext
def Layout.deviceInput (p : Layout) : ByteRegion := {p.input with base:=p.inputDeviceBase}
def Layout.deviceOutput (p : Layout) : ByteRegion := {p.output with base:=p.outputDeviceBase}
def Layout.inputBacking (p : Layout) (address : Nat) : Nat := p.input.base+(address-p.inputDeviceBase)
def Layout.outputBacking (p : Layout) (address : Nat) : Nat := p.output.base+(address-p.outputDeviceBase)
def InBacking (r : ByteRegion) (address width : Nat) : Prop :=
  r.base≤address ∧ address+width≤r.base+r.capacity
theorem translated_backing_bounds (r : ByteRegion) (deviceBase address width : Nat)
    (inside : InRegion {r with base:=deviceBase} address width) :
    InBacking r (r.base+(address-deviceBase)) width := by
  unfold InRegion at inside
  unfold InBacking
  dsimp at inside
  omega

/-- Missing bytes remain an observation, never a chosen input value. -/
def readWord (d : Storage) (m : ByteMemory) (address : Nat) : Option (Word d) :=
  if ∀ k, k<d.bytes → (m (address+k)).isSome then
    some (VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory.loadLE d.bytes
      (fun a => (m a).getD 0) address)
  else none

structure ByteWrite where
  pid : Nat
  lane : Nat
  inputAddress : Nat
  outputAddress : Nat
  deviceAddress : Nat
  value : BitVec 8
  deriving DecidableEq, Repr
def byteWrite (m : ByteMemory) (address : Nat) (value : BitVec 8) : ByteMemory :=
  fun a => if a=address then some value else m a
def applyWrites : List ByteWrite → ByteMemory → ByteMemory
  | [],m => m
  | w::rest,m => applyWrites rest (byteWrite m w.outputAddress w.value)

/-- Independent pointwise last-write lookup, with no interpreter invocation. -/
def byteResult (writes : List ByteWrite) (initial : ByteMemory) (address : Nat) : Option (BitVec 8) :=
  match writes.reverse.find? (fun w => decide (w.outputAddress=address)) with
  | none => initial address
  | some w => some w.value

def wordWrites (d : Storage) (pid lane input output actualOutput : Nat) (word : Word d) : List ByteWrite :=
  (List.range d.bytes).map fun k => ⟨pid,lane,input,output+k,actualOutput+k,
    VeriTile.Bench.TritonBenchG.ReluTritonKernel.ByteMemory.splitByte word k⟩

theorem writes_append (left right : List ByteWrite) (m : ByteMemory) :
    applyWrites (left++right) m=applyWrites right (applyWrites left m) := by
  induction left generalizing m with
  | nil => rfl
  | cons w rest ih => exact ih _
theorem writes_pointwise (writes : List ByteWrite) (initial : ByteMemory) (address : Nat) :
    applyWrites writes initial address=byteResult writes initial address := by
  induction writes using List.reverseRecOn with
  | nil => rfl
  | append_singleton rest w ih =>
    rw [writes_append]
    simp only [applyWrites,byteWrite]
    unfold byteResult
    simp only [List.reverse_append,List.reverse_cons,List.reverse_nil,List.nil_append,List.cons_append,List.find?_cons]
    by_cases h : w.outputAddress=address
    · simp [h]
    · simp [h,Ne.symm h,ih,byteResult]
theorem writes_memory (writes : List ByteWrite) (initial : ByteMemory) :
    applyWrites writes initial=byteResult writes initial := funext (writes_pointwise writes initial)

theorem writes_frame (writes : List ByteWrite) (initial : ByteMemory) (address : Nat)
    (outside : ∀ w∈writes, w.outputAddress≠address) :
    applyWrites writes initial address=initial address := by
  induction writes generalizing initial with
  | nil => rfl
  | cons w rest ih =>
    rw [applyWrites,ih _ (fun v hv => outside v (by simp [hv]))]
    simp [byteWrite,Ne.symm (outside w (by simp))]

theorem word_write_bounds (d : Storage) (pid lane input output actualOutput : Nat) (word : Word d)
    (r : ByteRegion) (inside : InBacking r output d.bytes) :
    ∀ w∈wordWrites d pid lane input output actualOutput word, r.Contains w.outputAddress := by
  intro w hw
  obtain ⟨k,hk,rfl⟩ := List.mem_map.mp hw
  have hlt := List.mem_range.mp hk
  unfold InBacking at inside
  exact ⟨by dsimp; omega,by dsimp; omega⟩

theorem large_size_cast_examples :
    (BitVec.ofNat 32 (2^31)).toInt= -(2^31) ∧
    active32 (2^31) 16 0 0=Bool.false ∧
    active32 (2^32+1) 16 0 0=Bool.true ∧
    active32 (2^32+1) 16 0 1=Bool.false ∧
    (offset32 16 (2^28) 0).toNat=0 := by decide +kernel

#specNonCircular byteResult avoiding [applyWrites,byteWrite,source,Expr.eval]
#axiomsClean writes_memory
#axiomsClean writes_frame
#axiomsClean word_write_bounds
#axiomsClean translated_backing_bounds
#axiomsClean large_size_cast_examples
#guard_msgs (drop info) in
#auditModuleAxioms
end
end TritonMul2Finite
