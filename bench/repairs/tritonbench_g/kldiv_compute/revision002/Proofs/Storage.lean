import bench.repairs.tritonbench_g.kldiv_compute.revision002.Proofs.Host
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.ConditionalNative

namespace VeriTile.Bench.Repairs.KldivCompute.Revision002.Physical
open VeriTile.Triton FP.Structural Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor World)
namespace Native
abbrev Buffer := VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.Buffer
abbrev Supported := VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.Supported
abbrev source_control_exact := VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.source_control_exact
abbrev active_address_exact := VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.active_address_exact
abbrev active_byte_bound := VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.active_byte_bound
end Native

abbrev Word := BitVec 32
def byteOf (word : Word) (byte : Fin 4) : BitVec 8 :=
  BitVec.ofNat 8 (word.toNat / 2^(8*byte.val))

/-- Ordinary FP32 raw storage realization for the allocation identities of one
call. `buffer` maps those identities to the actual device/base/capacity; it is
stable while allocations live. Words are all 2^32 patterns. No floating result,
source execution, copy correctness or global kernel law is part of this API. -/
structure Memory (σ : Type) where
  buffer : RegionName → Native.Buffer
  readable : Nat → Nat → Prop
  writable : Nat → Nat → Prop
  word : World Word σ → Nat → Nat → Word
  byte : World Word σ → Nat → Nat → BitVec 8
  load : ∀ s r i, s.live r → 4*(i+1)≤(buffer r).capacity →
    word s (buffer r).device ((buffer r).base+4*i)=(s.kernel.mem r i).read .real
  bytes : ∀ s r i (b : Fin 4), s.live r → 4*i+b.val<(buffer r).capacity →
    byte s (buffer r).device ((buffer r).base+4*i+b.val)=
      byteOf ((s.kernel.mem r i).read .real) b

def Backed {σ : Type} (mem : Memory σ) (s : World Word σ) (t : Tensor) : Prop :=
  s.live t.region ∧ ∀ i, i<t.numel → 4*(t.index i+1)≤(mem.buffer t.region).capacity

def logical {σ : Type} (mem : Memory σ) (neg : Word → Word)
    (s : World Word σ) (t : Tensor) (i : Nat) : Word :=
  let raw := mem.word s (mem.buffer t.region).device ((mem.buffer t.region).base+4*t.index i)
  if t.isNeg then neg raw else raw

theorem logical_eq {σ : Type} (mem : Memory σ) (neg : Word → Word)
    (s : World Word σ) (t : Tensor) (h : Backed mem s t) (i : Nat) (hi : i<t.numel) :
    logical mem neg s t i=value neg s.kernel t i := by
  unfold logical value
  rw [mem.load s t.region (t.index i) h.1 (h.2 i hi)]

/-- Ordinary successful setup contract for the exact requested pointers and
count. The supported grid contract is conditional; installed overflowing
launcher/grid-product behavior is explicitly deferred, not hidden in numerics. -/
structure Invocation {σ : Type} (mem : Memory σ) (x y work : Tensor) (n : Nat) : Prop where
  supported : Native.Supported n
  x_backing : (mem.buffer x.region).Valid n
  y_backing : (mem.buffer y.region).Valid n
  work_backing : (mem.buffer work.region).Valid n
  devices : (mem.buffer x.region).device=(mem.buffer work.region).device ∧
    (mem.buffer y.region).device=(mem.buffer work.region).device
  read_x : ∀ i b, i<n → b<4 → mem.readable (mem.buffer x.region).device
    ((mem.buffer x.region).base+4*i+b)
  read_y : ∀ i b, i<n → b<4 → mem.readable (mem.buffer y.region).device
    ((mem.buffer y.region).base+4*i+b)
  write_work : ∀ i b, i<n → b<4 → mem.writable (mem.buffer work.region).device
    ((mem.buffer work.region).base+4*i+b)

/-- Successful local copy_ leaves its actual destination backed. This is an
ordinary tensor-library validity law, with no KLDIV arithmetic condition. -/
structure Library {σ : Type} (rt : Runtime Word σ) (mem : Memory σ) : Prop where
  copy : ∀ dst src s out t, rt.copy dst src s=.ok out t → Backed mem t dst
  setup : ∀ x y work n s t, rt.setup x y work n s=.ok () t → Invocation mem x y work n

def ReturnedBytes {σ : Type} (mem : Memory σ) (neg : Word → Word) (M : Algebra Word)
    (x y output : Tensor) (initial final : World Word σ) : Prop :=
  (∀ i, i<x.numel → mem.word final (mem.buffer output.region).device
    ((mem.buffer output.region).base+4*output.index i)=
      Program.scalar M (logical mem neg initial x i) (logical mem neg initial y i)) ∧
  (∀ i (b : Fin 4), i<x.numel → mem.byte final (mem.buffer output.region).device
    ((mem.buffer output.region).base+4*output.index i+b.val)=
      byteOf (Program.scalar M (logical mem neg initial x i) (logical mem neg initial y i)) b) ∧
  (∀ r offset, initial.live r → offset<(mem.buffer r).capacity →
    mem.byte final (mem.buffer r).device ((mem.buffer r).base+offset)=
      mem.byte initial (mem.buffer r).device ((mem.buffer r).base+offset))

/-- Read the exact four bytes through this actual caller-returned descriptor,
including its storage offset and preserved empty_like layout. Every byte of
initially live storage is preserved, including arbitrary partial trailing words.
The byte law requires only that this individual byte exists. -/
theorem returned_bytes {σ : Type} (mem : Memory σ) (neg : Word → Word) (M : Algebra Word)
    (rt : Runtime Word σ) (lib : Library rt mem) (x y output : Tensor)
    (initial final : World Word σ) (e : ReturnedData neg M rt x y initial output final)
    (hx : Backed mem initial x) (hy : Backed mem initial y) :
    ReturnedBytes mem neg M x y output initial final := by
  have ho := lib.copy _ _ _ _ _ e.copy_call
  have words : ∀ i, i<x.numel → (final.kernel.mem output.region (output.index i)).read .real=
      Program.scalar M (logical mem neg initial x i) (logical mem neg initial y i) := by
    intro i hi
    have hiy : i<y.numel := lt_of_lt_of_le hi e.input_prefix
    rw [logical_eq mem neg initial x hx i hi,logical_eq mem neg initial y hy i hiy]
    simpa [value,e.output_not_neg] using e.contents i hi
  refine ⟨?_,?_,?_⟩
  · intro i hi
    rw [mem.load _ _ _ ho.1 (ho.2 i (by simpa [e.metadata.numel] using hi)),words i hi]
  · intro i b hi
    have cap := ho.2 i (by simpa [e.metadata.numel] using hi)
    have hb := b.isLt
    rw [mem.bytes _ _ _ b ho.1 (by omega),words i hi]
  · intro r offset hr cap
    let b : Fin 4 := ⟨offset%4, Nat.mod_lt _ (by decide)⟩
    have address : 4*(offset/4)+b.val=offset := by dsimp [b]; omega
    have bound : 4*(offset/4)+b.val<(mem.buffer r).capacity := by rw [address]; exact cap
    have after := mem.bytes final r (offset/4) b (e.caller_live _ hr) bound
    have before := mem.bytes initial r (offset/4) b hr bound
    rw [e.caller_frame r (offset/4) hr] at after
    simpa only [Nat.add_assoc,address] using after.trans before.symm

/-- Derive the native control/address obligations on the launch extracted
from this same caller execution; all source arithmetic has been widened before
products. Inactive lanes need no backing address. -/
def NativeData {σ : Type} {neg : Word → Word} {M : Algebra Word}
    {rt : Runtime Word σ} {x y output : Tensor} {initial final : World Word σ}
    (mem : Memory σ) (e : ReturnedData neg M rt x y initial output final) : Prop :=
  Invocation mem e.px e.py e.work output.numel ∧
  ∀ pid lane, pid<VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.grid output.numel → lane<1024 →
    VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.mask output.numel pid lane=Bool.true →
    ∀ t, t=e.px ∨ t=e.py ∨ t=e.work →
      (VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.pointer (mem.buffer t.region).base pid lane).toNat=
        (mem.buffer t.region).base+4*(pid*1024+lane) ∧
      (VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.pointer (mem.buffer t.region).base pid lane).toNat+4≤
        (mem.buffer t.region).base+(mem.buffer t.region).capacity ∧
      (VeriTile.Bench.Repairs.KldivCompute.ConditionalSupport.Native.pointer (mem.buffer t.region).base pid lane).toNat+4≤2^64

theorem actual_native {σ : Type} {neg : Word → Word} {M : Algebra Word}
    {rt : Runtime Word σ} {x y output : Tensor} {initial final : World Word σ}
    (mem : Memory σ) (lib : Library rt mem)
    (e : ReturnedData neg M rt x y initial output final) : NativeData mem e := by
  have inv := lib.setup _ _ _ _ _ _ e.launch_data.setup_call
  refine ⟨inv,?_⟩
  intro pid lane hp hl active t role
  have valid : (mem.buffer t.region).Valid output.numel := by
    rcases role with rfl|rfl|rfl
    · exact inv.x_backing
    · exact inv.y_backing
    · exact inv.work_backing
  have a := Native.active_address_exact (mem.buffer t.region) output.numel pid lane
    inv.supported valid hp hl active
  exact ⟨a.1,a.2.2.1,a.2.2.2⟩

#axiomsClean actual_native
#specNonCircular Memory avoiding [Program.scalar,Program.grid,Host.run]
#specNonCircular Library avoiding [Program.scalar,Program.grid,Host.run]
#axiomsClean returned_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.KldivCompute.Revision002.Physical
