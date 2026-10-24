import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Domain
import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Native
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000

abbrev Bytes := Nat → BitVec 8

def byteOf (w : BitVec 32) (j : Fin 4) : BitVec 8 := w.extractLsb' (8*j.val) 8

def wordOf (bytes : Fin 4 → BitVec 8) : BitVec 32 :=
  ((bytes 3 ++ bytes 2) ++ bytes 1) ++ bytes 0

theorem word_bytes (w : BitVec 32) : wordOf (byteOf w)=w := by
  change (((w.extractLsb' 24 8 ++ w.extractLsb' 16 8) ++ w.extractLsb' 8 8) ++ w.extractLsb' 0 8) = w
  bv_decide

/-- Ordinary allocator/view contract: address decoder is a partial inverse of
fresh, disjoint output intervals. Read-only input regions may overlap.
No computed values are fields here. -/
structure Layout (c : Shape) where
  base : Region → Nat
  decode : Nat → Option (Region × Nat × Fin 4)
  inverse : ∀ r b, WholeOutput c r b → ∀ j : Fin 4,
    decode (base r+4*b+j.val)=some (r,b,j)
  sound : ∀ addr r b j, decode addr=some (r,b,j) →
    b<extent c r ∧ addr=base r+4*b+j.val
  addressWidth : ∀ r b, b<extent c r → base r+4*b+3<2^64
  fresh : ∀ r,InputRegion r → ∀ b,b<extent c r → ∀ j : Fin 4,
    ∀ s d,WholeOutput c s d → ∀ k : Fin 4,base r+4*b+j.val≠base s+4*d+k.val

def loadWords (c : Shape) (layout : Layout c) (memory : Bytes) : Memory (BitVec 32) :=
  fun r b => wordOf (fun j => memory (layout.base r+4*b+j.val))

/-- Concrete byte realization of the already computed source stores. The
runtime boundary consists of this finite disjoint scatter, not an output oracle. -/
def scatterBytes (c : Shape) (layout : Layout c) (old : Bytes)
    (words : Memory (BitVec 32)) : Bytes := fun addr =>
  match layout.decode addr with
  | none => old addr
  | some (r,b,j) => if WholeOutput c r b then byteOf (words r b) j else old addr

def physicalForward (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) : Bytes :=
  scatterBytes c layout memory
    (Native.nativeForward a compiler c plan scale (loadWords c layout memory))

theorem byte_address_unique (c : Shape) (layout : Layout c)
    (r s : Region) (b d : Nat) (j k : Fin 4) (hb : WholeOutput c r b) (hd : WholeOutput c s d)
    (h : layout.base r+4*b+j.val=layout.base s+4*d+k.val) :
    r=s ∧ b=d ∧ j=k := by
  have hx := congrArg layout.decode h
  rw [layout.inverse r b hb j,layout.inverse s d hd k] at hx
  simpa only [Option.some.injEq,Prod.mk.injEq] using hx

theorem scatter_byte (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (r : Region) (b : Nat) (j : Fin 4) (h : WholeOutput c r b) :
    scatterBytes c layout old words (layout.base r+4*b+j.val)=byteOf (words r b) j := by
  simp only [scatterBytes,layout.inverse r b h j,if_pos h]

theorem scatter_word (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (r : Region) (b : Nat) (h : WholeOutput c r b) :
    loadWords c layout (scatterBytes c layout old words) r b=words r b := by
  unfold loadWords
  have hb : (fun j : Fin 4 => scatterBytes c layout old words (layout.base r+4*b+j.val))=
      byteOf (words r b) := by
    funext j
    exact scatter_byte c layout old words r b j h
  rw [hb,word_bytes]

theorem scatter_frame (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (addr : Nat) (h : ∀ r b j, layout.decode addr=some (r,b,j) → ¬WholeOutput c r b) :
    scatterBytes c layout old words addr=old addr := by
  unfold scatterBytes
  cases hd : layout.decode addr with
  | none => rfl
  | some cell =>
    rcases cell with ⟨r,b,j⟩
    simp only [if_neg (h r b j hd)]

/-- Exact bytes outside all actual output intervals, including padding, input
backing gaps, disabled HT and unrelated allocations, remain unchanged. -/
theorem physical_frame (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes) (addr : Nat)
    (h : ∀ r b j, layout.decode addr=some (r,b,j) → ¬WholeOutput c r b) :
    physicalForward a compiler c plan scale layout memory addr=memory addr :=
  scatter_frame c layout memory _ addr h

theorem physical_returned (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (hc : Sizes c) (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes)
    (z t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    loadWords c layout (physicalForward a compiler c plan scale layout memory)
      .result (resultAddress c z t f)=
    Native.observe a compiler (scalarOutput Native.sourceOps c plan
      (inputs c (Native.entry (loadWords c layout memory))) (.word .outside 0 scale) z t f) := by
  unfold physicalForward
  rw [scatter_word c layout memory _ .result (resultAddress c z t f)
    (Or.inr ⟨rfl,result_bound c z t f hz ht hf⟩)]
  exact Native.returned a compiler c hc plan scale _ z t f hz ht hf

theorem physical_terminal (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (hc : Sizes c) (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes)
    (hs : c.final=true) (z e f : Nat) (hz : z<c.Z) (he : e<c.K) (hf : f<c.V) :
    loadWords c layout (physicalForward a compiler c plan scale layout memory)
      .ht (stateAddress c z e f)=
    Native.observe a compiler (.store32 (scalarState Native.sourceOps c
      (inputs c (Native.entry (loadWords c layout memory))) z e f c.T)) := by
  unfold physicalForward
  rw [scatter_word c layout memory _ .ht (stateAddress c z e f)
    (Or.inl (Or.inr ⟨hs,rfl,state_address_bound c z e f hz he hf⟩))]
  exact Native.terminal a compiler c hc plan scale _ hs z e f hz he hf

/-- Byte loads from all source input regions remain the caller's original bytes. -/
theorem physical_inputs (a : Native.Instructions) (compiler : Native.Compiler) (c : Shape)
    (plan : Plans c) (scale : BitVec 32) (layout : Layout c) (memory : Bytes)
    (r : Region) (hr : InputRegion r) (b : Nat) (hb : b<extent c r) (j : Fin 4) :
    physicalForward a compiler c plan scale layout memory (layout.base r+4*b+j.val)=
      memory (layout.base r+4*b+j.val) := by
  apply physical_frame
  intro s d k hd hw
  exact layout.fresh r hr b hb j s d hw k (layout.sound _ s d k hd).2

#axiomsClean physical_returned
#axiomsClean physical_terminal
#axiomsClean physical_frame
#axiomsClean physical_inputs
#axiomsClean byte_address_unique
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
