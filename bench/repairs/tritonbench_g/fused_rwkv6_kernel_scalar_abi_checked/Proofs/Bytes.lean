import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Trace

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Physical
open FusedRwkv6Kernel
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 8000000

/-- Only the two kernel output allocations participate. No Torch sum output,
returned object, autograd state or wrapper execution appears in this contract. -/
structure Layout (c : Shape) where
  base : Region → Nat
  decode : Nat → Option (Region × Nat × Fin 4)
  inverse : ∀ r b,Output c r b → ∀ j : Fin 4,decode (base r+4*b+j.val)=some (r,b,j)
  sound : ∀ addr r b j,decode addr=some (r,b,j) → Output c r b ∧ addr=base r+4*b+j.val
  width : ∀ r b,Output c r b → base r+4*b+3<2^64
  fresh : ∀ r,InputRegion r → ∀ b,b<extent c r → ∀ j : Fin 4,
    ∀ s d,Output c s d → ∀ k : Fin 4,base r+4*b+j.val≠base s+4*d+k.val

def loadWords (c : Shape) (layout : Layout c) (memory : Bytes) : Memory (BitVec 32) :=
  fun r b => wordOf (fun j => memory (layout.base r+4*b+j.val))

/-- Exactly the finite store sites of the kernel, excluding the old host sum. -/
def wordTrace (c : Shape) : List (Region × Nat) :=
  (programs c).flatMap fun p => partialTrace c p++finalTrace c p

theorem word_trace_mem (c : Shape) (hc : Sizes c) (r : Region) (b : Nat) :
    (r,b)∈wordTrace c ↔ Output c r b := by
  simp only [wordTrace,List.mem_flatMap,List.mem_append,partial_trace_mem,final_trace_mem]
  constructor
  · rintro ⟨p,hp,hw⟩
    exact writes_output c p ((mem_programs c p).mp hp) hw
  · exact cover c hc

def byteTrace (c : Shape) (layout : Layout c) (words : Memory (BitVec 32)) : List ByteEvent :=
  (wordTrace c).flatMap fun (r,b) => List.ofFn fun j : Fin 4 =>
    ⟨layout.base r+4*b+j.val,byteOf (words r b) j⟩

def scatter (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32)) : Bytes :=
  fun addr => match layout.decode addr with
    | none => old addr
    | some (r,b,j) => byteOf (words r b) j

theorem scatter_byte (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (r : Region) (b : Nat) (j : Fin 4) (hw : Output c r b) :
    scatter c layout old words (layout.base r+4*b+j.val)=byteOf (words r b) j := by
  simp [scatter,layout.inverse r b hw j]

theorem scatter_word (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (r : Region) (b : Nat) (hw : Output c r b) :
    loadWords c layout (scatter c layout old words) r b=words r b := by
  unfold loadWords
  have h : (fun j : Fin 4 => scatter c layout old words (layout.base r+4*b+j.val))=byteOf (words r b) := by
    funext j
    exact scatter_byte c layout old words r b j hw
  rw [h,word_bytes]

def Outside (c : Shape) (layout : Layout c) (addr : Nat) : Prop :=
  ∀ r b,Output c r b → ∀ j : Fin 4,addr≠layout.base r+4*b+j.val

theorem scatter_frame (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (addr : Nat) (outside : Outside c layout addr) : scatter c layout old words addr=old addr := by
  unfold scatter
  cases hd : layout.decode addr with
  | none => rfl
  | some cell =>
    rcases cell with ⟨r,b,j⟩
    have hs:=layout.sound addr r b j hd
    exact False.elim (outside r b hs.1 j hs.2)

theorem byte_trace_mem (c : Shape) (layout : Layout c) (words : Memory (BitVec 32)) (e : ByteEvent) :
    e∈byteTrace c layout words ↔ ∃ r b,(r,b)∈wordTrace c ∧ ∃ j : Fin 4,
      e=⟨layout.base r+4*b+j.val,byteOf (words r b) j⟩ := by
  simp only [byteTrace,List.mem_flatMap,List.mem_ofFn]
  constructor
  · rintro ⟨⟨r,b⟩,h,j,hj⟩
    exact ⟨r,b,h,j,hj.symm⟩
  · rintro ⟨r,b,h,j,rfl⟩
    exact ⟨(r,b),h,j,rfl⟩

/-- Actual finite byte stores realize the output words and preserve every
other byte. A computed-output equality is not a premise. -/
theorem trace_realizes (c : Shape) (hc : Sizes c) (layout : Layout c)
    (words : Memory (BitVec 32)) (memory : Bytes) :
    executeBytes (byteTrace c layout words) memory=scatter c layout memory words := by
  funext addr
  have h : ∀ e∈byteTrace c layout words,e.value=scatter c layout memory words e.address := by
    intro e he
    obtain ⟨r,b,hw,j,rfl⟩:=(byte_trace_mem c layout words e).mp he
    exact (scatter_byte c layout memory words r b j ((word_trace_mem c hc r b).mp hw)).symm
  rw [execute_consistent _ _ _ h]
  by_cases hit : ∃ e∈byteTrace c layout words,e.address=addr
  · simp [hit]
  · rw [if_neg hit]
    symm
    apply scatter_frame
    intro r b hw j same
    apply hit
    refine ⟨⟨layout.base r+4*b+j.val,byteOf (words r b) j⟩,?_,same.symm⟩
    exact (byte_trace_mem c layout words _).mpr ⟨r,b,(word_trace_mem c hc r b).mpr hw,j,rfl⟩

theorem input_frame (c : Shape) (layout : Layout c) (old : Bytes) (words : Memory (BitVec 32))
    (r : Region) (hr : InputRegion r) (b : Nat) (hb : b<extent c r) (j : Fin 4) :
    scatter c layout old words (layout.base r+4*b+j.val)=old (layout.base r+4*b+j.val) := by
  apply scatter_frame
  exact fun s d hw k => layout.fresh r hr b hb j s d hw k

#axiomsClean trace_realizes
#axiomsClean scatter_word
#axiomsClean scatter_frame
#axiomsClean input_frame
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.Physical
