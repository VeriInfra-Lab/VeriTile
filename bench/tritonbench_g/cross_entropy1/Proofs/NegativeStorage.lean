import bench.tritonbench_g.cross_entropy1.Proofs.Storage

namespace VeriTile.Bench.Repairs.CrossEntropy1.Physical
open VeriTile.Triton Source Host
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

def flipSign (word : Word) : Word := word ^^^ 0x80000000

@[simp] theorem flipSign_involutive (word : Word) : flipSign (flipSign word)=word := by
  simp [flipSign,BitVec.xor_assoc]

/-- Torch's lazy-negative view changes the sign bit of the logical FP32
observation. All payloads, including signed zero and NaNs, remain in scope. -/
def EncodesNegation {α σ : Type} (encode : α → Word) (obs : Observations α σ) : Prop :=
  ∀ v, encode (obs.neg v)=flipSign (encode v)

theorem raw_of_logical {α σ : Type} [Inhabited α] (encode : α → Word) (obs : Observations α σ)
    (negation : EncodesNegation encode obs) (s : World α σ) (t : Tensor) (i : Nat) :
    rawWord encode s t.region (t.index i)=
      if t.isNeg then flipSign (encode (value obs s t i)) else encode (value obs s t i) := by
  by_cases neg : t.isNeg=Bool.true
  · simp only [rawWord,value,if_pos neg]
    rw [negation _,flipSign_involutive]
  · simp only [rawWord,value,if_neg neg]

/-- A logical copy into the original saved destination has the corresponding
raw sign-bit behavior. The statement keeps the original descriptor and offset. -/
theorem copied_raw {α σ : Type} [Inhabited α] (encode : α → Word) (obs : Observations α σ)
    (negation : EncodesNegation encode obs) (rt : Runtime α σ) (lib : Library obs rt)
    (dst src result : Tensor) (before after : World α σ)
    (dstLive : before.live dst.region) (srcLive : before.live src.region) (shape : dst.shape=src.shape)
    (call : rt.copy dst src before=.ok result after) (i : Nat) (inside : i<dst.numel) :
    result=dst ∧ rawWord encode after dst.region (dst.index i)=
      if dst.isNeg then flipSign (encode (value obs before src i)) else encode (value obs before src i) := by
  have copied := lib.copy dst src before result after dstLive srcLive shape call
  exact ⟨copied.1,by rw [raw_of_logical encode obs negation,copied.2.2.1 i inside]⟩

theorem logical_value_bytes {α σ : Type} [Inhabited α] (layout : Layout σ)
    (encode : α → Word) (obs : Observations α σ) (negation : EncodesNegation encode obs)
    (s : World α σ) (t : Tensor) (i : Nat) (j : Fin 4) (v : α)
    (active : layout.active s.native t.region) (backed : 4*(t.index i+1)≤(layout.buffer s.native t.region).capacity)
    (logical : value obs s t i=v) :
    bytes layout encode s (layout.buffer s.native t.region).device
      ((layout.buffer s.native t.region).base+4*t.index i+j.val)=
      byteOf (if t.isNeg then flipSign (encode v) else encode v) j := by
  rw [raw_bytes layout encode s t.region (t.index i) j active (by have h := j.isLt; omega),
    raw_of_logical encode obs negation,logical]

theorem negative_zero_payload : flipSign 0=0x80000000 ∧ flipSign 0x7fc12345=0xffc12345 := by decide +kernel

#axiomsClean flipSign_involutive
#axiomsClean raw_of_logical
#axiomsClean copied_raw
#axiomsClean logical_value_bytes
#axiomsClean negative_zero_payload
#specNonCircular EncodesNegation avoiding [Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Physical
