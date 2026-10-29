import bench.repairs.tritonbench_g.context_attn_llama.Kernels
import VeriTile.Triton.Float.Structural

namespace VeriTile.Bench.Repairs.ContextAttnLlama.FPViews
open VeriTile.Triton FP.Structural
noncomputable section

/-- IEEE binary32 logical negation, including signed zero and NaN payload bits.
The ordinary tensor-library view contract uses this sign-bit operation. -/
def sign (negative : Bool) (x : BitVec 32) : BitVec 32 :=
  if negative then x ^^^ (2147483648 : BitVec 32) else x

@[simp] theorem sign_involutive (negative : Bool) (x : BitVec 32) :
    sign negative (sign negative x) = x := by
  cases negative <;> simp [sign,BitVec.xor_assoc]

/-- The address and active predicate are captured before launch/copyback.
`initial` contains every tagged physical cell, including aliased metadata.
Only actual active writers need be distinct. -/
structure Output (I : Type) where
  owner : Nat
  address : I → RegionName × Nat
  negative : Bool
  initial : (RegionName × Nat) → Cell (BitVec 32)
  active : I → Prop
  scratchOffset : I → Nat

def Output.logical {I : Type} (o : Output I)
    (mem : (RegionName × Nat) → Cell (BitVec 32)) (i : I) : BitVec 32 :=
  sign o.negative ((mem (o.address i)).read .real)

def Output.copyBack {I : Type} (o : Output I) (scratch : I → BitVec 32) :
    (RegionName × Nat) → Cell (BitVec 32) := by
  classical
  exact fun cell => if h : ∃ i, o.active i ∧ o.address i = cell then
    .mk .real (sign o.negative (scratch h.choose)) else o.initial cell

theorem Output.copyBack_read {I : Type} (o : Output I) (scratch : I → BitVec 32)
    (unique : ∀ i j, o.active i → o.active j → o.address i = o.address j → i = j)
    (i : I) (hi : o.active i) :
    o.logical (o.copyBack scratch) i = scratch i := by
  classical
  have h : ∃ j, o.active j ∧ o.address j = o.address i := ⟨i,hi,rfl⟩
  have hj : h.choose = i := unique _ _ h.choose_spec.1 hi h.choose_spec.2
  simp [Output.logical,Output.copyBack,h,hj]

theorem Output.copyBack_frame {I : Type} (o : Output I) (scratch : I → BitVec 32)
    (cell : RegionName × Nat) (hc : ¬ ∃ i, o.active i ∧ o.address i = cell) :
    o.copyBack scratch cell = o.initial cell := by
  classical
  simp [Output.copyBack,hc]

/-- Copyback decisions do not reread metadata after any caller write. This
property holds even when those writes replace every physical metadata cell. -/
theorem Output.snapshot_stable {I : Type} (o : Output I) (scratch : I → BitVec 32)
    (changed : (RegionName × Nat) → Cell (BitVec 32)) (i : I) :
    ({o with initial := changed}).active i ↔ o.active i := Iff.rfl

/-- All eight resolved read roles are initial-memory contracts. They do not
assert anything about program success, program outputs or arithmetic. -/
structure Reads (r : Regions) (s : State (BitVec 32)) where
  query : Nat → BitVec 32
  key : Nat → BitVec 32
  value : Nat → BitVec 32
  start : Nat → Int
  sequence : Nat → Int
  prompt : Nat → Int
  request : Nat → Int
  pages : Nat → Int
  query_eq : ∀ n, (s.mem r.query n).read .real = query n
  key_eq : ∀ n, (s.mem r.key n).read .real = key n
  value_eq : ∀ n, (s.mem r.value n).read .real = value n
  start_eq : ∀ n, (s.mem r.start n).read .int = start n
  sequence_eq : ∀ n, (s.mem r.sequence n).read .int = sequence n
  prompt_eq : ∀ n, (s.mem r.prompt n).read .int = prompt n
  request_eq : ∀ n, (s.mem r.request n).read .int = request n
  pages_eq : ∀ n, (s.mem r.pages n).read .int = pages n

end
end VeriTile.Bench.Repairs.ContextAttnLlama.FPViews
