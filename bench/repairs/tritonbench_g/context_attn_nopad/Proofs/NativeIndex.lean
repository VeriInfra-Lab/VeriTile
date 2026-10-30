import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Source
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeIndex

/-- Exactly the host's finite axis partition. A chunk base is a multiple of
its native grid limit; its local program ID is below the final clipped size. -/
def AxisJob (n limit chunk localId : Nat) : Prop :=
  chunk*limit < n ∧ localId < min limit (n-chunk*limit)

theorem axis_coverage {n limit i : Nat} (hl : 0 < limit) (hi : i < n) :
    AxisJob n limit (i/limit) (i%limit) ∧ (i/limit)*limit+i%limit = i := by
  have hm := Nat.mod_lt i hl
  have he : (i/limit)*limit+i%limit = i := by
    simpa [Nat.mul_comm] using Nat.div_add_mod i limit
  constructor
  · constructor
    · omega
    · exact lt_min hm (by omega)
  · omega

theorem axis_inside {n limit chunk localId : Nat} (h : AxisJob n limit chunk localId) :
    chunk*limit+localId < n := by
  have := (lt_min_iff.mp h.2).2
  omega

theorem axis_unique {n limit chunk chunk' localId localId' : Nat}
    (hl : 0 < limit) (h : AxisJob n limit chunk localId)
    (h' : AxisJob n limit chunk' localId')
    (he : chunk*limit+localId = chunk'*limit+localId') :
    chunk = chunk' ∧ localId = localId' := by
  have hlocal := (lt_min_iff.mp h.2).1
  have hlocal' := (lt_min_iff.mp h'.2).1
  have hd : (chunk*limit+localId)/limit = chunk := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hl, Nat.div_eq_of_lt hlocal, Nat.zero_add]
  have hd' : (chunk'*limit+localId')/limit = chunk' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hl, Nat.div_eq_of_lt hlocal', Nat.zero_add]
  have hc : chunk = chunk' := by rw [← hd, ← hd', he]
  subst chunk
  exact ⟨rfl, Nat.add_left_cancel he⟩

theorem native_limits_positive : 0 < (2147483647 : Nat) ∧ 0 < (65535 : Nat) := by decide

/-- This pair reaches a valid nonnegative mathematical element offset while
an int32 product wraps negative. The repaired source widens before multiply. -/
theorem int32_stride_counterexample :
    (2 : Int) * 1073741824 = 2147483648 ∧
    (2 : Int) * 1073741824 > 2147483647 := by norm_num

/-- Local signed-64 representation lemma. It does not add a native-safe domain
to the host theorem or claim a compiler/machine refinement theorem. -/
def signed64 (x : Int) : Int := ((x + 2^63) % (2^64)) - 2^63

theorem signed64_exact {x : Int} (hlo : -(2^63) ≤ x) (hhi : x < 2^63) :
    signed64 x = x := by
  unfold signed64
  rw [Int.emod_eq_of_lt (by omega) (by norm_num at *; omega)]
  omega

/-- The exact logical metadata maximum bounds every sequence. Clipping a
Python max_input_len to that maximum removes only programs with no stores. -/
def capMaxInputLen (n bound : Nat) : Nat := min n bound

theorem capped_blocks (n bound : Nat) :
    (capMaxInputLen n bound + 127)/128 = min ((n+127)/128) ((bound+127)/128) := by
  unfold capMaxInputLen
  omega

theorem cap_preserves_active (n bound t : Nat) (ht : t < bound) :
    t < ((capMaxInputLen n bound+127)/128)*128 ↔ t < ((n+127)/128)*128 := by
  rw [capped_blocks]
  have hb : bound ≤ ((bound+127)/128)*128 := by omega
  omega

/-- The effective launch retains all tensor descriptors, metadata and arithmetic. -/
def capCall (c : Call) (bound : Nat) : Call :=
  { c with maxInputLen := capMaxInputLen c.maxInputLen bound }

theorem cap_source (c : Call) (bound : Nat) (r : Regions) :
    kernel (capCall c bound) r = kernel c r := rfl

theorem cap_shape {c : Call} (bound : Nat)
    (hs : VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Shape c) :
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Shape (capCall c bound) := by
  cases hs
  constructor <;> assumption

theorem cap_complete {c : Call} (bound : Nat)
    (hm : ∀ b, b < c.batches → c.length b ≤ bound)
    (hc : c.completeGrid) : (capCall c bound).completeGrid := by
  intro b hb
  change c.length b ≤ min c.maxInputLen bound
  exact le_min (hc b hb) (hm b hb)

/-- Ordinary tensor allocations have representable signed-64 storage extents.
This local bound composes with Host.Memory's tensor-address bounds. It is not a
new restriction on logical tensor shapes or on max_input_len. -/
theorem active_address_exact {addr extent : Nat}
    (haddr : addr < extent) (hextent : extent ≤ 2^63) :
    signed64 (addr : Int) = (addr : Int) := by
  apply signed64_exact
  · norm_num
  · exact_mod_cast lt_of_lt_of_le haddr hextent

end VeriTile.Bench.Repairs.ContextAttnNopad.NativeIndex
