import bench.tritonbench_g.vector_addition.FiniteCorrect

namespace VeriTile.Bench.TritonBenchG.VectorAddition.Native
open VeriTile.Triton DTypeScope TypedHost Finite
noncomputable section
open Classical
set_option maxHeartbeats 6000000

/-- Host metadata remains signed64 nonnegative; the annotated launch argument
is extracted as i32. The original Python grid uses the unnarrowed host value. -/
def hostExtent (n : Nat) : Prop := n<2^63
def grid (n : Nat) : Nat := (n+1023)/1024
def gridSupported (n : Nat) : Prop := grid n≤2^31-1
def bound (n : Nat) : BitVec 32 := BitVec.ofNat 32 n
def offset (pid lane : Nat) : BitVec 32 :=
  BitVec.ofNat 32 pid * BitVec.ofNat 32 1024 + BitVec.ofNat 32 lane
def mask (n pid lane : Nat) : Bool := (offset pid lane).slt (bound n)
def pointer (data width pid lane : Nat) : Nat :=
  (BitVec.ofNat 64 data + (offset pid lane).signExtend 64 * BitVec.ofNat 64 width).toNat
def activeTile (n pid : Nat) : List Nat := (List.range 1024).filter (mask n pid)

theorem bound_minimum : (bound (2^31)).toInt=-(2^31 : Int) := by decide +kernel
theorem all_masks_false_at_two31 (pid lane : Nat) : mask (2^31) pid lane=Bool.false := by
  have lower : -(2^31 : Int)≤(offset pid lane).toInt := by simpa using BitVec.le_toInt (offset pid lane)
  rw [mask,BitVec.slt_eq_decide,bound_minimum]
  exact decide_eq_false (not_lt.mpr lower)
theorem active_tile_empty_at_two31 (pid : Nat) : activeTile (2^31) pid=[] := by
  apply List.filter_eq_nil_iff.mpr
  intro lane _
  rw [all_masks_false_at_two31]
  decide
theorem counterexample_grid : grid (2^31)=2^21 ∧ gridSupported (2^31) ∧ hostExtent (2^31) := by
  norm_num [grid,gridSupported,hostExtent]
theorem wrapped_negative_lane :
    (offset (2^21) 0).toInt=-(2^31 : Int) ∧ mask (2^31+1) (2^21) 0=Bool.true := by decide +kernel
theorem wrapped_zero_lane : offset (2^22) 0=0 ∧ mask (2^32+1) (2^22) 0=Bool.true := by decide +kernel
theorem wrapped_pointer_can_reenter : pointer (2^33+32) 4 (2^21) 0=32 := by decide +kernel
theorem bounded_arithmetic (n pid lane : Nat) (hn : n<2^31)
    (hp : pid<grid n) (hl : lane<1024) :
    (offset pid lane).toInt=(pid*1024+lane : Nat) ∧ mask n pid lane=decide (pid*1024+lane<n) :=
  (source_index_arithmetic n pid lane hn hp hl).2
theorem bounded_pointer (v : View) (n pid lane : Nat) (hn : n<2^31)
    (valid : v.Valid n (rounded n)) (hp : pid<grid n) (hl : lane<1024) :
    pointer (v.address 0) (bytes v.dtype) pid lane=v.address (pid*1024+lane) :=
  source_pointer_exact v n pid lane hn valid hp hl

structure Buffer where
  base : Nat
  capacity : Nat
  data : Nat
  device : Option Nat
  readable : Nat → Bool
  writable : Nat → Bool

def Buffer.Contains (v : Buffer) (address : Nat) : Prop := v.base≤address ∧ address<v.base+v.capacity
def Buffer.Inside (v : Buffer) (address width : Nat) : Prop :=
  v.base≤address ∧ address+width≤v.base+v.capacity ∧ address+width≤2^64
def Buffer.Disjoint (a b : Buffer) : Prop := a.base+a.capacity≤b.base ∨ b.base+b.capacity≤a.base
def Buffer.Readable (v : Buffer) (device address width : Nat) : Prop :=
  v.Inside address width ∧ v.readable device=Bool.true
def Buffer.Writable (v : Buffer) (device address width : Nat) : Prop :=
  v.Inside address width ∧ v.writable device=Bool.true
structure Layout where
  x : Buffer
  y : Buffer
  output : Buffer
  device : Nat
def Layout.Fresh (p : Layout) : Prop := p.x.Disjoint p.output ∧ p.y.Disjoint p.output

/-- Allocation ownership is metadata. Access permissions are indexed by the
actual launching device and may allow a peer allocation from another GPU. -/
theorem peer_read_permitted :
    let peer : Buffer := ⟨4096,16,4096,some 1,(fun d => decide (d=0)),(fun _ => Bool.false)⟩
    peer.device≠some 0 ∧ peer.Readable 0 4096 4 := by
  norm_num [Buffer.Readable,Buffer.Inside]

theorem disjoint_span (input output : Buffer) (h : input.Disjoint output)
    (a w b z : Nat) (hi : input.Inside a w) (ho : output.Inside b z) :
    SpanDisjoint b (b+z) a (a+w) := by
  unfold Buffer.Disjoint at h
  unfold Buffer.Inside at hi ho
  unfold SpanDisjoint
  rcases h with h|h <;> omega

#axiomsClean all_masks_false_at_two31
#axiomsClean active_tile_empty_at_two31
#axiomsClean counterexample_grid
#axiomsClean wrapped_negative_lane
#axiomsClean wrapped_zero_lane
#axiomsClean wrapped_pointer_can_reenter
#axiomsClean bounded_arithmetic
#axiomsClean bounded_pointer
#axiomsClean peer_read_permitted
#axiomsClean disjoint_span
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.Native
