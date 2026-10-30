import bench.tritonbench_g.sin_computation.Proofs.NativeScalar

namespace SinComputationNative
open VeriTile.Bench.TritonBenchG.VectorAddition
open DTypeScope TypedHost Finite
abbrev hostExtent := Native.hostExtent
def grid (n : Nat) : Nat := n
def gridSupported (n : Nat) : Prop := grid n≤2^31-1
abbrev bound := Native.bound
def offset (pid lane : Nat) : BitVec 32 := BitVec.ofNat 32 pid * BitVec.ofNat 32 4 + BitVec.ofNat 32 lane
def mask (n pid lane : Nat) : Bool := (offset pid lane).slt (bound n)
def pointer (data width pid lane : Nat) : Nat :=
  (BitVec.ofNat 64 data + (offset pid lane).signExtend 64 * BitVec.ofNat 64 width).toNat
def activeTile (n pid : Nat) : List Nat := (List.range 4).filter (mask n pid)
abbrev Buffer := Native.Buffer
structure Layout where
  x : Buffer
  output : Buffer
  device : Nat

theorem all_masks_false_at_two31 (pid lane : Nat) : mask (2^31) pid lane=Bool.false := by
  have lower : -(2^31 : Int)≤(offset pid lane).toInt := by simpa using BitVec.le_toInt (offset pid lane)
  rw [mask,BitVec.slt_eq_decide,Native.bound_minimum]
  exact decide_eq_false (not_lt.mpr lower)
theorem active_tile_empty_at_two31 (pid : Nat) : activeTile (2^31) pid=[] := by
  apply List.filter_eq_nil_iff.mpr
  intro lane _
  rw [all_masks_false_at_two31]
  decide
theorem grid_two31_rejected : ¬gridSupported (2^31) ∧ hostExtent (2^31) := by norm_num [gridSupported,grid,hostExtent,Native.hostExtent]
theorem wrapped_negative_within_launch_domain :
    gridSupported (2^29+1) ∧ (offset (2^29) 0).toInt=-(2^31 : Int) ∧ mask (2^29+1) (2^29) 0=Bool.true := by unfold gridSupported grid; decide +kernel
theorem repeated_lane_within_launch_domain :
    gridSupported (2^30+1) ∧ offset (2^30) 0=offset 0 0 ∧ mask (2^30+1) (2^30) 0=Bool.true := by unfold gridSupported grid; decide +kernel
#axiomsClean all_masks_false_at_two31
#axiomsClean active_tile_empty_at_two31
#axiomsClean grid_two31_rejected
#axiomsClean wrapped_negative_within_launch_domain
#axiomsClean repeated_lane_within_launch_domain
#guard_msgs (drop info) in
#auditModuleAxioms
end SinComputationNative
