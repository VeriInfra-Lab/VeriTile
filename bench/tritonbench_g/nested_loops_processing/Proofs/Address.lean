import bench.tritonbench_g.nested_loops_processing.Proofs.Math
import bench.tritonbench_g.nested_loops_processing.Proofs.Shared.Address

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing
open VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address (Dense count_bound address_bytes address_apart)
open VeriTile.Bench.Repairs.LayerNormFwd.Address (signed64 signed64_exact)
noncomputable section

theorem columns_bounded (buffer : Dense (rows*cols)) (positive : 0<rows*cols) :
    0<cols ∧ cols≤2^62 := by
  have count:=count_bound buffer
  have hr:0<rows:=Nat.pos_of_mul_pos_right positive
  have le:cols≤rows*cols:=Nat.le_mul_of_pos_left cols hr
  exact ⟨Nat.pos_of_mul_pos_left positive,le_trans le count⟩

/-- Bounds include the final offset increments after the last store; no
pre-widened product is used. The allocation, not a new public cap, supplies
the dimension bound. -/
theorem offset_intermediates (buffer : Dense (rows*cols)) (positive : 0<rows*cols)
    (shift : Nat) (bound : shift≤48) (lane : Lane) :
    lane.1.val*cols≤2^62 ∧ address cols shift lane<2^63 ∧
    signed64 (lane.1.val*cols)=(lane.1.val*cols:Nat) ∧
    signed64 (address cols shift lane)=(address cols shift lane:Nat) := by
  have columns:=columns_bounded buffer positive
  have rowBound:lane.1.val≤1:=by omega
  have colBound:lane.2.val≤1:=by omega
  have mulBound:lane.1.val*cols≤cols:=by
    simpa using Nat.mul_le_mul_right cols rowBound
  have offsetBound:address cols shift lane<2^63:=by
    unfold address
    omega
  exact ⟨le_trans mulBound columns.2,offsetBound,
    signed64_exact (by omega),signed64_exact offsetBound⟩

theorem clamped_bound (positive : 0<count) (raw : Nat) : clamp count raw<count := by
  have h:=Nat.min_le_right raw (count-1)
  unfold clamp
  omega

theorem clamped_active (active : raw<count) : clamp count raw=raw := by
  unfold clamp
  exact Nat.min_eq_left (by omega)

/-- The runtime numel cast and the subtraction used before every pointer
formation are exact signed-int64 operations on every successful allocation. -/
theorem count_integer_operations (buffer : Dense count) (positive : 0<count) :
    count<2^63 ∧ signed64 count=(count:Nat) ∧
    0≤(count:Int)-1 ∧ (count:Int)-1<2^63 ∧
    ∀ raw,signed64 (clamp count raw)=(clamp count raw:Nat) := by
  have bound:=count_bound buffer
  have small:count<2^63:=by omega
  refine ⟨small,signed64_exact small,by omega,by omega,?_⟩
  intro raw
  exact signed64_exact (lt_trans (clamped_bound positive raw) small)

theorem all_pointer_bytes (buffer : Dense count) (positive : 0<count)
    (raw byte : Nat) (small : byte<4) :
    buffer.contains (buffer.address (clamp count raw)+byte) ∧
    buffer.address (clamp count raw)+byte<2^64 :=
  address_bytes buffer _ (clamped_bound positive raw) byte small

/-- All actual input shifts are from the unchanged cached-load schedule. -/
theorem all_load_offsets (buffer : Dense (rows*cols)) (positive : 0<rows*cols)
    (t : Nat) (store : t<24) (lane : Lane) :
    address cols (VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.CopyMath.inputShift t) lane<2^63 :=
  (offset_intermediates buffer positive _
    (le_trans (VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.CopyMath.inputShift_bound store) (by omega)) lane).2.1

theorem all_store_offsets (buffer : Dense (rows*cols)) (positive : 0<rows*cols)
    (t : Nat) (store : t<24) (lane : Lane) : address cols (2*t) lane<2^63 :=
  (offset_intermediates buffer positive _ (by omega) lane).2.1

#axiomsClean columns_bounded
#axiomsClean offset_intermediates
#axiomsClean all_pointer_bytes
#axiomsClean count_integer_operations
#axiomsClean all_load_offsets
#axiomsClean all_store_offsets
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing
