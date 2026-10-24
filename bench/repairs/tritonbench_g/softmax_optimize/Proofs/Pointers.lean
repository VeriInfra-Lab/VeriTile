import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Contract
import bench.repairs.tritonbench_g.softmax_optimize.Proofs.Native
namespace VeriTile.Bench.Repairs.SoftmaxOptimize.Pointers

/-- Nonempty input span is supplied by ordinary Tensor storage binding. -/
def allocation (x : TensorMetadata) (rank : x.shape.length=2)
    (span : x.dataPtr.toNat+4*x.numel≤2^64) : Native.Allocation (rows x) (columns x) where
  base := x.dataPtr.toNat
  bytes := 4*x.numel
  native_span := span
  holds_words := by rw [count_two x rank]

theorem pid_compatible (x : TensorMetadata) (h : InputContract x) (live : x.numel≠0) :
    Native.PIDCompatible (rows x) := by
  intro row
  exact lt_trans row.isLt (dims_positive x h live).2.2.1

theorem scalar_arguments (x : TensorMetadata) (h : InputContract x) (live : x.numel≠0) :
    (BitVec.ofNat 64 (rows x)).toInt=(rows x:Int) ∧
    (BitVec.ofNat 64 (columns x)).toInt=(columns x:Int) := by
  have b := dims_positive x h live
  constructor <;> apply Native.signed_exact <;> omega

theorem arange_exact (N : Nat) (lane : Fin (Config.tile N)) :
    (BitVec.ofNat 32 lane.val).toInt=(lane.val:Int) := by
  have b := Config.tile_le_cap N
  apply Native.signed_exact
  have l := lane.isLt
  omega

/-- Padded lanes and carried loop bases fit signed i64 before pointer scaling.
The source's unmasked full-tile loads are separately bounded by full_tile_active. -/
theorem padded_element (x : TensorMetadata) (h : InputContract x) (live : x.numel≠0)
    (row : Fin (rows x)) (start : Nat) (hs : start≤columns x+Config.tile (columns x))
    (lane : Fin (Config.tile (columns x))) :
    Native.nativeElement (columns x) row (start+lane.val)=
      (Native.element (columns x) row (start+lane.val):Int) := by
  have db := dims_positive x h live
  have cb := Config.tile_le_cap (columns x)
  have total := h.2.2.2.2.2.2.1
  rw [count_two x h.2.2.2.1] at total
  have rn : row.val*columns x<rows x*columns x := Nat.mul_lt_mul_of_pos_right row.isLt db.2.1
  unfold Native.nativeElement
  rw [Native.pid32_exact (pid_compatible x h live) row,BitVec.ofInt_natCast,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  apply Native.signed_exact
  have lb := lane.isLt
  omega

theorem previous_intermediates (N : Nat) (hn : 0<N) (nb : N<2^31) :
    let B := Config.tile N
    N+B-1<2^63 ∧ (N+B-1)/B*B<2^63 ∧
      B≤(N+B-1)/B*B ∧ Config.previous N B<N := by
  have bp := Config.tile_positive N
  have bc := Config.tile_le_cap N
  have div := Nat.div_mul_le_self (N+Config.tile N-1) (Config.tile N)
  have prod := Nat.mul_le_mul_right (Config.tile N)
    (Nat.div_pos (by omega : Config.tile N≤N+Config.tile N-1) bp)
  have tail := (Config.final_tail_nonempty_once hn bp).1
  dsimp only
  omega

theorem input_bytes (x : TensorMetadata) (h : InputContract x)
    (span : x.dataPtr.toNat+4*x.numel≤2^64)
    (row : Fin (rows x)) (column : Fin (columns x)) (byte : Fin 4) :
    Native.nativeByte x.dataPtr.toNat (columns x) row column byte=
      x.dataPtr.toNat+4*(row.val*columns x+column.val)+byte.val ∧
    x.dataPtr.toNat+4*(row.val*columns x+column.val)+byte.val<x.dataPtr.toNat+4*x.numel ∧
    (x.dataPtr.toNat+4*(row.val*columns x+column.val))%4=0 := by
  let a := allocation x h.2.2.2.1 span
  have act := Native.active_element_bound row column
  have count := count_two x h.2.2.2.1
  have alignment := h.2.2.2.2.2.2.2 (by omega)
  exact ⟨Native.native_byte_exact a row column byte,(Native.active_byte_bound a row column byte).1,by omega⟩

#axiomsClean pid_compatible
#axiomsClean scalar_arguments
#axiomsClean arange_exact
#axiomsClean padded_element
#axiomsClean previous_intermediates
#axiomsClean input_bytes
end VeriTile.Bench.Repairs.SoftmaxOptimize.Pointers
