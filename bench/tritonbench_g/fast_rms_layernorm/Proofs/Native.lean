import bench.tritonbench_g.fast_rms_layernorm.Kernels
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.Native
open VeriTile.Triton
namespace Shared
abbrev Buffer := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.Buffer
abbrev signedLimit := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.signedLimit
abbrev extent_bound := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.extent_bound
abbrev signed64 := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.signed64
abbrev widen32 := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.widen32
abbrev pointer_word := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.pointer_word
abbrev byte_span := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.byte_span
end Shared

structure Allocation (cells : Nat) where
  storageId : Nat
  buffer : Shared.Buffer
  holds : cells≤buffer.extent
  fp32Bytes : buffer.bytes=4

def offset (h : Call) (row col : Nat) := row*h.cols+col

theorem offset_bound (h : Call) (row col : Nat) (hr : row<h.rows) (hc : col<h.cols) :
    offset h row col<h.cells := by
  unfold offset Call.cells
  calc
    row*h.cols+col<row*h.cols+h.cols := Nat.add_lt_add_left hc _
    _ = (row+1)*h.cols := by ring
    _ ≤ h.rows*h.cols := Nat.mul_le_mul_right h.cols (Nat.succ_le_of_lt hr)

theorem access_inside (h : Call) (a : Allocation h.cells) (row col : Nat)
    (hr : row<h.rows) (hc : col<h.cols) : offset h row col<a.buffer.extent :=
  (offset_bound h row col hr hc).trans_le a.holds

theorem columns_fit (h : Call) : h.cols<2147483648 := by
  have c:=h.covered
  have b:=h.blockLimit
  omega

theorem lanes_fit (h : Call) (lane : Fin h.block) : lane.val<2147483648 := by
  have b:=h.blockLimit
  have l:=lane.isLt
  omega

theorem row_bound (h : Call) (a : Allocation h.cells) : h.rows<Shared.signedLimit := by
  have cap:=a.holds.trans_lt (Shared.extent_bound a.buffer)
  have small : h.rows≤h.cells := Nat.le_mul_of_pos_right _ h.positive
  exact small.trans_lt cap

theorem accessed_word (h : Call) (a : Allocation h.cells) (row col : Nat)
    (hr : row<h.rows) (hc : col<h.cols) (pid32 : row<2147483648) :
    ((BitVec.ofInt 64 (BitVec.ofNat 32 row).toInt)*BitVec.ofNat 64 h.cols+BitVec.ofNat 64 col).toInt=
      (offset h row col : Int) := by
  rw [Shared.widen32 row pid32,←BitVec.ofNat_mul,←BitVec.ofNat_add]
  exact Shared.signed64 _ ((access_inside h a row col hr hc).trans (Shared.extent_bound a.buffer))

#axiomsClean accessed_word
#axiomsClean row_bound
end VeriTile.Bench.Repairs.FastRmsLayernorm.Native
