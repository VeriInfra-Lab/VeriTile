import bench.repairs.tritonbench_g.decay_cumsum.Proofs.Geometry
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.NativeIndex

/-! Fixed-width facts reuse the frozen candidate's generic allocation/BitVec
lemmas. Each buffer is an actual same-call packed allocation; the tail is an
interior view of that allocation. ABI facts describe executed program IDs and
the ordinary signed32 arange intrinsic, not an additional tensor-shape filter.
-/
namespace VeriTile.Bench.Repairs.DecayCumsum.NativeIndex
open VeriTile.Triton
namespace Shared
abbrev Buffer := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.Buffer
abbrev signedLimit := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.signedLimit
abbrev signed64 := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.signed64
abbrev widen32 := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.widen32
abbrev extent_bound := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.extent_bound
abbrev pointer_word := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.pointer_word
abbrev byte_span := VeriTile.Bench.Repairs.ChunkCumsumVector.NativeIndex.byte_span
end Shared

structure Allocation (h : Call) where
  buffer : Shared.Buffer
  holds : h.cells≤buffer.extent
  fp32Bytes : buffer.bytes=4

structure PID (h : Call) (p : Phase) extends Geometry.Program h p where
  feature32 : feature<2147483648
  chunk32 : chunk<2147483648
  row32 : row<2147483648

def widened (n : Nat) : BitVec 64 := BitVec.ofInt 64 (BitVec.ofNat 32 n).toInt

def relative (h : Call) (p : Phase) (prog : PID h p) (step lane : Nat) : Nat :=
  prog.row*h.headStride+prog.chunk*p.span*h.width+prog.feature*h.BK+lane+step*h.width

def offsetWord (h : Call) (p : Phase) (prog : PID h p) (step lane : Nat) : BitVec 64 :=
  widened prog.row*BitVec.ofNat 64 h.headStride+
    widened prog.chunk*BitVec.ofNat 64 p.span*BitVec.ofNat 64 h.width+
    widened prog.feature*BitVec.ofNat 64 h.BK+widened lane+
    BitVec.ofNat 64 step*BitVec.ofNat 64 h.width

theorem offset_word (h : Call) (p : Phase) (prog : PID h p)
    (step lane : Nat) (lane32 : lane<2147483648) :
    offsetWord h p prog step lane=BitVec.ofNat 64 (relative h p prog step lane) := by
  simp only [offsetWord,widened,Shared.widen32 _ prog.row32,Shared.widen32 _ prog.chunk32,
    Shared.widen32 _ prog.feature32,Shared.widen32 _ lane32,
    ←BitVec.ofNat_mul,←BitVec.ofNat_add,relative]

theorem shifted_offset (h : Call) (p : Phase) (prog : PID h p) (step lane : Nat) :
    p.start*h.width+relative h p prog step lane=
      Geometry.offset h p prog.toProgram step lane := by
  unfold relative Geometry.offset Geometry.time Geometry.column
  ring

theorem source_addresses (h : Call) (p : Phase) (hp : p∈phases h)
    (allocation : Allocation h) (prog : PID h p) (step : Fin p.span) (lane : Fin h.BK)
    (lane32 : lane.val<2147483648) :
    (offsetWord h p prog step.val lane.val).toInt=(relative h p prog step.val lane.val:Int) ∧
    (BitVec.ofNat 64 (allocation.buffer.base+p.start*h.width*allocation.buffer.bytes)+
      offsetWord h p prog step.val lane.val*BitVec.ofNat 64 allocation.buffer.bytes).toNat=
      allocation.buffer.base+Geometry.offset h p prog.toProgram step.val lane.val*allocation.buffer.bytes ∧
    allocation.buffer.base+Geometry.offset h p prog.toProgram step.val lane.val*allocation.buffer.bytes+
      allocation.buffer.bytes≤18446744073709551616 := by
  have inside := (Geometry.offset_inside h p hp prog.toProgram step lane).trans_le allocation.holds
  have shifted := shifted_offset h p prog step.val lane.val
  have relBound : relative h p prog step.val lane.val<Shared.signedLimit := by
    have bound : Geometry.offset h p prog.toProgram step.val lane.val<Shared.signedLimit :=
      inside.trans (Shared.extent_bound allocation.buffer)
    omega
  rw [offset_word h p prog step.val lane.val lane32]
  refine ⟨Shared.signed64 _ relBound,?_,Shared.byte_span allocation.buffer _ inside⟩
  have wordEq : BitVec.ofNat 64 (allocation.buffer.base+p.start*h.width*allocation.buffer.bytes)+
      BitVec.ofNat 64 (relative h p prog step.val lane.val)*BitVec.ofNat 64 allocation.buffer.bytes=
      BitVec.ofNat 64 allocation.buffer.base+
      BitVec.ofNat 64 (Geometry.offset h p prog.toProgram step.val lane.val)*BitVec.ofNat 64 allocation.buffer.bytes := by
    simp only [←BitVec.ofNat_mul,←BitVec.ofNat_add]
    congr 1
    rw [←shifted]
    ring
  rw [wordEq]
  exact Shared.pointer_word allocation.buffer _ inside

theorem loop_bound (h : Call) (p : Phase) (hp : p∈phases h)
    (allocation : Allocation h) (prog : PID h p) (featurePositive : 0<h.DK) :
    0<p.span ∧ p.span<Shared.signedLimit ∧
    (BitVec.ofNat 64 p.span).toInt=(p.span:Int) ∧
    (BitVec.ofNat 64 (p.span-1)).toInt=(p.span-1:Nat) := by
  have bounds := Geometry.phase_positive h p hp
  have timeBound : p.span≤h.T := by
    have hc := Nat.le_mul_of_pos_left p.span bounds.1
    omega
  have width := Geometry.width_positive h featurePositive
  have rows := Nat.zero_lt_of_lt prog.rowInside
  have total : h.T≤h.cells := by
    unfold Call.cells Call.headStride
    exact (Nat.le_mul_of_pos_right h.T width).trans
      (Nat.le_mul_of_pos_left (h.T*h.width) rows)
  have limit := allocation.holds.trans_lt (Shared.extent_bound allocation.buffer)
  have bound := (timeBound.trans total).trans_lt limit
  exact ⟨bounds.2.1,bound,Shared.signed64 _ bound,
    Shared.signed64 _ ((Nat.sub_le _ _).trans_lt bound)⟩

#axiomsClean source_addresses
#axiomsClean loop_bound
end VeriTile.Bench.Repairs.DecayCumsum.NativeIndex
