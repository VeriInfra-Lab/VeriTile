import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.HostPrepared
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.Integer

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.PreparedBounds
open HostSnapshots HostPrepared

/-- Ordinary allocation facts for the three returned floating clones. These
describe allocated tensor storage, not a newly restricted numerical domain. -/
structure Storage (shape : Shape) where
  input : Allocation
  weight : Allocation
  output : Allocation
  inputElements : input.elements = shape.tokens*shape.inner
  weightElements : weight.elements = shape.adapters*shape.columns*shape.inner
  outputElements : output.elements = shape.outputRows*shape.outputColumns
  inputWidth : 2 ≤ input.bytesPerElement
  weightWidth : 2 ≤ weight.bytesPerElement
  outputWidth : 2 ≤ output.bytesPerElement

def matrixView (allocation : Allocation) (rows columns rowStride columnStride : Nat)
    (size : allocation.elements = rows*columns)
    (layout : GridStride.Contiguous rows columns rowStride columnStride) : MatrixView where
  allocation := allocation
  rows := rows
  columns := columns
  offset := 0
  rowStride := rowStride
  columnStride := columnStride
  coordinateBound := by
    intro r c hr hc
    simpa only [Nat.zero_add,GridStride.coordinate_linear layout hr hc,size] using linear_lt hr hc

def weightView (allocation : Allocation) (adapters columns inner adapterStride columnStride innerStride : Nat)
    (size : allocation.elements = adapters*columns*inner)
    (layout : SnapshotLayout.WeightContiguous adapters columns inner adapterStride columnStride innerStride) : WeightView where
  allocation := allocation
  adapters := adapters
  columns := columns
  inner := inner
  offset := 0
  adapterStride := adapterStride
  columnStride := columnStride
  innerStride := innerStride
  coordinateBound := by
    intro a c k ha hc hk
    rw [Nat.zero_add,Nat.mul_comm a adapterStride,SnapshotLayout.weight_linear layout ha hc hk,size]
    exact linear_lt (linear_lt ha hc) hk

/-- Every source gather/scatter pointer expression is exact in i64, with the
actual clone strides (including ignored singleton-stride freedom). -/
theorem source_pointer_words {α : Host.Role → Type}
    (abi : SourceABI) (shape : Shape) (batches : Nat)
    (entry : (role : Host.Role) → View α role) (caller : List Window)
    (domain : Domain abi shape batches entry) (p : Prepared entry caller) (storage : Storage shape)
    (segment : Segment) (fits : SegmentFits shape segment) (live : segment.active)
    (positive : 0 < shape.columns) (r c k : Nat) (hk : k < shape.inner) :
    (Integer.i64 segment.start*Integer.i64 (strides p).inputRow +
      Integer.i64 (r%segment.length)*Integer.i64 (strides p).inputRow +
      Integer.i64 k*Integer.i64 (strides p).inputInner).toInt =
      (segment.start*(strides p).inputRow+(r%segment.length)*(strides p).inputRow+k*(strides p).inputInner : Nat) ∧
    (Integer.i64 (strides p).weightAdapter*Integer.i64 (segment.adapter.getD 0)+
      Integer.i64 k*Integer.i64 (strides p).weightInner+
      Integer.i64 (c%shape.columns)*Integer.i64 (strides p).weightColumn).toInt =
      ((strides p).weightAdapter*segment.adapter.getD 0+k*(strides p).weightInner+(c%shape.columns)*(strides p).weightColumn : Nat) ∧
    (Integer.i64 (safeRow segment r)*Integer.i64 (strides p).outputRow+
      Integer.i64 (safeColumn shape c)*Integer.i64 (strides p).outputColumn).toInt =
      (safeRow segment r*(strides p).outputRow+safeColumn shape c*(strides p).outputColumn : Nat) := by
  obtain ⟨hx,hw,hy⟩ := layouts abi shape batches entry caller domain p
  have hrow := gathered_row_in_bounds shape segment fits live r
  have hcol := gathered_column_in_bounds shape positive c
  have ha : segment.adapter.getD 0 < shape.adapters := by
    cases he : segment.adapter with
    | none => simp [Segment.active,he] at live
    | some a => simpa [he] using fits.adapter a he live.1
  have bx := matrix_address_i64
    (matrixView storage.input shape.tokens shape.inner (strides p).inputRow (strides p).inputInner storage.inputElements hx)
    (segment.start+r%segment.length) k hrow hk
  have bw := weight_address_i64
    (weightView storage.weight shape.adapters shape.columns shape.inner (strides p).weightAdapter
      (strides p).weightColumn (strides p).weightInner storage.weightElements hw)
    (segment.adapter.getD 0) (c%shape.columns) k ha hcol hk
  have by' := matrix_address_i64
    (matrixView storage.output shape.outputRows shape.outputColumns (strides p).outputRow
      (strides p).outputColumn storage.outputElements hy)
    (safeRow segment r) (safeColumn shape c) (safeRow_lt shape segment fits live r)
    (safeColumn_lt shape domain.capacity positive c)
  simp only [matrixView,weightView,Nat.zero_add,Nat.add_mul] at bx bw by'
  refine ⟨?_,?_,?_⟩
  · simpa only [Integer.i64,←BitVec.ofNat_mul,←BitVec.ofNat_add] using Integer.i64_value bx
  · have bw' : (strides p).weightAdapter*segment.adapter.getD 0+k*(strides p).weightInner+
        (c%shape.columns)*(strides p).weightColumn < 2^63 := by
      simpa only [Nat.mul_comm (segment.adapter.getD 0) (strides p).weightAdapter] using bw
    simpa only [Integer.i64,←BitVec.ofNat_mul,←BitVec.ofNat_add] using Integer.i64_value bw'
  · simpa only [Integer.i64,←BitVec.ofNat_mul,←BitVec.ofNat_add] using Integer.i64_value by'

theorem dimensions_and_pid (shape : Shape) (storage : Storage shape) (segment : Segment)
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (capacity : shape.slice+shape.columns ≤ shape.outputColumns) (pid : Nat)
    (hp : pid < tiles segment.length*tiles shape.columns) :
    shape.outputRows < 2^62 ∧ shape.outputColumns < 2^62 ∧ shape.inner < 2^62 ∧
    shape.columns+32 < 2^63 ∧ shape.inner+16 < 2^63 ∧
    (Integer.i64 pid).toInt = (pid : Int) := by
  have bo := low_or_single_element_bound storage.output storage.outputWidth
  rw [storage.outputElements] at bo
  have bi := low_or_single_element_bound storage.input storage.inputWidth
  rw [storage.inputElements] at bi
  have rows := fits.output live
  have tokens := fits.input live
  have hr : 0 < shape.outputRows := by have := live.1; omega
  have hc : 0 < shape.outputColumns := by omega
  have ht : 0 < shape.tokens := by have := live.1; omega
  have br : shape.outputRows < 2^62 := by nlinarith
  have bc : shape.outputColumns < 2^62 := by nlinarith
  have bk : shape.inner < 2^62 := by nlinarith
  refine ⟨br,bc,bk,?_,?_,Integer.accepted_pid_i64 shape segment fits live positive capacity bo pid hp⟩ <;>
    norm_num at bc bk ⊢ <;> omega

#axiomsClean matrixView
#axiomsClean weightView
#axiomsClean source_pointer_words
#axiomsClean dimensions_and_pid
end VeriTile.Bench.Repairs.SgmvExpandSlice.PreparedBounds
