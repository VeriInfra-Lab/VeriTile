import bench.tritonbench_g.fast_ce_loss.Proofs.KernelContract

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelAccess
open VeriTile.Triton FP.Structural KernelSource KernelContract KernelContract.Storage
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native (nat64_signed widen32_exact)
noncomputable section
set_option maxHeartbeats 4000000

/-- Native pointer arithmetic uses the actual argument data pointer, already
including Tensor.storageOffset. No allocation-base alignment is assumed. -/
def pointer64 (width pointer offset : Nat) : BitVec 64 :=
  BitVec.ofNat 64 pointer+BitVec.ofNat 64 width*BitVec.ofNat 64 offset

def rowOffset64 (row stride : Nat) : BitVec 64 :=
  (BitVec.ofNat 32 row).signExtend 64 * BitVec.ofNat 64 stride

def column64 (block width lane : Nat) : BitVec 64 :=
  (BitVec.ofNat 32 block).signExtend 64 * BitVec.ofNat 64 width + BitVec.ofNat 64 lane

 theorem pointer_bits (width pointer offset : Nat) :
    pointer64 width pointer offset=BitVec.ofNat 64 (pointer+width*offset) := by
  simp only [pointer64,← BitVec.ofNat_mul,← BitVec.ofNat_add]

 theorem row_bits (row stride : Nat) (hr : row<2^31) :
    rowOffset64 row stride=BitVec.ofNat 64 (row*stride) := by
  simp only [rowOffset64,widen32_exact row hr,← BitVec.ofNat_mul]

 theorem column_bits (block width lane : Nat) (hb : block<2^31) :
    column64 block width lane=BitVec.ofNat 64 (block*width+lane) := by
  simp only [column64,widen32_exact block hb,← BitVec.ofNat_mul,← BitVec.ofNat_add]

 theorem argument_backed (alloc : RegionName → Allocation) (t : Tensor)
    (width pointer i offset : Nat) (storage : TensorStorage alloc t width pointer)
    (hi : i<t.numel) (index : t.index i=t.storageOffset+offset) :
    pointer+width*(offset+1)≤(alloc t.region).base+(alloc t.region).capacity := by
  have backed := storage.backed i hi
  rw [storage.pointerEq]
  rw [index] at backed
  nlinarith

 theorem argument_pointer (alloc : RegionName → Allocation) (t : Tensor)
    (width pointer i offset : Nat) (storage : TensorStorage alloc t width pointer)
    (positive : 0<width) (hi : i<t.numel) (index : t.index i=t.storageOffset+offset) :
    (pointer64 width pointer offset).toNat=(alloc t.region).base+width*t.index i := by
  have backed := argument_backed alloc t width pointer i offset storage hi index
  have bound : pointer+width*offset<2^64 := by
    have space := storage.addressSpace
    nlinarith
  rw [pointer_bits,BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound,storage.pointerEq,index]
  ring

/-- Backing bounds imply representability of every active element offset.
Unused singleton strides themselves need not equal a contiguous stride. -/
 theorem active_fp32_offset (alloc : RegionName → Allocation) (t : Tensor)
    (pointer i offset : Nat) (storage : TensorStorage alloc t 4 pointer)
    (hi : i<t.numel) (index : t.index i=t.storageOffset+offset) :
    offset<2^63 := by
  have backed := storage.backed i hi
  have space := storage.addressSpace
  rw [index] at backed
  norm_num at space ⊢
  omega

 theorem active_int64_offset (alloc : RegionName → Allocation) (t : Tensor)
    (pointer i offset : Nat) (storage : TensorStorage alloc t 8 pointer)
    (hi : i<t.numel) (index : t.index i=t.storageOffset+offset) :
    offset<2^63 := by
  have backed := storage.backed i hi
  have space := storage.addressSpace
  rw [index] at backed
  norm_num at space ⊢
  omega

 theorem matrix_pointer (alloc : RegionName → Allocation) (t : Tensor)
    (region : RegionName) (rows cols stride pointer row col : Nat)
    (view : MatrixView t region rows cols stride) (storage : TensorStorage alloc t 4 pointer)
    (hr : row<rows) (hc : col<cols) :
    (pointer64 4 pointer (row*stride+col)).toNat=
      (alloc region).base+4*t.index (row*cols+col) ∧
    (BitVec.ofNat 64 (row*stride+col)).toInt=(row*stride+col : Nat) := by
  have index : t.index (row*cols+col)=t.storageOffset+(row*stride+col) := by
    simpa only [Nat.add_assoc] using matrix_index t region rows cols stride row col view hr hc
  have hi := matrix_numel t region rows cols stride row col view hr hc
  constructor
  · simpa only [view.regionEq] using argument_pointer alloc t 4 pointer _ _ storage (by omega) hi index
  · exact nat64_signed _ (active_fp32_offset alloc t pointer _ _ storage hi index)

 theorem flat_pointer (alloc : RegionName → Allocation) (t : Tensor)
    (region : RegionName) (count width pointer i : Nat)
    (view : FlatView t region count) (storage : TensorStorage alloc t width pointer)
    (positive : 0<width) (hi : i<count) :
    (pointer64 width pointer i).toNat=(alloc region).base+width*t.index i := by
  simpa only [view.regionEq] using argument_pointer alloc t width pointer i i storage positive
    (by simpa only [view.numel] using hi) (view.index i hi)

 theorem logits_words (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (t : Tensor) (region : RegionName)
    (rows cols stride pointer row col : Nat) (view : MatrixView t region rows cols stride)
    (binding : FloatArgument alloc bytes words t pointer) (hr : row<rows) (hc : col<cols) :
    (words.mem region (row*stride+col)).read .real=
      wordAt (bytes region) (4*t.index (row*cols+col)) := by
  have cells := binding.cells (row*cols+col) (matrix_numel t region rows cols stride row col view hr hc)
  simpa only [tensorValue,matrix_index t region rows cols stride row col view hr hc,
    view.regionEq,Nat.add_assoc,Nat.add_sub_cancel_left] using cells

 theorem labels_words (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte)
    (words : State Word) (t : Tensor) (region : RegionName) (count pointer row : Nat)
    (view : FlatView t region count) (binding : LabelArgument alloc bytes words t pointer) (hr : row<count) :
    (words.mem region row).read .int=intAt (bytes region) (8*t.index row) := by
  have cells := binding.cells row (by simpa only [view.numel] using hr)
  simpa only [tensorLabel,view.index row hr,view.regionEq,Nat.add_sub_cancel_left] using cells

 theorem forward_label (p : Parameters) (views : ForwardViews p chunks)
    (alloc : RegionName → Allocation) (bytes : RegionName → Nat → Byte) (words : State Word)
    (inputs : ForwardInputs p views alloc bytes words) (row : Nat) (hr : row<p.rows)
    (pid : words.pids 0=row) : label p words = -100 ∨ 0≤label p words := by
  have h := inputs.labelsValid row hr
  have same : tensorLabel words views.labels row=label p words := by
    simp only [tensorLabel,views.labelsView.index row hr,views.labelsView.regionEq,
      Nat.add_sub_cancel_left,label,pid]
  rw [same] at h
  exact h.imp_right And.left

#axiomsClean argument_pointer
#axiomsClean active_fp32_offset
#axiomsClean active_int64_offset
#axiomsClean matrix_pointer
#axiomsClean flat_pointer
#axiomsClean logits_words
#axiomsClean labels_words
#axiomsClean forward_label
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.KernelAccess
