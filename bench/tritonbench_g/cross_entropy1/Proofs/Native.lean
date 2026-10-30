import bench.tritonbench_g.cross_entropy1.Kernels
import bench.tritonbench_g.cross_entropy1.Proofs.NativeSupport

namespace VeriTile.Bench.Repairs.CrossEntropy1.Native
open VeriTile.Triton
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native (nat64_signed widen32_exact)
abbrev Buffer := VeriTile.Bench.Repairs.CrossEntropy1.TensorModel.Native.Buffer

/-- Successful ordinary CUDA setup limits each grid dimension. The installed
grid-product compatibility issue remains separate; no host size cap is added. -/
structure Supported (rows blocks width cols : Nat) : Prop where
  rowsLimit : rows ≤ 2^31-1
  blocksLimit : blocks ≤ 65535
  widthPositive : 0<width
  widthLimit : width ≤ 65536
  colsRepresentable : cols<2^63

def widened (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64
def column (width block lane : Nat) : BitVec 64 :=
  widened block * BitVec.ofNat 64 width + widened lane
def columnMask (cols width block lane : Nat) : Bool :=
  (column width block lane).slt (BitVec.ofNat 64 cols)
def element (base row stride width block lane : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + widened row * BitVec.ofNat 64 stride + column width block lane
def scalarOutput (base rows row block : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + widened block * BitVec.ofNat 64 rows + widened row
def address (base elementBytes : Nat) (index : BitVec 64) : BitVec 64 :=
  BitVec.ofNat 64 base + index * BitVec.ofNat 64 elementBytes

theorem column_bounds (rows blocks width cols block lane : Nat)
    (api : Supported rows blocks width cols) (hb : block<blocks) (hl : lane<width) :
    block<2^31 ∧ lane<2^31 ∧ block*width+lane<2^32 ∧ (block+1)*width<2^33 := by
  have blocksBound := api.blocksLimit
  have widthBound := api.widthLimit
  have product := Nat.mul_le_mul_left block widthBound
  have upper := Nat.mul_le_mul_left (block+1) widthBound
  norm_num at blocksBound widthBound ⊢
  omega

theorem column_word (rows blocks width cols block lane : Nat)
    (api : Supported rows blocks width cols) (hb : block<blocks) (hl : lane<width) :
    column width block lane = BitVec.ofNat 64 (block*width+lane) := by
  have h := column_bounds rows blocks width cols block lane api hb hl
  simp [column,widened,widen32_exact block h.1,widen32_exact lane h.2.1,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

theorem source_mask (rows blocks width cols block lane : Nat)
    (api : Supported rows blocks width cols) (hb : block<blocks) (hl : lane<width) :
    (column width block lane).toInt = (block*width+lane : Nat) ∧
    columnMask cols width block lane = decide (block*width+lane<cols) := by
  have h := column_bounds rows blocks width cols block lane api hb hl
  have col := column_word rows blocks width cols block lane api hb hl
  have exactCol : (column width block lane).toInt = (block*width+lane : Nat) := by
    rw [col,nat64_signed _ (by omega)]
  refine ⟨exactCol,?_⟩
  rw [columnMask,BitVec.slt_eq_decide,exactCol,nat64_signed cols api.colsRepresentable]
  simp only [Int.ofNat_lt]

theorem element_word (base row stride width block lane rows blocks cols : Nat)
    (api : Supported rows blocks width cols) (hr : row<rows) (hb : block<blocks) (hl : lane<width) :
    element base row stride width block lane =
      BitVec.ofNat 64 (base+row*stride+(block*width+lane)) := by
  have rowBound := api.rowsLimit
  have rowSmall : row<2^31 := by omega
  simp [element,column_word rows blocks width cols block lane api hb hl,
    widened,widen32_exact row rowSmall,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

/-- Every active FP32 or int64 memory observation uses its actual allocation
base and byte capacity. Only the observed element must fit, including views
whose backing allocation has a partial trailing word. -/
theorem backed_address (buffer : Buffer) (index elementBytes : Nat)
    (positive : 0<elementBytes) (backed : elementBytes*(index+1)≤buffer.capacity)
    (addressSpace : buffer.base+buffer.capacity≤2^64) :
    (address buffer.base elementBytes (BitVec.ofNat 64 index)).toNat =
        buffer.base+elementBytes*index ∧
    buffer.base+elementBytes*index+elementBytes≤buffer.base+buffer.capacity ∧
    buffer.base+elementBytes*index+elementBytes≤2^64 := by
  have bound : buffer.base+elementBytes*index<2^64 := by
    rw [Nat.mul_add, Nat.mul_one] at backed
    omega
  refine ⟨?_,?_,?_⟩
  · simp [address,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,
      BitVec.toNat_ofNat,Nat.mul_comm index elementBytes]
    norm_num at bound
    exact bound
  · rw [Nat.mul_add,Nat.mul_one] at backed
    omega
  · rw [Nat.mul_add,Nat.mul_one] at backed
    omega

theorem fp32_index_signed (buffer : Buffer) (index : Nat)
    (backed : 4*(index+1)≤buffer.capacity) (addressSpace : buffer.base+buffer.capacity≤2^64) :
    (BitVec.ofNat 64 index).toInt = (index : Int) := by
  apply nat64_signed
  norm_num at addressSpace ⊢
  omega

/-- Nonoverlapping logical rows have unique physical element addresses,
including arbitrary row gaps and a common nonzero storage offset. -/
theorem row_address_injective (base stride cols row otherRow col otherCol : Nat)
    (strideBound : cols≤stride) (hc : col<cols) (hd : otherCol<cols)
    (same : base+row*stride+col=base+otherRow*stride+otherCol) :
    row=otherRow ∧ col=otherCol := by
  rcases lt_trichotomy row otherRow with h|h|h
  · have gap := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt h)
    simp only [Nat.succ_mul] at gap
    omega
  · subst otherRow
    exact ⟨rfl,by omega⟩
  · have gap := Nat.mul_le_mul_right stride (Nat.succ_le_of_lt h)
    simp only [Nat.succ_mul] at gap
    omega

theorem tile_lane_unique (width block otherBlock lane otherLane : Nat)
    (hl : lane<width) (ho : otherLane<width)
    (same : block*width+lane=otherBlock*width+otherLane) :
    block=otherBlock ∧ lane=otherLane := by
  exact row_address_injective 0 width width block otherBlock lane otherLane
    (Nat.le_refl _) hl ho (by simpa using same)

theorem scalar_output_word (base rows row block blocks width cols : Nat)
    (api : Supported rows blocks width cols) (hr : row<rows) (hb : block<blocks) :
    scalarOutput base rows row block=BitVec.ofNat 64 (base+block*rows+row) := by
  have rowBound := api.rowsLimit
  have blockBound := api.blocksLimit
  have rs : row<2^31 := by omega
  have bs : block<2^31 := by norm_num at blockBound ⊢; omega
  simp [scalarOutput,widened,widen32_exact row rs,widen32_exact block bs,
    BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat]

/-- Masked lanes may form pointers but never observe them. Even those integer
intermediates remain signed-64 representable using an actually backed row start. -/
theorem all_lane_index_signed (buffer : Buffer) (base row stride block lane rows blocks width cols : Nat)
    (api : Supported rows blocks width cols) (hb : block<blocks) (hl : lane<width)
    (rowBacked : 4*(base+row*stride+1)≤buffer.capacity)
    (addressSpace : buffer.base+buffer.capacity≤2^64) :
    (BitVec.ofNat 64 (base+row*stride+(block*width+lane))).toInt=
      (base+row*stride+(block*width+lane) : Nat) := by
  apply nat64_signed
  have h := (column_bounds rows blocks width cols block lane api hb hl).2.2.1
  norm_num at h addressSpace ⊢
  omega

def shiftedLabel (label start : Int) : BitVec 64 := BitVec.ofInt 64 label-BitVec.ofInt 64 start

theorem shifted_label (label start total : Int) (size : 0<total ∧ total<2^63)
    (target : 0≤label ∧ label<total) (rankStart : 0≤start ∧ start<total) :
    (shiftedLabel label start).toInt=label-start := by
  have word : shiftedLabel label start=BitVec.ofInt 64 (label-start) := by
    simp only [shiftedLabel,sub_eq_add_neg,BitVec.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg]
  rw [word]
  apply BitVec.toInt_ofInt_eq_self
  all_goals norm_num at size ⊢
  all_goals omega

/-- Ignored labels are replaced before subtraction, including INT64_MIN. -/
theorem ignored_label_shift (label ignored start : Int) (same : label=ignored) :
    (shiftedLabel (if label=ignored then start else label) start).toInt=0 := by
  simp [same,shiftedLabel]

#axiomsClean scalar_output_word
#axiomsClean all_lane_index_signed
#axiomsClean shifted_label
#axiomsClean ignored_label_shift
#axiomsClean source_mask
#axiomsClean element_word
#axiomsClean backed_address
#axiomsClean fp32_index_signed
#axiomsClean row_address_injective
#axiomsClean tile_lane_unique
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropy1.Native
