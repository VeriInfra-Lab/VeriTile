import bench.repairs.tritonbench_g.cross_entropy2.Kernels
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Native
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Storage

namespace VeriTile.Bench.Repairs.CrossEntropy2.Native
open VeriTile.Triton
open VeriTile.Bench.Repairs.CrossEntropy1.Native
open VeriTile.Bench.Repairs.CrossEntropy1.Physical
noncomputable section

/-- Int32 labels share the very same four-byte word with FP32 aliases. The
integer observation is signed; it is not a separate tagged allocation. -/
def Int32Read {α σ : Type} [Inhabited α] (encode : α → Word)
    (s : VeriTile.Bench.Repairs.CrossEntropy1.Host.World α σ) (r : RegionName) (i : Nat) : Prop :=
  (s.memory r i).read .int=(rawWord encode s r i).toInt

theorem int32_byte_coupling {α σ : Type} [Inhabited α] (layout : Layout σ) (encode : α → Word)
    (s : VeriTile.Bench.Repairs.CrossEntropy1.Host.World α σ) (r : RegionName) (i : Nat)
    (active : layout.active s.native r) (backed : 4*(i+1)≤(layout.buffer s.native r).capacity)
    (coupled : Int32Read encode s r i) :
    (s.memory r i).read .int=(rawWord encode s r i).toInt ∧
    ∀ j : Fin 4, bytes layout encode s (layout.buffer s.native r).device ((layout.buffer s.native r).base+4*i+j.val)=
      byteOf (rawWord encode s r i) j := by
  refine ⟨coupled,?_⟩
  intro j
  exact raw_bytes layout encode s r i j active (by have hj := j.isLt; omega)

def safeTarget (label start : Int) (cols : Nat) : Int :=
  if start≤label ∧ label<start+(cols : Int) then label else start

def safeShift (label start : Int) (cols : Nat) : BitVec 64 :=
  BitVec.ofInt 64 (safeTarget label start cols)-BitVec.ofInt 64 start

theorem int32_widen (word : BitVec 32) : (word.signExtend 64).toInt=word.toInt :=
  BitVec.toInt_signExtend_of_le (by decide)

/-- LLVM/PTX may commute the safe select with a modular subtraction. The
discarded subtraction has no `nsw` promise and cannot affect the selected word. -/
theorem select_after_subtraction (label start : Int) (cols : Nat) :
    (if start≤label ∧ label<start+(cols : Int) then
      BitVec.ofInt 64 label-BitVec.ofInt 64 start else 0)=safeShift label start cols := by
  by_cases inside : start≤label ∧ label<start+(cols : Int)
  · simp [safeShift,safeTarget,inside]
  · simp [safeShift,safeTarget,inside]

/-- Selection occurs before subtraction. Arbitrary int32 labels therefore
cannot overflow even when the class start is close to INT64_MAX. -/
theorem safe_shift_signed (label start : Int) (cols total : Nat)
    (startNonneg : 0≤start) (metadata : start+(cols : Int)≤(total : Int)) (totalBound : total<2^63) :
    (safeShift label start cols).toInt=
      if start≤label ∧ label<start+(cols : Int) then label-start else 0 := by
  by_cases owned : start≤label ∧ label<start+(cols : Int)
  · have word : safeShift label start cols=BitVec.ofInt 64 (label-start) := by
      simp [safeShift,safeTarget,owned,sub_eq_add_neg,BitVec.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg]
    rw [word,if_pos owned]
    apply BitVec.toInt_ofInt_eq_self
    all_goals norm_num at totalBound ⊢
    all_goals omega
  · simp [safeShift,safeTarget,owned]

theorem safe_target_comparison (label start : Int) (cols col : Nat) (active : col<cols) :
    ((start≤label ∧ label<start+(cols : Int)) ∧ (col : Int)=safeTarget label start cols-start) ↔
      start+(col : Int)=label := by
  by_cases owned : start≤label ∧ label<start+(cols : Int)
  · simp only [safeTarget,if_pos owned,owned,true_and,↓reduceIte]
    omega
  · simp only [safeTarget,if_neg owned,owned,false_and,false_iff,↓reduceIte]
    omega

/-- The source's ownership guard and selected subtraction give precisely the
mathematical block-owner branch, including all non-owning signed labels. -/
theorem safe_block_owner (p : Parameters) (target : Int) (block : Nat)
    (startNonneg : 0≤p.classStart) (metadata : p.classStart+(p.cols : Int)≤(p.totalClasses : Int))
    (totalBound : p.totalClasses<2^63) :
    (Source.owns p target ∧
      (block*p.block : Int)≤(safeShift target p.classStart p.cols).toInt ∧
      (safeShift target p.classStart p.cols).toInt<(min p.cols ((block+1)*p.block) : Nat)) ↔
      Source.blockOwns p target block := by
  rw [safe_shift_signed target p.classStart p.cols p.totalClasses startNonneg metadata totalBound]
  unfold Source.owns Source.blockOwns
  by_cases owns : p.classStart≤target ∧ target<p.classStart+(p.cols : Int)
  · simp only [owns,if_pos,true_and]
  · simp only [owns,if_neg,false_and,false_iff]
    have lower : (0 : Int)≤(block*p.block : Int) := mul_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
    have upper := Nat.min_le_left p.cols ((block+1)*p.block)
    omega
/-- Scalar output indexing and every padded-lane index reuse the frozen
64-bit derivation. The z-loss has the same scalar address formula. -/
theorem all_output_slots (p : Parameters) (count row block : Nat)
    (api : Supported p.rows count p.block p.cols) (hr : row<p.rows) (hb : block<count) :
    scalarOutput p.lossBase p.rows row block=BitVec.ofNat 64 (p.lossBase+block*p.rows+row) ∧
    scalarOutput p.lseBase p.rows row block=BitVec.ofNat 64 (p.lseBase+block*p.rows+row) ∧
    scalarOutput p.zLossBase p.rows row block=BitVec.ofNat 64 (p.zLossBase+block*p.rows+row) :=
  ⟨scalar_output_word _ _ _ _ _ _ _ api hr hb,
   scalar_output_word _ _ _ _ _ _ _ api hr hb,
   scalar_output_word _ _ _ _ _ _ _ api hr hb⟩

#axiomsClean int32_byte_coupling
#axiomsClean safe_shift_signed
#axiomsClean safe_target_comparison
#axiomsClean safe_block_owner
#axiomsClean select_after_subtraction
#axiomsClean int32_widen
#axiomsClean all_output_slots
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Native
