import bench.repairs.tritonbench_g.logsumexp_fwd.Proofs.Contract

namespace VeriTile.Bench.Repairs.LogsumexpFwd.Machine
open VeriTile.Triton
open MaxReduction
noncomputable section

theorem columns_bound (a : Arguments) (h : InputContract a) (nonempty : 0<a.rows) : a.columns<2^31 := by
  have size : a.rows*a.columns<2^31 := by rw [input_numel a h]; exact h.2.2.2.2.2.2.1
  exact lt_of_le_of_lt (Nat.le_mul_of_pos_left _ nonempty) size

theorem block_covers_or_cap (a : Arguments) : a.columns≤a.block ∨ a.block=65536 := by
  by_cases hn : MaxReduction.nextPower a.columns≤65536
  · left
    simpa [Arguments.block,Nat.min_eq_left hn] using MaxReduction.nextPower_covers a.columns
  · right
    exact Nat.min_eq_right (by omega)

/-- The original two-dimensional CUDA launch is within grid.y without a
new runtime guard. Empty leading shapes submit no launch. -/
theorem grid_bound (a : Arguments) (h : InputContract a) (nonempty : 0<a.rows) :
    a.rows<2^31 ∧ a.blocks≤32768 ∧ a.rows*a.blocks<2^31 := by
  have hd := columns_bound a h nonempty
  have hb := block_positive a
  have hc : a.blocks≤32768 := by
    rcases block_covers_or_cap a with covers|cap
    · have one : a.blocks≤1 := by
        change (a.columns+a.block-1)/a.block≤1
        apply Nat.le_of_lt_succ
        apply (Nat.div_lt_iff_lt_mul hb).mpr
        omega
      omega
    · change (a.columns+a.block-1)/a.block≤32768
      rw [cap]
      omega
  refine ⟨h.2.2.2.2.2.2.2.2.2.2,hc,?_⟩
  by_cases zero : a.columns=0
  · have empty : a.blocks=0 := by
      dsimp [Arguments.blocks]
      rw [zero]
      exact Nat.div_eq_of_lt (by omega)
    simp [empty]
  · exact scratch_bound a h (Nat.pos_of_ne_zero zero)

theorem native_mask (a : Arguments) (h : InputContract a) (row block lane : Nat)
    (hr : row<a.rows) (hb : block<a.blocks) (hl : lane<a.block) :
    (Address.word block*Address.word a.block+Address.word lane).slt (Address.word a.columns)=
      decide (block*a.block+lane<a.columns) :=
  MaxReduction.Machine.scalar_mask a.columns a.block block lane
    (columns_bound a h (by omega)) (block_positive a) (block_bound a) hb hl

/-- Widened source pointer polynomials do not wrap, even for padded lanes.
Only lanes satisfying the proved mask are loaded. -/
theorem native_input_offset (a : Arguments) (h : InputContract a) (row block lane : Nat)
    (hr : row<a.rows) (hb : block<a.blocks) (hl : lane<a.block) :
    (Address.word row*Address.word a.columns+
      (Address.word block*Address.word a.block+Address.word lane)).toNat=
      row*a.columns+(block*a.block+lane) := by
  have start := active_block a block hb
  have size : a.rows*a.columns<2^31 := by rw [input_numel a h]; exact h.2.2.2.2.2.2.1
  have rowEnd : row*a.columns+a.columns≤a.rows*a.columns := by
    simpa [Nat.add_mul] using Nat.mul_le_mul_right a.columns hr
  have cap := block_bound a
  have bound : row*a.columns+(block*a.block+lane)<2^64 := by omega
  simp only [Address.word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt bound

theorem native_output_offset (a : Arguments) (h : InputContract a) (row block : Nat)
    (hr : row<a.rows) (hb : block<a.blocks) :
    (Address.word row*Address.word a.blocks+Address.word block).toNat=row*a.blocks+block := by
  have size := (grid_bound a h (by omega)).2.2
  have bound := output_offset_bound a row block hr hb
  simp only [Address.word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by omega)

/-- Exact FP32 byte address under the ordinary backed-allocation law. -/
theorem pointer32 (base count index : Nat) (backed : base+4*count≤2^64) (active : index<count) :
    (Address.word base+Address.word index*Address.word 4).toNat=base+4*index :=
  MaxReduction.Machine.active_pointer base count 4 index (by decide) backed active

#axiomsClean grid_bound
#axiomsClean native_mask
#axiomsClean native_input_offset
#axiomsClean native_output_offset
#axiomsClean pointer32
end
end VeriTile.Bench.Repairs.LogsumexpFwd.Machine
