import bench.repairs.tritonbench_g.logsumexp_fwd.Contract
import bench.repairs.tritonbench_g.max_reduction.Proofs.Machine

namespace VeriTile.Bench.Repairs.LogsumexpFwd
open VeriTile.Torch
noncomputable section

theorem checkInputs_iff (a : Arguments) : checkInputs a=.ok () ↔ InputContract a := by
  classical
  simp only [checkInputs,InputContract]
  split_ifs <;> simp_all

theorem shape_product (shape : List Nat) (nonempty : shape≠[]) :
    shape.dropLast.prod*shape.getLastD 0=shape.prod := by
  have split := List.dropLast_append_getLast nonempty
  have product := congrArg List.prod split
  simpa only [List.getLastD_eq_getLast?, List.getLast?_eq_some_getLast nonempty, Option.getD_some, List.prod_append, List.prod_singleton] using product

theorem input_numel (a : Arguments) (h : InputContract a) : a.rows*a.columns=a.input.numel :=
  shape_product _ h.2.2.2.1

theorem block_positive (a : Arguments) : 0<a.block := by
  exact lt_min (MaxReduction.nextPower_positive _) (by decide)

theorem block_bound (a : Arguments) : a.block≤65536 := Nat.min_le_right _ _

theorem blocks_positive (a : Arguments) (hd : 0<a.columns) : 0<a.blocks := by
  have hb := block_positive a
  change 0<(a.columns+a.block-1)/a.block
  exact Nat.div_pos_iff.mpr ⟨hb,by omega⟩

theorem blocks_le_columns (a : Arguments) (hd : 0<a.columns) : a.blocks≤a.columns := by
  have hb := block_positive a
  have hm : a.columns≤a.columns*a.block := Nat.le_mul_of_pos_right _ hb
  have hprod : a.columns+a.block-1<(a.columns+1)*a.block := by
    rw [Nat.add_mul,Nat.one_mul]
    omega
  exact Nat.le_of_lt_succ ((Nat.div_lt_iff_lt_mul hb).mpr hprod)

theorem scratch_bound (a : Arguments) (h : InputContract a) (hd : 0<a.columns) :
    a.rows*a.blocks<2^31 :=
  lt_of_le_of_lt (Nat.mul_le_mul_left _ (blocks_le_columns a hd))
    (by rw [input_numel a h]; exact h.2.2.2.2.2.2.1)

theorem active_block (a : Arguments) (block : Nat) (hb : block<a.blocks) : block*a.block<a.columns :=
  MaxReduction.Machine.segment_start _ _ _ (block_positive a) hb

theorem input_offset_bound (a : Arguments) (h : InputContract a) (row block lane : Nat)
    (hr : row<a.rows) (active : block*a.block+lane<a.columns) :
    row*a.columns+(block*a.block+lane)<a.input.numel := by
  rw [←input_numel a h]
  exact MaxReduction.Address.output_bound _ _ _ _ hr active

theorem output_offset_bound (a : Arguments) (row block : Nat) (hr : row<a.rows) (hb : block<a.blocks) :
    row*a.blocks+block<a.rows*a.blocks := MaxReduction.Address.output_bound _ _ _ _ hr hb

#axiomsClean checkInputs_iff
#axiomsClean input_numel
#axiomsClean scratch_bound
#axiomsClean input_offset_bound
end
end VeriTile.Bench.Repairs.LogsumexpFwd
