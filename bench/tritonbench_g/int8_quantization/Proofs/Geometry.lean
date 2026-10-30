import bench.tritonbench_g.int8_quantization.Kernels
import bench.tritonbench_g.int8_quantization.Proofs.Words

namespace VeriTile.Bench.Repairs.Int8Quantization.Geometry
open VeriTile.Triton
open Int8Quantization.Words (signed64 signed64_exact)
noncomputable section
set_option maxHeartbeats 3000000

def blocks (length block : Nat) : Nat := (length+block-1)/block

theorem blocks_cover (length block : Nat) (positive : 0< block) :
    length ≤ blocks length block*block := by
  have rem := Nat.mod_lt (length+block-1) positive
  have decomp := Nat.mod_add_div (length+block-1) block
  dsimp [blocks]
  rw [Nat.mul_comm block] at decomp
  omega

theorem row_block (length block row : Nat) (positive : 0< block) (active : row< length) :
    row/block < blocks length block := by
  apply (Nat.div_lt_iff_lt_mul positive).mpr
  exact lt_of_lt_of_le active (blocks_cover length block positive)

/-- Every logical row/channel is assigned to its own batch/block and one
active lane, including non-power-of-two tiles and partial final blocks. -/
theorem coordinate (batches length channels block batch row channel width : Nat)
    (hb : 0< block) (hc : 0< channels) (hbatch : batch< batches)
    (hr : row< length) (hd : channel< channels) (cover : block*channels≤ width) :
    let group := batch*blocks length block+row/block
    let lane := row%block*channels+channel
    group < batches*blocks length block ∧
    lane< width ∧ lane< block*channels ∧
    (group%blocks length block)*block+lane/channels=row ∧
    (group/blocks length block)*length*channels+
      (group%blocks length block)*block*channels+lane=batch*length*channels+row*channels+channel := by
  dsimp
  have rb := row_block length block row hb hr
  have cb : 0< blocks length block := lt_of_le_of_lt (Nat.zero_le _) rb
  have rm := Nat.mod_lt row hb
  have lane : row%block*channels+channel< block*channels := by
    nlinarith
  have gm : (batch*blocks length block+row/block)%blocks length block=row/block := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt rb]
  have gd : (batch*blocks length block+row/block)/blocks length block=batch := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ cb,Nat.div_eq_of_lt rb]
    omega
  have ld : (row%block*channels+channel)/channels=row%block := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hc,Nat.div_eq_of_lt hd]
    omega
  have rd := Nat.mod_add_div row block
  rw [Nat.mul_comm block] at rd
  refine ⟨by nlinarith,lt_of_lt_of_le lane cover,lane,?_,?_⟩
  · rw [gm,ld]
    omega
  · rw [gm,gd]
    nlinarith

/-- The literal widened address is inside the allocated FP32 tensor whenever
its source mask is active. Inactive lanes do not dereference their addresses. -/
theorem active_offset (batches length channels block batch tile lane : Nat)
    (hc : 0< channels) (hb : batch< batches)
    (active : tile*block+lane/channels< length) :
    batch*length*channels+tile*block*channels+lane< batches*length*channels := by
  have rem := Nat.mod_lt lane hc
  have decomp := Nat.mod_add_div lane channels
  have inner : tile*block*channels+lane< length*channels := by nlinarith
  have next : (batch+1)*length*channels≤ batches*length*channels :=
    Nat.mul_le_mul_right channels (Nat.mul_le_mul_right length (by omega))
  nlinarith

theorem widened_offset (batches length channels block batch tile lane : Nat)
    (hc : 0< channels) (hb : batch< batches)
    (active : tile*block+lane/channels< length)
    (allocation : 4*(batches*length*channels)≤2^63-1) :
    signed64 (signed64 (signed64 ((batch : Int)*length)*channels)+
      signed64 (signed64 ((tile : Int)*block)*channels)+(lane : Int))=
      (batch*length*channels+tile*block*channels+lane : Nat) := by
  have bound := active_offset batches length channels block batch tile lane hc hb active
  have nc : 0< channels := hc
  have b0 : batch*length≤ batch*length*channels := Nat.le_mul_of_pos_right _ hc
  have l0 : tile*block≤ tile*block*channels := Nat.le_mul_of_pos_right _ hc
  have one : signed64 ((batch : Int)*length)=(batch*length : Nat) := by
    rw [←Nat.cast_mul];apply signed64_exact <;> omega
  have two : signed64 (((batch*length : Nat) : Int)*channels)=(batch*length*channels : Nat) := by
    rw [←Nat.cast_mul];apply signed64_exact <;> omega
  have three : signed64 ((tile : Int)*block)=(tile*block : Nat) := by
    rw [←Nat.cast_mul];apply signed64_exact <;> omega
  have four : signed64 (((tile*block : Nat) : Int)*channels)=(tile*block*channels : Nat) := by
    rw [←Nat.cast_mul];apply signed64_exact <;> omega
  rw [one,two,three,four]
  simp only [←Nat.cast_add]
  apply signed64_exact <;> omega

#axiomsClean blocks_cover
#axiomsClean coordinate
#axiomsClean active_offset
#axiomsClean widened_offset
end
end VeriTile.Bench.Repairs.Int8Quantization.Geometry
