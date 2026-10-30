import bench.repairs.tritonbench_g.log_softmax.Proofs.Contract
import bench.repairs.tritonbench_g.max_reduction.Proofs.Machine

namespace VeriTile.Bench.Repairs.LogSoftmax.Machine
open VeriTile.Triton Address
noncomputable section
open Classical
abbrev word := MaxReduction.Address.word

/-- All source predicates use widened signed integers. Raw padded rows and
columns fit signed64 before clamping; the dense allocation then bounds every
clamped multiply/add, including masked lanes, as proved in Address. -/
theorem native_masks (a : Arguments) (h : InputContract a) (nonempty : 0<a.input.numel)
    (BM pid k i j : Nat) (retained : Grid.RetainedBlock BM)
    (hp : pid<Grid.groups a.M BM) (hk : k<a.K) (hi : i<BM) (hj : j<Grid.blockColumns a.N) :
    (word pid*word BM+word i).slt (word a.M)=decide (pid*BM+i<a.M) ∧
    (word j).slt (word a.N)=decide (j<a.N) ∧ (word k).slt (word a.K)=Bool.true := by
  have bounds := launch_bounds a h nonempty BM retained
  have size : a.M*a.N*a.K<2^31 := by rw [input_numel a h.2.2]; exact h.1.2.2.2.2.2.1
  have mb : a.M<2^31 := lt_of_le_of_lt
    ((Nat.le_mul_of_pos_right a.M bounds.2.1).trans (Nat.le_mul_of_pos_right _ bounds.2.2.1)) size
  have nb : a.N≤131072 := (h.2.2.2.2.resolve_left (by omega)).1
  have cb : Grid.blockColumns a.N≤131072 := MaxReduction.nextPower_bound _ 17 nb
  have pb : pid<2^31 := lt_trans (lt_of_lt_of_le hp
    (Nat.le_mul_of_pos_right _ bounds.2.2.1)) bounds.2.2.2.2.2
  have row := raw_row_exact pid BM i pb (Grid.block_bounds retained).2 hi
  refine ⟨?_,?_,?_⟩
  · simp only [word,MaxReduction.Address.word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,BitVec.slt_eq_decide]
    change decide ((word (pid*BM+i)).toInt<(word a.M).toInt)=_
    rw [MaxReduction.Machine.signed64 _ (by omega),MaxReduction.Machine.signed64 _ (by omega)]
    simp only [Int.ofNat_lt]
  · rw [BitVec.slt_eq_decide,MaxReduction.Machine.signed64 _ (by omega),MaxReduction.Machine.signed64 _ (by omega)]
    simp only [Int.ofNat_lt]
  · rw [BitVec.slt_eq_decide,MaxReduction.Machine.signed64 _ (by have := bounds.2.2.2.1; omega),
      MaxReduction.Machine.signed64 _ (by have := bounds.2.2.2.1; omega)]
    simp [hk]

theorem active_offset (q : Coord M N K) :
    offset N K (safe q.1.val M) (safe q.2.1.val N) (safe q.2.2.val K)=linear q := by
  rw [safe_active q.1.isLt,safe_active q.2.1.isLt,safe_active q.2.2.isLt]
  rfl

/-- Exact hardware FP32 base-plus-four-times-index under ordinary valid
allocation backing, with the complete four-byte footprint within the buffer. -/
theorem pointer32 (buffer : Dense count) (index : Nat) (active : index<count) :
    (word buffer.base+word index*word 4).toNat=buffer.address index ∧
    (∀ b,b<4 → buffer.contains (buffer.address index+b)) := by
  refine ⟨?_,fun b hb => (address_bytes buffer index active b hb).1⟩
  have backing : buffer.base+4*count≤2^64 := by simpa [Nat.mul_comm] using buffer.finite
  simpa [Dense.address,Nat.mul_comm] using MaxReduction.Machine.active_pointer buffer.base count 4 index
    (by decide) backing active

#axiomsClean native_masks
#axiomsClean active_offset
#axiomsClean pointer32
end
end VeriTile.Bench.Repairs.LogSoftmax.Machine
