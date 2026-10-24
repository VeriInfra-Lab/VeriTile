import bench.tritonbench_g.rms_matmul_rbe.Proofs.FP8Witness
import bench.tritonbench_g.rms_matmul_rbe.Proofs.QKVWitness

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVByteWitness
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Eight finite disjoint allocations, with nonzero batch offset, unrelated
weight strides, partial N and a real allocated unmasked K-padding lane. -/
def allocation : FlatAlloc := FlatAlloc.ofList "flat"
  [("X", 16), ("QW", 8), ("KW", 9), ("VW", 10), ("R", 5), ("Q", 24), ("K", 26), ("V", 28)]

def initial : BlockState where
  mem := fun r _ => if r = "X" then MemCell.real 1
    else if r = "QW" then MemCell.of .int 64
    else if r = "KW" then MemCell.of .int 66
    else if r = "VW" then MemCell.of .int 68
    else if r = "R" then MemCell.of .int 60
    else MemCell.of .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

abbrev oneByte := ByteWitness.oneByte
abbrev twoByte := ByteWitness.twoByte
def threeByte : ByteExecution.Byte := ⟨66, by decide⟩
def fourByte : ByteExecution.Byte := ⟨68, by decide⟩

theorem threeByte_value : threeByte.value = 3 := by
  have h : (Float8E5Bits.ofInt 66).decodeRat = some (3 : ℚ) := by
    rw [show Float8E5Bits.ofInt 66 = (66 : Fin 256) from rfl]
    norm_num [Float8E5Bits.decodeRat, Float8E5Bits.exponent,
      Float8E5Bits.fraction, Float8E5Bits.sign]
  simp [ByteExecution.Byte.value, threeByte, h]

theorem fourByte_value : fourByte.value = 4 := by
  have h : (Float8E5Bits.ofInt 68).decodeRat = some (4 : ℚ) := by
    rw [show Float8E5Bits.ofInt 68 = (68 : Fin 256) from rfl]
    norm_num [Float8E5Bits.decodeRat, Float8E5Bits.exponent,
      Float8E5Bits.fraction, Float8E5Bits.sign]
  simp [ByteExecution.Byte.value, fourByte, h]

theorem inputQ : ByteExecution.Inputs initial "X" "QW" "R"
    1 1 1 1 2 2 11 7 2 3 1 2
    (fun _ _ _ => 1) (fun _ _ => oneByte) (fun _ _ _ => twoByte) :=
  ⟨fun _ _ _ => rfl, fun _ _ => rfl, fun _ _ _ => rfl⟩

theorem inputK : ByteExecution.Inputs initial "X" "KW" "R"
    1 1 1 1 2 2 11 7 2 4 1 2
    (fun _ _ _ => 1) (fun _ _ => oneByte) (fun _ _ _ => threeByte) :=
  ⟨fun _ _ _ => rfl, fun _ _ => rfl, fun _ _ _ => rfl⟩

theorem inputV : ByteExecution.Inputs initial "X" "VW" "R"
    1 1 1 1 2 2 11 7 2 5 1 2
    (fun _ _ _ => 1) (fun _ _ => oneByte) (fun _ _ _ => fourByte) :=
  ⟨fun _ _ _ => rfl, fun _ _ => rfl, fun _ _ _ => rfl⟩

def kernel (input : TypedSource.Storage) (variant : Bool) : ComputeKernel :=
  TypedSource.qkv input Bool.true variant "X" "QW" "KW" "VW" "R" "Q" "K" "V"
    1 1 1 11 7 2 3 1 4 1 5 1 2 17 5 2 19 7 2 21 9 2 1 2 2 3 10000 2

theorem disjoint : allocation.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)
theorem closed : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := FlatAlloc.ofList_closed _ _

theorem reads (W : RegionName) (stride : Nat)
    (hb : stride < allocation.extent W) :
    AccessSafety.Reads allocation.extent initial "X" W "R"
      1 1 1 1 2 2 11 7 2 stride 1 2 := by
  refine ⟨?_, ?_, ?_⟩
  · intro t ht i e
    have ht' : t = 0 := by norm_num [cdiv] at ht; omega
    subst t
    simp only [rowIndex, Nat.mod_one, Nat.zero_mul, Nat.add_zero]
    change 11 + (0 * 2 + e.val) * 2 < 16
    have := e.isLt
    omega
  · intro t ht e j
    have ht' : t = 0 := by norm_num [cdiv] at ht; omega
    subst t
    simp only [colIndex, Nat.mod_one, Nat.zero_mul, Nat.add_zero]
    have he : e.val ≤ 1 := by have := e.isLt; omega
    have hm := Nat.mul_le_mul_right stride he
    simp only [one_mul] at hm
    simpa using lt_of_le_of_lt hm hb
  · intro t ht e
    have ht' : t = 0 := by norm_num [cdiv] at ht; omega
    subst t
    change (0 * 2 + e.val) * 2 < 5
    have := e.isLt
    omega

theorem active_lanes (i : TileIndex [1, 2]) (hi : active initial 1 1 1 2 i) :
    i.1.val = 0 ∧ i.2.1.val = 0 := by
  have hm := i.1.isLt
  have hn := hi.2
  change 0 * 2 + i.2.1.val < 1 at hn
  omega

theorem writes (OUT : RegionName) (sb sm : Nat) (hb : sb < allocation.extent OUT)
    (i : TileIndex [1, 2]) (hi : active initial 1 1 1 2 i) :
    outOffset initial 1 1 2 sb sm 2 i < allocation.extent OUT := by
  obtain ⟨hm, hn⟩ := active_lanes i hi
  simpa [outOffset, rowGlobal, colGlobal, initial, pidM, pidN, cdiv, hm, hn] using hb

/-- The full three-output source contract has no execution, readback or
bounds hypotheses left to assume, for all six floating profiles and variants. -/
theorem contract (input : TypedSource.Storage) (variant : Bool) :
    Contracts.QKVByteIO (kernel input variant) allocation initial "Q" "K" "V"
      1 1 1 17 5 2 19 7 2 21 9 2 1 2 2 2
      (fun _ _ _ => 1) (fun _ _ => oneByte) (fun _ _ _ => twoByte)
      (fun _ _ _ => threeByte) (fun _ _ _ => fourByte) := by
  have hq := Geometry.active_injective initial 1 1 1 2 17 5 2 (by decide) (by decide)
  have hk := Geometry.active_injective initial 1 1 1 2 19 7 2 (by decide) (by decide)
  have hv := Geometry.active_injective initial 1 1 1 2 21 9 2 (by decide) (by decide)
  have rq := reads "QW" 3 (by decide)
  have rk := reads "KW" 4 (by decide)
  have rv := reads "VW" 5 (by decide)
  have oq := writes "Q" 17 5 (by decide)
  have ok := writes "K" 19 7 (by decide)
  have ov := writes "V" 21 9 (by decide)
  refine ⟨?_, QKVByteFlat.arrays input variant allocation "X" "QW" "KW" "VW" "R" "Q" "K" "V" initial
    1 1 1 11 7 2 3 1 4 1 5 1 2 17 5 2 19 7 2 21 9 2 1 2 2 3 10000 2 _ _ _ _ _ rfl hq hk hv
    (by decide) (by decide) (by decide) (by decide) inputQ inputK inputV
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) disjoint closed rq rk rv oq ok ov⟩
  exact QKVByteFlat.safe_bytes input variant allocation.extent "X" "QW" "KW" "VW" "R" "Q" "K" "V" initial
    1 1 1 11 7 2 3 1 4 1 5 1 2 17 5 2 19 7 2 21 9 2 1 2 2 3 10000 2
    (by intros; rfl) hq hk hv (by decide) (by decide) (by decide) (by decide)
    (ByteExecution.inputs_bytes initial "X" "QW" "R" 1 1 1 1 2 2 11 7 2 3 1 2 _ _ _ inputQ)
    (ByteExecution.inputs_bytes initial "X" "KW" "R" 1 1 1 1 2 2 11 7 2 4 1 2 _ _ _ inputK)
    (ByteExecution.inputs_bytes initial "X" "VW" "R" 1 1 1 1 2 2 11 7 2 5 1 2 _ _ _ inputV)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide) rq rk rv oq ok ov

/-- All outputs are nonzero and different, and a masked Q output cell keeps
its original integer sentinel after the entire flattened three-pass run. -/
theorem cells (input : TypedSource.Storage) (variant : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel input variant).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "Q" 17) = 2 ∧
      t.readMem allocation.flat (allocation.addr "K" 19) = 3 ∧
      t.readMem allocation.flat (allocation.addr "V" 21) = 4 ∧
      t.mem allocation.flat (allocation.addr "Q" 19) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "K" 21) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "V" 23) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "QW" 3) = MemCell.of .int 64 ∧
      t.mem allocation.flat (allocation.addr "R" 2) = MemCell.of .int 60 := by
  obtain ⟨_, t, he, hq, hk, hv, hf⟩ := contract input variant
  refine ⟨t, he, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · change t.readMem allocation.flat
      (allocation.addr "Q" (outOffset initial 1 1 2 17 5 2 (0, 0, PUnit.unit))) = 2
    rw [hq _ (by decide)]
    norm_num [PaddedRMSGemm.output, PaddedRMSGemm.norm, PaddedRMSGemm.weightedSum,
      ByteWitness.oneByte_value, ByteWitness.twoByte_value, threeByte_value, fourByte_value,
      cdiv, Fin.sum_univ_succ]
  · change t.readMem allocation.flat
      (allocation.addr "K" (outOffset initial 1 1 2 19 7 2 (0, 0, PUnit.unit))) = 3
    rw [hk _ (by decide)]
    norm_num [PaddedRMSGemm.output, PaddedRMSGemm.norm, PaddedRMSGemm.weightedSum,
      ByteWitness.oneByte_value, ByteWitness.twoByte_value, threeByte_value, fourByte_value,
      cdiv, Fin.sum_univ_succ]
  · change t.readMem allocation.flat
      (allocation.addr "V" (outOffset initial 1 1 2 21 9 2 (0, 0, PUnit.unit))) = 4
    rw [hv _ (by decide)]
    norm_num [PaddedRMSGemm.output, PaddedRMSGemm.norm, PaddedRMSGemm.weightedSum,
      ByteWitness.oneByte_value, ByteWitness.twoByte_value, threeByte_value, fourByte_value,
      cdiv, Fin.sum_univ_succ]
  all_goals
    rw [hf _ _ (Or.inr ?_)]
    · rw [allocation.flattenState_mem_addr disjoint initial (by decide) (by decide)]
      rfl
    · intro r o hw
      rcases hw with ⟨i, hi, hw⟩
      obtain ⟨hm, hn⟩ := active_lanes i hi
      rcases hw with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      all_goals simp [outOffset, rowGlobal, colGlobal, initial, pidM, pidN, cdiv, hm, hn,
        allocation, FlatAlloc.addr, FlatAlloc.ofList]
      all_goals decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.QKVByteWitness
