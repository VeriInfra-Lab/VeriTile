import bench.tritonbench_g.rms_rbe_matmul.Proofs.Witness

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteWitness
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000

abbrev allocation := FiniteWitness.allocation

/-- Genuine integer byte inputs and a nonzero batch id. RMS decodes to one,
matrix weights decode to two, and all unspecified cells are typed sentinels. -/
def initial : BlockState where
  mem := fun r _ => if r = "X" then .real 1
    else if r = "W" then .of .int 64
    else if r = "R" then .of .int 60
    else .of .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun _ _ => 0

def oneByte : ByteExecution.Byte := ⟨60, by decide⟩
def twoByte : ByteExecution.Byte := ⟨64, by decide⟩

theorem oneByte_value : oneByte.value = 1 := by
  have h : (Float8E5Bits.ofInt 60).decodeRat = some (1 : ℚ) := by
    rw [show Float8E5Bits.ofInt 60 = (60 : Fin 256) from rfl]
    norm_num [Float8E5Bits.decodeRat, Float8E5Bits.exponent,
      Float8E5Bits.fraction, Float8E5Bits.sign]
  simp [ByteExecution.Byte.value, oneByte, h]

theorem twoByte_value : twoByte.value = 2 := by
  have h : (Float8E5Bits.ofInt 64).decodeRat = some (2 : ℚ) := by
    rw [show Float8E5Bits.ofInt 64 = (64 : Fin 256) from rfl]
    norm_num [Float8E5Bits.decodeRat, Float8E5Bits.exponent,
      Float8E5Bits.fraction, Float8E5Bits.sign]
  simp [ByteExecution.Byte.value, twoByte, h]

def kernel (input : TypedSource.Storage) (variant : Bool) : ComputeKernel :=
  TypedSource.kernel input Bool.true variant "X" "W" "R" "Y"
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2

theorem inputs : ByteExecution.Inputs initial "X" "W" "R"
    1 1 1 1 2 2 11 7 2 3 1 2
    (fun _ _ _ => 1) (fun _ _ => oneByte) (fun _ _ _ => twoByte) := by
  exact ⟨fun _ _ _ => rfl, fun _ _ => rfl, fun _ _ _ => rfl⟩

theorem reads : AccessSafety.Reads allocation.extent initial "X" "W" "R"
    1 1 1 1 2 2 11 7 2 3 1 2 := FiniteWitness.reads

theorem writes (i : TileIndex [1, 2]) (hi : active initial 1 1 1 2 i) :
    outOffset initial 1 1 2 17 5 2 i < allocation.extent "Y" :=
  FiniteWitness.writes i hi

/-- Every byte-input and finite-allocation premise is discharged for both
complete variants and all six formal X labels, with K=1/BK=2 and N=1/BN=2. -/
theorem contract (input : TypedSource.Storage) (variant : Bool) :
    Contracts.ByteIO (kernel input variant) allocation initial "Y"
      1 1 1 17 5 2 1 2 2 2
      (fun _ _ _ => 1) (fun _ _ => oneByte) (fun _ _ _ => twoByte) := by
  refine ⟨?_, FlatIO.bytes input variant allocation "X" "W" "R" "Y" initial
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2
    _ _ _ FiniteWitness.disjoint FiniteWitness.closed rfl (by decide) (by decide)
    (by decide) (by decide) inputs reads writes⟩
  exact ByteExecution.complete input variant allocation.extent "X" "W" "R" "Y" initial
    1 1 1 11 7 2 3 1 2 17 5 2 1 2 2 3 Bool.true 10000 2
    (by intros; rfl) (by decide) (by decide)
    (ByteExecution.inputs_bytes initial "X" "W" "R" 1 1 1 1 2 2 11 7 2 3 1 2 _ _ _ inputs)
    reads writes

private theorem output_address (i : TileIndex [1, 2]) (hi : active initial 1 1 1 2 i) :
    outOffset initial 1 1 2 17 5 2 i = 17 := by
  have hr : i.1.val = 0 := by omega
  have hc : i.2.1.val = 0 := by
    have hc := hi.2
    change 0 * 2 + i.2.1.val < 1 at hc
    omega
  simp [outOffset, rowGlobal, colGlobal, initial, pidM, pidN, cdiv, hr, hc]

/-- The full flattened byte source produces two at the nonzero batch output;
every other raw cell is preserved, including all input bytes and masked tails. -/
theorem execution (input : TypedSource.Storage) (variant : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel input variant).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "Y" 17) = 2 ∧
      (∀ r o, (r ≠ allocation.flat ∨ o ≠ allocation.addr "Y" 17) →
        t.mem r o = (allocation.flattenState initial).mem r o) := by
  obtain ⟨_, t, he, hv, hf⟩ := contract input variant
  refine ⟨t, he, ?_, ?_⟩
  · change t.readMem allocation.flat
      (allocation.addr "Y" (outOffset initial 1 1 2 17 5 2 (0, 0, PUnit.unit))) = 2
    rw [hv _ (by decide)]
    norm_num [PaddedRMSGemm.output, PaddedRMSGemm.norm, PaddedRMSGemm.weightedSum,
      oneByte_value, twoByte_value, cdiv, Fin.sum_univ_succ]
  · intro r o hn
    apply hf r o
    rcases hn with hn | hn
    · exact Or.inl hn
    · exact Or.inr (fun i hi => by simpa only [output_address i hi] using hn)

/-- Concrete nonzero readback, masked/before-output sentinels and unchanged
raw weight bytes are all observations of that same complete source execution. -/
theorem cells (input : TypedSource.Storage) (variant : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel input variant).eraseDType.toAlgKernel)
        (allocation.flattenState initial) = some t ∧
      t.readMem allocation.flat (allocation.addr "Y" 17) = 2 ∧
      t.mem allocation.flat (allocation.addr "Y" 19) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "Y" 0) = MemCell.of .int 37 ∧
      t.mem allocation.flat (allocation.addr "W" 3) = MemCell.of .int 64 ∧
      t.mem allocation.flat (allocation.addr "R" 2) = MemCell.of .int 60 := by
  obtain ⟨t, he, hv, hf⟩ := execution input variant
  refine ⟨t, he, hv, ?_, ?_, ?_, ?_⟩
  all_goals
    rw [hf _ _ (Or.inr (by decide)),
      allocation.flattenState_mem_addr FiniteWitness.disjoint initial (by decide) (by decide)]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.ByteWitness
