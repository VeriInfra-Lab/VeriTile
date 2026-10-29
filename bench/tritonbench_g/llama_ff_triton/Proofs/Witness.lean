import bench.tritonbench_g.llama_ff_triton.Proofs.Flat

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.FiniteWitness
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

/-- Five finite disjoint allocations include both unmasked K lanes, one active
output, one masked output and a typed sentinel between output addresses. -/
def allocation : FlatAlloc :=
  FlatAlloc.ofList "flat" [("A", 3), ("W1", 8), ("W3", 8), ("R", 4), ("Y", 3)]

def initial (fp8 : Bool) : BlockState where
  mem := fun r _ => if r = "A" then .real 1
    else if r = "W1" ∨ r = "W3" ∨ r = "R" then
      if fp8 then .of .int 60 else .real 1
    else .of .int 37
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1

def oneByte : Mathematics.Byte := ⟨60, by decide⟩

theorem oneByte_value : oneByte.value = 1 := by
  have h : (Float8E5Bits.ofInt 60).decodeRat = some (1 : ℚ) := by
    rw [show Float8E5Bits.ofInt 60 = (60 : Fin 256) from rfl]
    norm_num [Float8E5Bits.decodeRat, Float8E5Bits.exponent,
      Float8E5Bits.fraction, Float8E5Bits.sign]
  simp [Mathematics.Byte.value, oneByte, h]

def source (fp8 variant : Bool) : ComputeKernel :=
  TypedSource.kernel fp8 variant "A" "W1" "W3" "Y" "R"
    1 1 1 9 2 7 2 7 2 7 2 3 1 2 2 2

private theorem formula (i : Fin 1) (j : Fin 2) :
    Mathematics.output 1 2 1 2 2 (fun _ _ _ => 1) (fun _ _ => 1)
      (fun _ _ _ => 1) (fun _ _ _ => 1) i j = Real.sigmoid 1 := by
  norm_num [Mathematics.output, Mathematics.norm, Mathematics.weightedSum,
    Mathematics.steps, PaddedRMSGemm.norm, PaddedRMSGemm.weightedSum, cdiv, Fin.sum_univ_succ]

private theorem extents : allocation.extent "A" = 3 ∧ allocation.extent "W1" = 8 ∧
    allocation.extent "W3" = 8 ∧ allocation.extent "R" = 4 ∧ allocation.extent "Y" = 3 := by decide

private theorem reads (fp8 : Bool) :
    AccessSafety.Reads allocation.extent (initial fp8) "A" "W1" "W3" "R"
      1 1 1 1 2 2 9 2 7 2 7 2 3 := by
  unfold AccessSafety.Reads
  simp only [extents.1, extents.2.1, extents.2.2.1, extents.2.2.2.1]
  refine ⟨?_, ?_, ?_, ?_⟩
  all_goals
    intros
    norm_num [rowIndex, rowGlobal, colIndex, colGlobal, pidM, pidN, initial,
      allocation, FlatAlloc.ofList, FlatAlloc.listExtent, cdiv, Region.mk.injEq, String.reduceEq] at *
    omega

private theorem writes (fp8 : Bool) :
    ∀ i : TileIndex [1, 2], active (initial fp8) 1 1 1 2 i →
      outOffset (initial fp8) 1 1 2 7 2 i < allocation.extent "Y" := by
  intro i hi
  rw [extents.2.2.2.2]
  simp [active, rowGlobal, colGlobal, pidM, pidN, initial, cdiv] at hi
  have hi0 : i.1.val = 0 := by omega
  have hi1 : i.2.1.val = 0 := by omega
  norm_num [outOffset, rowGlobal, colGlobal, pidM, pidN, initial, hi0, hi1,
    allocation, FlatAlloc.ofList, FlatAlloc.listExtent, cdiv, Region.mk.injEq, String.reduceEq]

/-- A successful nonempty flat run of both full sources. K=1/BK=2 keeps the
padding lane in the real RMS/GEMM sums; BN=2/N=1 masks the second output lane.
The numerical result is nonzero and all other raw typed cells are preserved. -/
theorem execution (fp8 variant : Bool) :
    (source fp8 variant).eraseDType.toAlgKernel.TraceSafe allocation.extent (initial fp8) ∧
    ∃ t, exec (allocation.flattenKernel (source fp8 variant).eraseDType.toAlgKernel)
        (allocation.flattenState (initial fp8)) = some t ∧
      t.readMem allocation.flat (allocation.addr "Y" 0) = Real.sigmoid 1 ∧
      (∀ r o, (r ≠ allocation.flat ∨ o ≠ allocation.addr "Y" 0) →
        t.mem r o = (allocation.flattenState (initial fp8)).mem r o) := by
  have hd : allocation.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)
  have hc : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := FlatAlloc.ofList_closed _ _
  have ha : ∀ (fp8 : Bool) (i : TileIndex [1, 2]), active (initial fp8) 1 1 1 2 i →
      outOffset (initial fp8) 1 1 2 7 2 i = 0 := by
    intro fp8 i hi
    have hi0 : i.1.val = 0 := by omega
    have hi1 : i.2.1.val = 0 := by
      simpa [active, rowGlobal, colGlobal, pidM, pidN, initial, cdiv, hi0] using hi
    simp [outOffset, rowGlobal, colGlobal, pidM, pidN, initial, cdiv, hi0, hi1]
  have finish : ∀ fp8,
      (∃ t, exec (allocation.flattenKernel (source fp8 variant).eraseDType.toAlgKernel)
          (allocation.flattenState (initial fp8)) = some t ∧
        (∀ i : TileIndex [1, 2], active (initial fp8) 1 1 1 2 i →
          t.readMem allocation.flat (allocation.addr "Y" (outOffset (initial fp8) 1 1 2 7 2 i)) = Real.sigmoid 1) ∧
        (∀ r o, (r ≠ allocation.flat ∨ ∀ i : TileIndex [1, 2], active (initial fp8) 1 1 1 2 i →
          o ≠ allocation.addr "Y" (outOffset (initial fp8) 1 1 2 7 2 i)) →
          t.mem r o = (allocation.flattenState (initial fp8)).mem r o)) →
      ∃ t, exec (allocation.flattenKernel (source fp8 variant).eraseDType.toAlgKernel)
          (allocation.flattenState (initial fp8)) = some t ∧
        t.readMem allocation.flat (allocation.addr "Y" 0) = Real.sigmoid 1 ∧
        (∀ r o, (r ≠ allocation.flat ∨ o ≠ allocation.addr "Y" 0) →
          t.mem r o = (allocation.flattenState (initial fp8)).mem r o) := by
    intro fp8 h
    obtain ⟨t, he, hv, hf⟩ := h
    refine ⟨t, he, ?_, ?_⟩
    · simpa [outOffset, rowGlobal, colGlobal, pidM, pidN, initial, cdiv] using
        hv (0, 0, ()) (by simp [active, rowGlobal, colGlobal, pidM, pidN, initial, cdiv])
    · intro r o hn
      apply hf r o
      rcases hn with hr | ho
      · exact Or.inl hr
      · exact Or.inr (fun i hi => by simpa [ha fp8 i hi] using ho)
  cases fp8
  · have hin : RealExecution.Inputs (initial Bool.false) "A" "W1" "W3" "R"
        1 1 1 1 2 2 9 2 7 2 7 2 3 (fun _ _ _ => 1) (fun _ _ => 1)
        (fun _ _ _ => 1) (fun _ _ _ => 1) := by
      simp [RealExecution.Inputs, aElem, rmsElem, wElem, initial, BlockState.readMem]
    refine ⟨AccessSafety.complete_fp16 variant allocation.extent "A" "W1" "W3" "Y" "R" (initial Bool.false)
      1 1 1 9 2 7 2 7 2 7 2 3 1 2 2 2 (by intros; rfl) (reads Bool.false) (writes Bool.false), ?_⟩
    apply finish Bool.false
    simpa only [formula] using FlatIO.half variant allocation "A" "W1" "W3" "Y" "R" (initial Bool.false)
      1 1 1 9 2 7 2 7 2 7 2 3 1 2 2 2 (fun _ _ _ => 1) (fun _ _ => 1)
      (fun _ _ _ => 1) (fun _ _ _ => 1) hd hc (by simp [allocation, FlatAlloc.ofList])
      (by intros; rfl) (by decide) (by decide) hin (reads Bool.false) (writes Bool.false)
  · have hin : ByteExecution.Inputs (initial Bool.true) "A" "W1" "W3" "R"
        1 1 1 1 2 2 9 2 7 2 7 2 3 (fun _ _ _ => 1) (fun _ _ => oneByte)
        (fun _ _ _ => oneByte) (fun _ _ _ => oneByte) := by
      simp [ByteExecution.Inputs, aElem, initial, BlockState.readMem,
        BlockState.readMemTyped, oneByte, MemCell.of, MemCell.readAs, MemCell.real]
    have hbytes := ByteExecution.inputs_bytes _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ hin
    refine ⟨ByteExecution.complete variant allocation.extent "A" "W1" "W3" "Y" "R" (initial Bool.true)
      1 1 1 9 2 7 2 7 2 7 2 3 1 2 2 2 (by intros; rfl)
      (by decide) (by decide) (by decide) hbytes (reads Bool.true) (writes Bool.true), ?_⟩
    apply finish Bool.true
    simpa only [oneByte_value, formula] using FlatIO.bytes variant allocation "A" "W1" "W3" "Y" "R" (initial Bool.true)
      1 1 1 9 2 7 2 7 2 7 2 3 1 2 2 2 (fun _ _ _ => 1) (fun _ _ => oneByte)
      (fun _ _ _ => oneByte) (fun _ _ _ => oneByte) hd hc (by simp [allocation, FlatAlloc.ofList])
      (by intros; rfl) (by decide) (by decide) (by decide) (by decide) (by decide)
      hin (reads Bool.true) (writes Bool.true)

theorem output_positive : 0 < Real.sigmoid 1 := Real.sigmoid_pos 1

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.FiniteWitness
