import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FPArithmetic
import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FPInteger
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] Bits.narrow

def rawWritten (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.block*2]).foldl (fun t j =>
    if active p s j then t.write p.output (outputOffset p s j) (.mk .bf16 (converted (laneNibble p s j))) else t) s

private theorem masked_memory {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
    l.foldl (fun m i => if active i then (fun r' o => if r'=r ∧ o=off i then value i else m r' o) else m) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]
      rw [ih]
      rfl
    · simpa only [hi, ↓reduceIte] using ih s

/-- Generic final-source-slice lemma. It reasons about an arbitrary pre-store
state, so its proof never normalizes the earlier integer execution certificate. -/
theorem tail_memory (rest : Algebra Word) (p : Parameters) (start final : State Word)
    (pid : Nat) (words : TileIndex [p.block*2] → Int)
    (hw : start.regs .int [p.block*2] "result" = some words)
    (hp : start.regs .nat [] "pid" = some (fun _ => pid))
    (hc : start.regs .nat [] "n_elements_out" = some (fun _ => p.count*2))
    (h : run (backend rest) ((source p).surfaceBody.drop 25 |>.take 6) start = some final) :
    final.mem = (TileShape.allIndices [p.block*2]).foldl (fun mem j =>
      if pid*(p.block*2)+j.1.val < p.count*2 then
        (fun r o => if r=p.output ∧ o=pid*(p.block*2)+j.1.val then
          .mk .bf16 (embed (Bits.narrow ⟨BitVec.ofInt 32 (words j)⟩)) else mem r o)
      else mem) start.mem := by
  dsimp only [source, unscaled, ComputeKernel.surfaceBody, List.drop, List.take] at h
  simp [run, step, evalExpr, evalComputeOp, FP.Structural.evalOp, store,
    backend, hw, hp, hc, ComputeDType.eraseDType, FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.natLt,
    FP.Structural.ofFloat, FP.Structural.toFloat, Option.bind,
    FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast] at h
  simp [State.setReg, State.regs, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast] at h
  cases h
  simp [masked_memory, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]

theorem raw_execution_memory (rest : Algebra Word) (p : Parameters) (s : State Word) :
    (execution rest p s).val.mem = (rawWritten p s).mem := by
  have ht := tail_memory rest p (signExecution rest p s).val (storeExecution rest p s).val
    (s.pids 0) (fun j => integerWord (laneNibble p s j))
    (signExecution_result rest p s) (signExecution_pid rest p s)
    (signExecution_n_elements_out rest p s) (storeExecution rest p s).property
  change (storeExecution rest p s).val.mem = _
  rw [ht, signExecution_mem]
  simp [rawWritten, masked_memory, converted, active, outputOffset]

/-- The output mask forces the parent byte to be an active load. Hence no
condition on any inactive undef value is needed. -/
theorem active_lane_table (p : Parameters) (s : State Word) (hbytes : ActiveBytes p s)
    (j : TileIndex [p.block*2]) (hj : active p s j) :
    converted (laneNibble p s j) = expectedValue p s j := by
  have hp := active_parent p s j hj
  have hdiv : j.1.val/2 < p.block := by have := j.1.isLt; omega
  obtain ⟨b, hb⟩ := hbytes ⟨j.1.val/2,hdiv⟩ hp
  have hloaded : loaded p s (j.1.val/2) = b.toNat := by
    simp only [loaded, hp, ↓reduceIte, hb, Cell.read_same]
  have hinit : (s.mem p.input (inputOffset p s (j.1.val/2))).read .nat = b.toNat := by
    rw [hb, Cell.read_same]
  simp only [laneNibble, hloaded, expectedValue, initialNibble, hinit, BitVec.ofNat_toNat, BitVec.setWidth_eq]
  split_ifs
  · rw [(byte_halves b).1, converted_table]
  · rw [(byte_halves b).2, converted_table]

theorem execution_memory (rest : Algebra Word) (p : Parameters) (s : State Word)
    (hbytes : ActiveBytes p s) :
    (execution rest p s).val.mem = (written p s).mem := by
  rw [raw_execution_memory]
  have h : rawWritten p s = written p s := by
    unfold rawWritten written
    congr 1
    funext t j
    by_cases hj : active p s j
    · simp only [hj, ↓reduceIte, active_lane_table p s hbytes j hj]
    · simp only [hj, ↓reduceIte]
  rw [h]

theorem residual_backend_independent (rest other : Algebra Word) (p : Parameters)
    (s : State Word) :
    (execution rest p s).val.mem = (execution other p s).val.mem := by
  rw [raw_execution_memory, raw_execution_memory]

/-- Memory results depend only on initial memory and program id, never on the
arbitrary values returned by masked-off loads. This includes out-of-range pid. -/
theorem undef_independent (rest : Algebra Word) (p : Parameters) (s : State Word)
    (u : (dtype : TileDType) → RegionName → Nat → Value Word dtype)
    (hbytes : ActiveBytes p s) :
    (execution rest p s).val.mem = (execution rest p {s with undef := u}).val.mem := by
  have hu : ActiveBytes p {s with undef := u} := hbytes
  rw [execution_memory rest p s hbytes, execution_memory rest p _ hu]
  simp [written, masked_memory, expectedValue, initialNibble, inputOffset, outputOffset, active]

#axiomsClean tail_memory
#axiomsClean undef_independent

#axiomsClean raw_execution_memory
#axiomsClean active_lane_table
#axiomsClean execution_memory
#axiomsClean residual_backend_independent
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
