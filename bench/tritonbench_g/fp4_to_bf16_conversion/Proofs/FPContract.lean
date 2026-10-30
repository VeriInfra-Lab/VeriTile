import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.FPResult
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

private theorem masked_memory {α ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (active : ι → Prop) [DecidablePred active] (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (value i) else t) s).mem =
    l.foldl (fun m i => if active i then (fun r' o => if r'=r ∧ o=off i then value i else m r' o) else m) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    simp only [List.foldl_cons]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]; rw [ih]; rfl
    · simpa only [hi, ↓reduceIte] using ih s

def rawWritten (P : Primitives) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.block*2]).foldl (fun t j =>
    if active p s j then t.write p.output (outputOffset p s j) (.mk .bf16
      (P.mulBF16 (converted (laneNibble p s j)) (scaleValue P (loadedScale p s (j.1.val/32))))) else t) s

theorem tail_memory (rest : Algebra Word) (P : Primitives) (p : Parameters) (start final : State Word)
    (pid : Nat) (values : TileIndex [p.block*2] → Word)
    (hv : start.regs .bf16 [p.block*2] "output" = some values)
    (hp : start.regs .nat [] "pid" = some (fun _ => pid))
    (hc : start.regs .nat [] "n_elements_out" = some (fun _ => p.count*2))
    (h : run (backend rest P) ((source p).surfaceBody.drop 43 |>.take 4) start = some final) :
    final.mem = (TileShape.allIndices [p.block*2]).foldl (fun mem j =>
      if pid*(p.block*2)+j.1.val < p.count*2 then
        (fun r o => if r=p.output ∧ o=pid*(p.block*2)+j.1.val then .mk .bf16 (values j) else mem r o)
      else mem) start.mem := by
  dsimp only [source, scaledCandidate, ComputeKernel.surfaceBody, List.drop, List.take, offsetCode, List.append] at h
  simp [run, step, evalExpr, evalComputeOp, FP.Structural.evalOp, store,
    backend, Fp4ToBf16.FPValue.backend, hv, hp, hc, ComputeDType.eraseDType,
    FP.Structural.numeric, FP.Structural.natLt, FP.Structural.ofFloat, FP.Structural.toFloat,
    Option.bind, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast] at h
  cases h
  simp [masked_memory, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]

theorem raw_execution_memory (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word) :
    (execution rest P p s).val.mem = (rawWritten P p s).mem := by
  have ht := tail_memory rest P p (productExecution rest P p s).val (storeExecution rest P p s).val
    (s.pids 0) (fun j => P.mulBF16 (converted (laneNibble p s j))
      (scaleValue P (loadedScale p s (j.1.val/32))))
    (productExecution_output rest P p s) (productExecution_pid rest P p s)
    (productExecution_n_elements_out rest P p s) (storeExecution rest P p s).property
  change (storeExecution rest P p s).val.mem = _
  rw [ht, productExecution_mem]
  simp [rawWritten, masked_memory, active, outputOffset, Fp4ToBf16.FPValue.outputOffset, base]

theorem execution_memory (rest : Algebra Word) (P : Primitives) (p : Parameters) (s : State Word)
    (hbytes : ActiveBytes p s) :
    (execution rest P p s).val.mem = (written P p s).mem := by
  rw [raw_execution_memory]
  have h : rawWritten P p s = written P p s := by
    unfold rawWritten written
    congr 1
    funext t j
    by_cases hj : active p s j
    · have hb := Fp4ToBf16.FPValue.active_lane_table (base p) s hbytes j hj
      simpa only [hj, ↓reduceIte, expectedValue, byteValue] using
        congrArg (fun v => t.write p.output (outputOffset p s j)
          (.mk .bf16 (P.mulBF16 v (scaleValue P (loadedScale p s (j.1.val/32)))))) hb
    · simp only [hj, ↓reduceIte]
  rw [h]

theorem residual_backend_independent (rest other : Algebra Word) (P : Primitives)
    (p : Parameters) (s : State Word) :
    (execution rest P p s).val.mem = (execution other P p s).val.mem := by
  rw [raw_execution_memory, raw_execution_memory]

/-- This corollary is deliberately restricted to complete groups. The primary
contract above retains every count and includes the source's scale undef. -/
theorem grouped_undef_independent (rest : Algebra Word) (P : Primitives) (p : Parameters)
    (s : State Word) (u : (dtype : TileDType) → RegionName → Nat → Value Word dtype)
    (hbytes : ActiveBytes p s) (hgrouped : p.count%16=0) :
    (execution rest P p s).val.mem = (execution rest P p {s with undef := u}).val.mem := by
  have hu : ActiveBytes p {s with undef := u} := hbytes
  rw [execution_memory rest P p s hbytes, execution_memory rest P p _ hu]
  simp only [written, masked_memory]
  congr 1
  funext mem j
  have ha : active p {s with undef := u} j = active p s j := rfl
  have ho : outputOffset p {s with undef := u} j = outputOffset p s j := rfl
  by_cases hj : active p s j
  · have hs := active_scale_of_grouped p s hgrouped j hj
    have hs' : scaleOffset p {s with undef := u} (j.1.val/32) < scaleCount p := hs
    have hv : expectedValue P p s j = expectedValue P p {s with undef := u} j := by
      simp only [expectedValue, loadedScale, hs, hs', ↓reduceIte]
      rfl
    simp only [ha, ho, hj, ↓reduceIte, hv]
  · simp only [ha, ho, hj, ↓reduceIte]

#axiomsClean grouped_undef_independent

#axiomsClean tail_memory
#axiomsClean raw_execution_memory
#axiomsClean execution_memory
#axiomsClean residual_backend_independent
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16Conversion.FPValue
