import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPFusedComplete
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word bounded_integer_cast)
open FPCompiled (FMAPlan plannedDotAcc plannedFMA)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The zero-scale product remains separately rounded before the fused
weight-scale-minus-zero instruction. This is a total bit operation, including
NaNs, infinities, subnormals and both signs of zero. -/
def fusedWeight (p : Parameters) (s : State Word) (step : Nat) (i : BIndex p) : Word :=
  QuantizeKvTransform.MixedBits.embed (FPCompiled.HalfFMA.fused
    (weightNibble p s step i) (zeroNibble p s step (i.2.1,PUnit.unit))
    (QuantizeKvTransform.MixedBits.half (scaleValue p s step (i.2.1,PUnit.unit))))

theorem zeroValue_scalar (rest : Algebra Word) (p : Parameters) (s : State Word)
    (step : Nat) (i : TileIndex [p.blockColumns]) :
    zeroValue (FPSoftware.scalar rest) p s step i =
      QuantizeKvTransform.MixedBits.embed (FPSoftware.scaledZero p s step i) := by
  have hz := zeroNibble_bounds p s step i
  simp only [zeroValue,FPSoftware.scalar,Int4Matmul.FPSoftware.scalar]
  simp only [or_true,true_or,and_true,↓reduceIte]
  rw [bounded_integer_cast _ (by omega)]
  simp [FPSoftware.scaledZero,QuantizeKvTransform.MixedBits.half_embedding]

theorem bValue_scalar (rest : Algebra Word) (p : Parameters) (s : State Word)
    (step : Nat) (i : BIndex p) :
    bValue (FPSoftware.scalar rest) p s step i = fusedWeight p s step i := by
  rw [bValue,zeroValue_scalar]
  simp only [QuantizeKvTransform.MixedBits.half_embedding,fusedWeight,
    FPCompiled.HalfFMA.fused,FPCompiled.HalfFMA.zeroProduct,FPSoftware.scaledZero]

/-- Independent initial-memory bit recurrence for the fusion-enabled graph.
The dot always starts from the live accumulator and traverses every K lane
once in the selected explicit software order; no padding product is dropped. -/
def plannedAccum (plan : FMAPlan) (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step+1 => fun i => (plan p.blockInner).indices.foldl (fun acc t => FP.fma .ieeeValue
      (QuantizeKvTransform.MixedBits.widen (FPSoftware.machineA p s step (i.1,t,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.widen (fusedWeight p s step (t,i.2.1,PUnit.unit))) acc)
      (plannedAccum plan p s step i)

def plannedOutput (plan : FMAPlan) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (plannedAccum plan p s (steps p) i)

def plannedReference (plan : FMAPlan) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then
      t.write p.output (outputOffset p s i) (.mk .fp16 (plannedOutput plan p s i)) else t) s

theorem accumulation_planned (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word) (k : Nat) :
    accumulation (FPSoftware.scalar rest) (plannedDotAcc plan) p s k = plannedAccum plan p s k := by
  induction k with
  | zero => simp [accumulation,plannedAccum,FPSoftware.scalar,Int4Matmul.FPSoftware.scalar,
      FP.Software.scalarFP32,FP.Software.fp32Context]
  | succ k ih =>
    funext i
    simp only [accumulation,ih,bValue_scalar,FPSoftware.aValue_scalar]
    simp only [FPSoftware.scalar,Int4Matmul.FPSoftware.scalar,fusedWeight,
      QuantizeKvTransform.MixedBits.half_embedding]
    simp only [plannedDotAcc,plannedFMA,plannedAccum,fusedWeight]

theorem output_planned (plan : FMAPlan) (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (FPSoftware.scalar rest) (plannedDotAcc plan) p s = plannedOutput plan p s := by
  funext i
  unfold outputValue
  rw [accumulation_planned]
  rfl

theorem planned_memory_correctness (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word) :
    ∃ final, exec (FPSoftware.scalar rest) (plannedDotAcc plan) p s = some final ∧
      final.mem = (plannedReference plan p s).mem := by
  simpa only [written,output_planned,plannedReference] using
    correctness_memory (FPSoftware.scalar rest) (plannedDotAcc plan) p s

theorem planned_readback_correctness (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i=j) :
    ∃ final, exec (FPSoftware.scalar rest) (plannedDotAcc plan) p s = some final ∧
      FPCompiled.Post p s final (plannedOutput plan p s) := by
  simpa only [output_planned] using correctness (FPSoftware.scalar rest) (plannedDotAcc plan) p s hw

theorem planned_rest_independence (plan : FMAPlan) (a b : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ t u, exec (FPSoftware.scalar a) (plannedDotAcc plan) p s=some t ∧
      exec (FPSoftware.scalar b) (plannedDotAcc plan) p s=some u ∧ t.mem=u.mem := by
  obtain ⟨t,ht,hm⟩ := planned_memory_correctness plan a p s
  obtain ⟨u,hu,hn⟩ := planned_memory_correctness plan b p s
  exact ⟨t,u,ht,hu,hm.trans hn.symm⟩

#axiomsClean zeroValue_scalar
#axiomsClean bValue_scalar
#axiomsClean accumulation_planned
#axiomsClean output_planned
#axiomsClean planned_memory_correctness
#axiomsClean planned_readback_correctness
#axiomsClean planned_rest_independence
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPFused
