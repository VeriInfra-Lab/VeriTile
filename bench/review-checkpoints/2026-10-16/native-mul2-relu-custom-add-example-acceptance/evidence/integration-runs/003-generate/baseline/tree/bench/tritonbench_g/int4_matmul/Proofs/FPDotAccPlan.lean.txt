import bench.tritonbench_g.int4_matmul.Proofs.FPDotAccSoftware
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAcc
open VeriTile.Triton FP.Structural FPTyped
open FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Explicit software instruction order. Validity is a finite index permutation,
not a floating-point law: it prevents dropping, repeating, or inventing products.
It does not assert that reordering preserves the result or models a tensor core. -/
structure FMAOrder (k : Nat) where
  indices : List (Fin k)
  permutation : indices.Perm (List.finRange k)

abbrev FMAPlan := (k : Nat) → FMAOrder k

def ascendingPlan : FMAPlan := fun k => ⟨List.finRange k, .refl _⟩
def descendingPlan : FMAPlan := fun k => ⟨(List.finRange k).reverse, List.reverse_perm _⟩

theorem order_length (plan : FMAPlan) (k : Nat) : (plan k).indices.length = k := by
  rw [(plan k).permutation.length_eq]
  simp

theorem order_each_once (plan : FMAPlan) (k : Nat) (i : Fin k) :
    (plan k).indices.count i = 1 := by
  rw [(plan k).permutation.count_eq]
  simp

def plannedFMA (plan : FMAPlan) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → Word)
    (b : TileIndex (batch ++ [k,n]) → Word)
    (c : TileIndex (batch ++ [m,n]) → Word) : TileIndex (batch ++ [m,n]) → Word :=
  match batch with
  | [] => fun i => (plan k).indices.foldl (fun acc t => FP.fma .ieeeValue
      (QuantizeKvTransform.MixedBits.widen (a (i.1,t,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.widen (b (t,i.2.1,PUnit.unit))) acc) (c i)
  | _ :: tail => fun i => plannedFMA plan (batch := tail)
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2
termination_by batch.length

def plannedDotAcc (plan : FMAPlan) : DotAcc Word := ⟨fun _ _ {batch} {m} {k} {n} a b c => plannedFMA plan a b c⟩

/-- Independent initial-memory recurrence: half dequantization, exact half
widening, one fp32 FMA per listed product, and the actual carried value. -/
def plannedAccum (plan : FMAPlan) (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step + 1 => fun i => (plan p.blockInner).indices.foldl (fun acc t => FP.fma .ieeeValue
      (QuantizeKvTransform.MixedBits.widen
        ((s.mem p.input (aOffset p s step (i.1,t,PUnit.unit))).read .fp16))
      (QuantizeKvTransform.MixedBits.widen (FPClosed.halfWeight p s step (t,i.2.1,PUnit.unit)))
      acc) (plannedAccum plan p s step i)

def plannedOutput (plan : FMAPlan) (p : Parameters) (s : State Word) (i : CIndex p) : Word :=
  QuantizeKvTransform.MixedBits.narrow .ieeeValue (plannedAccum plan p s p.steps i)

def plannedReference (plan : FMAPlan) (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then
      t.write p.output (outputOffset p s i) (.mk .fp16 (plannedOutput plan p s i)) else t) s

theorem accumulation_planned (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word) (k : Nat) :
    accumulation (FPSoftware.scalar rest) (plannedDotAcc plan) p s k = plannedAccum plan p s k := by
  induction k with
  | zero => simp [accumulation, plannedAccum, FPSoftware.scalar,
      FP.Software.scalarFP32, FP.Software.fp32Context]
  | succ k ih =>
    funext i
    simp only [accumulation, ih, FPSoftware.bValue_scalar]
    simp only [FPSoftware.scalar, FPSoftware.dequantized,
      QuantizeKvTransform.MixedBits.half_embedding]
    simp only [plannedDotAcc, plannedFMA, plannedAccum, FPClosed.halfWeight, aValue]

theorem output_planned (plan : FMAPlan) (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (FPSoftware.scalar rest) (plannedDotAcc plan) p s = plannedOutput plan p s := by
  funext i
  unfold outputValue
  rw [accumulation_planned]
  rfl

theorem planned_memory_correctness (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (FPSoftware.scalar rest) (plannedDotAcc plan)) (source p) s = some final ∧
      final.mem = (plannedReference plan p s).mem := by
  simpa only [written, output_planned, plannedReference] using
    correctness_memory (FPSoftware.scalar rest) (plannedDotAcc plan) p s

theorem planned_readback_correctness (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (backend (FPSoftware.scalar rest) (plannedDotAcc plan)) (source p) s = some final ∧
      Post p s final (plannedOutput plan p s) := by
  simpa only [output_planned] using correctness (FPSoftware.scalar rest) (plannedDotAcc plan) p s hw

theorem planned_rest_independence (plan : FMAPlan) (a b : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ t u,
      FP.Structural.exec (backend (FPSoftware.scalar a) (plannedDotAcc plan)) (source p) s = some t ∧
      FP.Structural.exec (backend (FPSoftware.scalar b) (plannedDotAcc plan)) (source p) s = some u ∧
      t.mem = u.mem := by
  obtain ⟨t,ht,hm⟩ := planned_memory_correctness plan a p s
  obtain ⟨u,hu,hn⟩ := planned_memory_correctness plan b p s
  exact ⟨t,u,ht,hu,hm.trans hn.symm⟩

theorem ascending_accum (p : Parameters) (s : State Word) (k : Nat) :
    plannedAccum ascendingPlan p s k = FPDotAccGap.carryAccum p s k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    funext i
    simp only [plannedAccum, ascendingPlan, FPDotAccGap.carryAccum, FPDotAccGap.carryBlock, ih]

theorem ascending_output (p : Parameters) (s : State Word) :
    plannedOutput ascendingPlan p s = FPDotAccGap.carryOutput p s := by
  funext i
  simp only [plannedOutput, FPDotAccGap.carryOutput, ascending_accum]

#axiomsClean order_length
#axiomsClean order_each_once
#axiomsClean accumulation_planned
#axiomsClean output_planned
#axiomsClean planned_memory_correctness
#axiomsClean planned_readback_correctness
#axiomsClean planned_rest_independence
#axiomsClean ascending_accum
#axiomsClean ascending_output
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAcc
