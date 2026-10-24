import bench.tritonbench_g.int4_matmul.Proofs.FPDotAccComplete
import bench.tritonbench_g.int4_matmul.Proofs.FPDotAccGap
namespace VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAcc
open VeriTile.Triton FP.Structural FPTyped
open FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

/-- Explicit ascending-K fp32 FMA schedule seeded by the *live* accumulator.
Operand widening reads binary16 bits, and each FMA rounds once to binary32.
This is a software schedule, not a claimed tensor-core implementation. -/
def sequentialFMA {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch ++ [m,k]) → Word)
    (b : TileIndex (batch ++ [k,n]) → Word)
    (c : TileIndex (batch ++ [m,n]) → Word) : TileIndex (batch ++ [m,n]) → Word :=
  match batch with
  | [] => fun i => (List.finRange k).foldl (fun acc t => FP.fma .ieeeValue
      (QuantizeKvTransform.MixedBits.widen (a (i.1,t,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.widen (b (t,i.2.1,PUnit.unit))) acc) (c i)
  | _ :: tail => fun i => sequentialFMA (batch := tail)
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2
termination_by batch.length

def softwareDotAcc : DotAcc Word := ⟨fun _ _ {batch} {m} {k} {n} a b c => sequentialFMA a b c⟩

def softwareBackend : Algebra Word := backend (FPSoftware.scalar FPClosed.unused) softwareDotAcc

/-- The independent recurrence was already used to exhibit a concrete
failure of the source dot+add replacement law. Here it becomes the proven
output of the new explicit-carry graph, at arbitrary dimensions/loop count. -/
theorem accumulation_software (rest : Algebra Word) (p : Parameters) (s : State Word) (k : Nat) :
    accumulation (FPSoftware.scalar rest) softwareDotAcc p s k = FPDotAccGap.carryAccum p s k := by
  induction k with
  | zero => simp [accumulation, FPDotAccGap.carryAccum, FPSoftware.scalar,
      FP.Software.scalarFP32, FP.Software.fp32Context]
  | succ k ih =>
    funext i
    simp only [accumulation, ih, FPSoftware.bValue_scalar]
    simp only [FPSoftware.scalar, FPSoftware.dequantized,
      QuantizeKvTransform.MixedBits.half_embedding]
    simp only [softwareDotAcc, sequentialFMA, FPDotAccGap.carryAccum,
      FPDotAccGap.carryBlock, FPClosed.halfWeight, aValue]

theorem output_software (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (FPSoftware.scalar rest) softwareDotAcc p s = FPDotAccGap.carryOutput p s := by
  funext i
  unfold outputValue
  rw [accumulation_software]
  rfl

/-- No output assumption, associativity axiom, finite-value premise, or
numerical callback. Every unused backend produces the same complete memory. -/
def reference (p : Parameters) (s : State Word) : State Word :=
  (TileShape.allIndices [p.blockRows,p.blockColumns]).foldl
    (fun t i => if active p s i then
      t.write p.output (outputOffset p s i) (.mk .fp16 (FPDotAccGap.carryOutput p s i)) else t) s

theorem correctness_memory_software (rest : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (FPSoftware.scalar rest) softwareDotAcc) (source p) s = some final ∧
      final.mem = (reference p s).mem := by
  simpa only [written, output_software, reference] using
    correctness_memory (FPSoftware.scalar rest) softwareDotAcc p s

theorem all_memory_independent_of_unused (a b : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ t u,
      FP.Structural.exec (backend (FPSoftware.scalar a) softwareDotAcc) (source p) s = some t ∧
      FP.Structural.exec (backend (FPSoftware.scalar b) softwareDotAcc) (source p) s = some u ∧
      t.mem = u.mem := by
  obtain ⟨t,ht,hm⟩ := correctness_memory_software a p s
  obtain ⟨u,hu,hn⟩ := correctness_memory_software b p s
  exact ⟨t,u,ht,hu,hm.trans hn.symm⟩

/-- Original accepted complete-loop premises, without new numerical laws or
configuration restrictions. The graph identity is explicitly different. -/
theorem correctness_software (rest : Algebra Word) (p : Parameters) (s : State Word)
    (hTrip : p.inner = p.blockInner * p.steps) (hSplit : s.pids 1 = 0)
    (hInputs : PackedInputs p s)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j) :
    ∃ final, FP.Structural.exec (backend (FPSoftware.scalar rest) softwareDotAcc) (source p) s = some final ∧
      Post p s final (FPDotAccGap.carryOutput p s) ∧
      ∀ step, step < p.steps → ∀ i : BIndex p, ∃ word zero : BitVec 32,
        s.mem p.weight (bOffset p s step i) = .mk .int word.toInt ∧
        s.mem p.zero (zeroOffset p s step i) = .mk .int zero.toInt ∧
        difference p s step i = packedDifference p s i word zero := by
  obtain ⟨final,he,hp⟩ := correctness (FPSoftware.scalar rest) softwareDotAcc p s hw
  refine ⟨final,he,?_,?_⟩
  · simpa only [output_software] using hp
  · exact packed_difference_bridge p s hInputs

#axiomsClean accumulation_software
#axiomsClean output_software
#axiomsClean correctness_memory_software
#axiomsClean all_memory_independent_of_unused
#axiomsClean correctness_software
end
end VeriTile.Bench.TritonBenchG.Int4Matmul.FPDotAcc
