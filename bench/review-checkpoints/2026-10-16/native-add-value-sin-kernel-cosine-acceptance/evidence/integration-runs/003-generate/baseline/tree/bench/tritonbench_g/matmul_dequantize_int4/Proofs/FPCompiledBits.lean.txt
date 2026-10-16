import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPCompiledComplete
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPHalfFMA
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
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


/-- No unsupported primitive is asserted correct. The full output theorem
below is independent of this unused fallback. -/
def unused : Algebra Word where
  literal := fun _ _ _ => FP.Value.zero .fp32
  negInf := FP.Value.zero .fp32
  binary := fun _ _ _ _ _ => FP.Value.zero .fp32
  unary := fun _ _ _ => FP.Value.zero .fp32
  cast := fun _ _ _ x => x
  fromNat := fun _ _ => FP.Value.zero .fp32
  fromInt := fun _ _ => FP.Value.zero .fp32
  fp32Bits := fun _ => FP.Value.zero .fp32
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => FP.Value.zero .fp32
  reduceSum := fun _ {_} _ _ _ _ => FP.Value.zero .fp32

/-- Independent concrete initial-memory recurrence. The source's A row mask
is kept; there is no invented K mask. Dequantization retains two rounded
half products and a half subtraction for this fusion-disabled graph. -/
def plannedAccum (plan : FMAPlan) (p : Parameters) (s : State Word) : Nat → CIndex p → Word
  | 0 => fun _ => FP.Value.zero .fp32
  | step + 1 => fun i => (plan p.blockInner).indices.foldl (fun acc t => FP.fma .ieeeValue
      (QuantizeKvTransform.MixedBits.widen (FPSoftware.machineA p s step (i.1,t,PUnit.unit)))
      (QuantizeKvTransform.MixedBits.widen (FPSoftware.dequantized p s step (t,i.2.1,PUnit.unit)))
      acc) (plannedAccum plan p s step i)

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
    simp only [accumulation,ih,FPSoftware.bValue_scalar,FPSoftware.aValue_scalar]
    simp only [FPSoftware.scalar,Int4Matmul.FPSoftware.scalar,FPSoftware.dequantized,
      QuantizeKvTransform.MixedBits.half_embedding]
    simp only [plannedDotAcc,plannedFMA,plannedAccum,FPSoftware.dequantized]

theorem output_planned (plan : FMAPlan) (rest : Algebra Word) (p : Parameters) (s : State Word) :
    outputValue (FPSoftware.scalar rest) (plannedDotAcc plan) p s = plannedOutput plan p s := by
  funext i
  unfold outputValue
  rw [accumulation_planned]
  rfl

theorem correctness_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial = some final ∧
      final.mem = (written M D p initial).mem := by
  let start := preludeExecution M D p initial
  obtain ⟨mid, hm, hi⟩ := loopExecution M D p initial start.val 0 (by omega)
    (preludeInvariant M D p initial)
  let finish := epilogueExecution M D p initial mid hi
  refine ⟨finish.val, ?_, epilogueMemory M D p initial mid hi⟩
  change run (backend M D) (source p).surfaceBody initial = some finish.val
  rw [source_decomposition, run_append, run_append, start.property]
  have hsingle (s : State α) (hn : s.regs .nat [] "num_pid_k" = some (fun _ => steps p)) :
      run (backend M D) [.forRangeDyn "k" (.constNat 0) (.ref .nat [] "num_pid_k") (.constNat 1) (body p)] s =
        FP.Structural.range (backend M D) "k" 0 (steps p) 1 (body p) s := by
    simp [run, step, evalOp_unfold, hn]
  simp only [Option.bind_some]
  rw [hsingle start.val (preludeInvariant M D p initial).numk, hm]
  simp only [Option.bind_some]
  exact finish.property

theorem planned_memory_correctness (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend (FPSoftware.scalar rest) (plannedDotAcc plan)) (source p) s = some final ∧
      final.mem = (plannedReference plan p s).mem := by
  simpa only [written,output_planned,plannedReference] using
    correctness_memory (FPSoftware.scalar rest) (plannedDotAcc plan) p s

theorem planned_readback_correctness (plan : FMAPlan) (rest : Algebra Word)
    (p : Parameters) (s : State Word)
    (hw : ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i=j) :
    ∃ final, FP.Structural.exec (backend (FPSoftware.scalar rest) (plannedDotAcc plan)) (source p) s = some final ∧
      Post p s final (plannedOutput plan p s) := by
  simpa only [output_planned] using correctness (FPSoftware.scalar rest) (plannedDotAcc plan) p s hw

theorem planned_rest_independence (plan : FMAPlan) (a b : Algebra Word) (p : Parameters) (s : State Word) :
    ∃ t u,
      FP.Structural.exec (backend (FPSoftware.scalar a) (plannedDotAcc plan)) (source p) s = some t ∧
      FP.Structural.exec (backend (FPSoftware.scalar b) (plannedDotAcc plan)) (source p) s = some u ∧
      t.mem=u.mem := by
  obtain ⟨t,ht,hm⟩ := planned_memory_correctness plan a p s
  obtain ⟨u,hu,hn⟩ := planned_memory_correctness plan b p s
  exact ⟨t,u,ht,hu,hm.trans hn.symm⟩

#axiomsClean order_length
#axiomsClean order_each_once
#axiomsClean accumulation_planned
#axiomsClean output_planned
#axiomsClean correctness_memory
#axiomsClean planned_memory_correctness
#axiomsClean planned_readback_correctness
#axiomsClean planned_rest_independence
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
