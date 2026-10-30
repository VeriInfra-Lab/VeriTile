import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPFusedBits
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiledWitness
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Two K iterations and one output. These deliberately use the original
arbitrary-Nat semantic scope, not an invented valid hardware tile restriction. -/
def parameters (mode : Bool) : Parameters :=
  ⟨"input","output","scale","weight","zero",1,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,mode⟩

def initial : State Word where
  mem := fun r off => if r = ("weight" : RegionName) then .mk .int (3 : BitVec 32).toInt
    else if r = ("zero" : RegionName) then .mk .int (2 : BitVec 32).toInt
    else if r = ("input" : RegionName) then
      .mk .fp16 (QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0x3c00))
    else if r = ("scale" : RegionName) then
      .mk .fp16 (QuantizeKvTransform.MixedBits.embed
        (MatmulDequantInt4.Binary16.Value.ofBits (if off=0 then 0x3c01 else 0x3c00)))
    else .mk .fp16 (QuantizeKvTransform.MixedBits.embed MatmulDequantInt4.Binary16.Value.zero)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane (mode : Bool) : CIndex (parameters mode) := (⟨0,by change 0<1; decide⟩,⟨0,by change 0<1; decide⟩,PUnit.unit)

def separateResult (mode : Bool) : Word := QuantizeKvTransform.MixedBits.embed
  (MatmulDequantInt4.Binary16.Value.ofBits (if mode then 0x4002 else 0x4001))
def fusedResult (mode : Bool) : Word := QuantizeKvTransform.MixedBits.embed
  (MatmulDequantInt4.Binary16.Value.ofBits (if mode then 0x4001 else 0x4000))

theorem distinct_full_loop_values (mode : Bool) :
    FPCompiled.plannedOutput FPCompiled.ascendingPlan (parameters mode) initial (lane mode) = separateResult mode ∧
    FPFused.plannedOutput FPCompiled.ascendingPlan (parameters mode) initial (lane mode) = fusedResult mode ∧
    separateResult mode ≠ fusedResult mode := by
  cases mode <;> decide +kernel

theorem original_packed_scope (mode : Bool) : PackedInputs (parameters mode) initial := by
  intro k hk i
  exact ⟨⟨3,by simp [initial,parameters]⟩,⟨2,by simp [initial,parameters]⟩⟩

theorem layout (mode : Bool) :
    ∀ i j, active (parameters mode) initial i → active (parameters mode) initial j →
      outputOffset (parameters mode) initial i=outputOffset (parameters mode) initial j → i=j := by
  intro i j _ _ _
  change (i : Fin 1 × Fin 1 × PUnit) = j
  exact @Subsingleton.elim (Fin 1 × Fin 1 × PUnit) inferInstance i j

theorem full_source_success (rest : Algebra Word) (mode : Bool) :
    ∃ t u,
      FP.Structural.exec (FPCompiled.backend (FPSoftware.scalar rest)
        (FPCompiled.plannedDotAcc FPCompiled.ascendingPlan)) (FPCompiled.source (parameters mode)) initial = some t ∧
      FPFused.exec (FPSoftware.scalar rest) (FPCompiled.plannedDotAcc FPCompiled.ascendingPlan)
        (parameters mode) initial = some u ∧
      t.mem "output" 0 = .mk .fp16 (separateResult mode) ∧
      u.mem "output" 0 = .mk .fp16 (fusedResult mode) ∧
      t.mem "weight" 0 = initial.mem "weight" 0 ∧
      u.mem "weight" 0 = initial.mem "weight" 0 := by
  obtain ⟨t,ht,hp⟩ := FPCompiled.planned_readback_correctness FPCompiled.ascendingPlan rest
    (parameters mode) initial (layout mode)
  obtain ⟨u,hu,hq⟩ := FPFused.planned_readback_correctness FPCompiled.ascendingPlan rest
    (parameters mode) initial (layout mode)
  have ha : active (parameters mode) initial (lane mode) := by cases mode <;> decide
  have ho : outputOffset (parameters mode) initial (lane mode) = 0 := by cases mode <;> decide
  refine ⟨t,u,ht,hu,?_,?_,?_,?_⟩
  · simpa only [ho,(distinct_full_loop_values mode).1] using hp.1 (lane mode) ha
  · simpa only [ho,(distinct_full_loop_values mode).2.1] using hq.1 (lane mode) ha
  · exact hp.2 _ _ (Or.inl (by change ("weight" : RegionName) ≠ "output"; decide))
  · exact hq.2 _ _ (Or.inl (by change ("weight" : RegionName) ≠ "output"; decide))

#axiomsClean distinct_full_loop_values
#axiomsClean original_packed_scope
#axiomsClean layout
#axiomsClean full_source_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiledWitness
