import bench.tritonbench_g.multinomial_sampling.FPCorrect
namespace VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
noncomputable section

def witnessParameters : Parameters :=
  { scores := "scores", seeds := "seeds", offsets := "offsets", indices := "indices", output := "indices"
    strideScoreBatch := 3, strideScoreToken := 1, strideIndexBatch := 3, strideIndexToken := 1
    batches := 1, tokens := 3, block := 2, blockTokens := 2
    blockDomain := ⟨by decide,by decide,⟨1,by decide⟩⟩
    tokensDomain := ⟨by decide,by decide,⟨1,by decide⟩⟩ }
def witnessState : State Word :=
  { mem := fun r o =>
      if r="scores" then .mk .real (if o=0 then 0x3d63666a else if o=1 then 0 else if o=2 then 0x3e800000 else 0x7fc00000)
      else if r="seeds" then .mk .int 0
      else if r="offsets" then .mk .int 1
      else if r="indices" then .mk .int (if o=0 then 41 else if o=1 then 17 else 99)
      else .mk .nat 901
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => defaultValue d }
def row0 : Row witnessParameters := (⟨0,by decide⟩,PUnit.unit)

theorem witness_seed_domain : SeedDomain witnessParameters witnessState := by
  intro i
  rcases i with ⟨i,⟨⟩⟩
  fin_cases i <;> decide +kernel

/-- The first loop iteration selects index41; rounded subtraction makes the
second interval overlap, so the second iteration replaces it with index99.
The final token is a partial block and the second batch row is inactive. -/
theorem overlap_values :
    (rounds (software .fp32 ascendingPlans) .fp32 witnessParameters witnessState 1).output row0 = 41 ∧
    (rounds (software .fp32 ascendingPlans) .fp32 witnessParameters witnessState 2).output row0 = 99 := by
  decide +kernel

/-- Complete nonempty source execution, with two loop iterations, both tails,
real Philox, input-index/output aliasing, and frame. This supporting example
does not replace the fully parameterized public contract. -/
theorem alias_overlap_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest (software .fp32 ascendingPlans))
        (source .fp32 witnessParameters) witnessState=some final ∧
      final.mem "indices" 0=.mk .int 99 ∧
      final.mem "indices" 1=.mk .int 17 ∧
      final.mem "indices" 2=.mk .int 99 ∧
      final.mem "scores" 3=.mk .real (0x7fc00000 : Word) ∧
      final.mem "other" 0=.mk .nat 901 := by
  obtain ⟨final,he,hm,hv,hf⟩ := FPCorrect.multinomial_full_software_correctness
    rest .fp32 ascendingPlans witnessParameters witnessState witness_seed_domain
  refine ⟨final,he,?_,?_,?_,?_,?_⟩
  · have h := hv row0 (by decide)
    change final.mem "indices" 0=.mk .int ((rounds (software .fp32 ascendingPlans)
      .fp32 witnessParameters witnessState 2).output row0) at h
    rw [overlap_values.2] at h
    exact h
  · have h := hf "indices" 1 (Or.inr (by
      intro i hi
      simp only [active,row,witnessParameters,witnessState,Nat.zero_mul,Nat.zero_add] at hi ⊢
      omega))
    exact h
  · have h := hf "indices" 2 (Or.inr (by
      intro i hi
      simp only [active,row,witnessParameters,witnessState,Nat.zero_mul,Nat.zero_add] at hi ⊢
      omega))
    exact h
  · exact hf "scores" 3 (Or.inl (by decide))
  · exact hf "other" 0 (Or.inl (by decide))

#axiomsClean witness_seed_domain
#axiomsClean overlap_values
#axiomsClean alias_overlap_execution
end
end VeriTile.Bench.TritonBenchG.MultinomialSampling.FPValue
