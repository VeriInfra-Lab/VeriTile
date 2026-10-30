import bench.tritonbench_g.multinomial_sampling.Proofs.IOUndef

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOUndefRegression
open VeriTile.Triton FP.Structural SourceValue
noncomputable section

def p : Parameters where
  scores := "scores"
  seeds := "seeds"
  offsets := "offsets"
  indices := "indices"
  output := "out"
  strideScoreBatch := 1
  strideScoreToken := 1
  strideIndexBatch := 1
  strideIndexToken := 1
  batches := 1
  tokens := 1
  block := 2
  blockTokens := 1
  blockDomain := ⟨by decide, by decide, 1, rfl⟩
  tokensDomain := ⟨by decide, by decide, 0, rfl⟩

def vals (base : ProgramIO Nat Nat) (memory : base.Inputs)
    (seed : Row p → Int) (offset label : Int) : (IOUndef.addUndefined p 0 base).Inputs :=
  ⟨memory, Fin.cases seed
    (Fin.cases (fun _ => offset) (Fin.cases (fun _ => label) (fun u => Fin.elim0 u)))⟩

/-- A one-row batch with a two-lane block retains both signed64 endpoints:
the active memory seed and the padded nonzero fallback seed. Offsets and
labels can lie outside signed64 because the original SeedDomain does not
constrain them. There is no block-divisibility assumption. -/
theorem partial_block_extreme_seeds (base : ProgramIO Nat Nat) (memory : base.Inputs)
    (offset label : Int) :
    IOUndef.SeedDomain p 0 base
      (vals base memory (fun _ => 9223372036854775807) offset label)
      (fun _ _ => -9223372036854775808) := by
  intro i
  by_cases active : IOUndef.row p 0 i < p.batches
  · simp only [IOUndef.seed, active, ↓reduceDIte]
    norm_num [Random.IntegerType.inRange]
  · simp only [IOUndef.seed, active, ↓reduceDIte]
    change Random.IntegerType.i64.inRange 9223372036854775807 = Bool.true
    norm_num [Random.IntegerType.inRange]

theorem padded_seed_above_i64_rejected (base : ProgramIO Nat Nat) (memory : base.Inputs) :
    ¬ IOUndef.SeedDomain p 0 base
      (vals base memory (fun _ => 9223372036854775808) 0 0) (fun _ _ => 0) := by
  intro domain
  have invalid := domain ((1 : Fin 2), PUnit.unit)
  change Random.IntegerType.i64.inRange 9223372036854775808 = Bool.true at invalid
  norm_num [Random.IntegerType.inRange] at invalid

theorem padded_seed_below_i64_rejected (base : ProgramIO Nat Nat) (memory : base.Inputs) :
    ¬ IOUndef.SeedDomain p 0 base
      (vals base memory (fun _ => -9223372036854775809) 0 0) (fun _ _ => 0) := by
  intro domain
  have invalid := domain ((1 : Fin 2), PUnit.unit)
  change Random.IntegerType.i64.inRange (-9223372036854775809) = Bool.true at invalid
  norm_num [Random.IntegerType.inRange] at invalid

/-- Even the seed fallback at an active lane is irrelevant and unrestricted.
It is neither demanded by InputBinding nor included in the seed range law. -/
theorem active_fallback_unrestricted (base : ProgramIO Nat Nat) (memory : base.Inputs)
    (unrelated offset label : Int) :
    IOUndef.SeedDomain p 0 base
      (vals base memory (fun i => if i.1.val = 0 then unrelated else -13) offset label)
      (fun _ _ => -7) := by
  intro ⟨i, laneUnit⟩
  have hi : i.val < 2 := i.isLt
  obtain ⟨i, bound⟩ := i
  interval_cases i
  · change Random.IntegerType.i64.inRange (-7) = Bool.true
    norm_num [Random.IntegerType.inRange]
  · change Random.IntegerType.i64.inRange (-13) = Bool.true
    norm_num [Random.IntegerType.inRange]

/-- The original typed state domain can be recovered exactly for an arbitrary
program ID. Extracting finite logical fallbacks adds no global undef law. -/
theorem original_domain_iff (base : ProgramIO Nat Nat) (params : Parameters)
    (pid : Nat) (memory : base.Inputs) (s : State Nat) (samePid : s.pids 0 = pid)
    (memoryBound : base.InputBinding memory s)
    (activeSeed : (i : Row params) → IOUndef.row params pid i < params.batches → Int)
    (seedBound : ∀ i h, (s.mem params.seeds (IOUndef.row params pid i)).read .int =
      activeSeed i h) :
    SourceValue.SeedDomain params s ↔ IOUndef.SeedDomain params pid base
      (IOUndef.extract params pid base memory s) activeSeed :=
  IOUndef.seed_domain_iff params pid base _ s samePid
    (IOUndef.extract_binding params pid base memory s memoryBound).2 activeSeed seedBound

#axiomsClean partial_block_extreme_seeds
#axiomsClean padded_seed_above_i64_rejected
#axiomsClean padded_seed_below_i64_rejected
#axiomsClean active_fallback_unrestricted
#axiomsClean original_domain_iff
end
end VeriTile.Bench.Repairs.MultinomialSampling.IOUndefRegression
