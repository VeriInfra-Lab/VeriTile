import bench.repairs.tritonbench_g.multinomial_sampling.IO

namespace VeriTile.Bench.Repairs.MultinomialSampling.IOUndef
open VeriTile.Triton FP.Structural SourceValue
noncomputable section

variable {α β : Type} (p : Parameters) (pid : Nat) (base : ProgramIO α β)

theorem undef_seed (vals : (addUndefined p pid base).Inputs) (s : State β)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (i : Row p) (inactive : ¬ row p pid i < p.batches) :
    s.undef .int p.seeds (row p pid i) = vals.undefined (0 : Fin 3) i :=
  bound (0 : Fin 3) i inactive

theorem undef_offset (vals : (addUndefined p pid base).Inputs) (s : State β)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (i : Row p) (inactive : ¬ row p pid i < p.batches) :
    s.undef .int p.offsets (row p pid i) = vals.undefined (1 : Fin 3) i :=
  bound (1 : Fin 3) i inactive

theorem undef_label (vals : (addUndefined p pid base).Inputs) (s : State β)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (i : Row p) (inactive : ¬ row p pid i < p.batches) :
    s.undef .int p.indices (row p pid i * p.strideIndexBatch) =
      vals.undefined (2 : Fin 3) i := bound (2 : Fin 3) i inactive

variable [Inhabited β]

theorem seed_value (vals : (addUndefined p pid base).Inputs) (s : State β)
    (samePid : s.pids 0 = pid)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (activeSeed : (i : Row p) → row p pid i < p.batches → Int)
    (loads : ∀ i h, (s.mem p.seeds (row p pid i)).read .int = activeSeed i h)
    (i : Row p) :
    SourceValue.seedValue p s i = seed p pid base vals activeSeed i := by
  unfold SourceValue.seedValue SourceValue.intLoad SourceValue.active SourceValue.row
  simp only [samePid]
  change (if decide (row p pid i < p.batches) then _ else _) = _
  by_cases active : row p pid i < p.batches
  · simpa only [seed, active, decide_true, ↓reduceIte] using loads i active
  · simpa only [seed, active, decide_false, Bool.false_eq_true, ↓reduceIte] using
      undef_seed p pid base vals s bound i active

/-- Both directions are proved: introducing explicit logical fallback inputs
neither weakens nor narrows the source theorem's original seed domain. -/
theorem seed_domain_iff (vals : (addUndefined p pid base).Inputs) (s : State β)
    (samePid : s.pids 0 = pid)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (activeSeed : (i : Row p) → row p pid i < p.batches → Int)
    (loads : ∀ i h, (s.mem p.seeds (row p pid i)).read .int = activeSeed i h) :
    SourceValue.SeedDomain p s ↔ SeedDomain p pid base vals activeSeed := by
  unfold SourceValue.SeedDomain SeedDomain
  simp_rw [seed_value p pid base vals s samePid bound activeSeed loads]

theorem offset_value (vals : (addUndefined p pid base).Inputs) (s : State β)
    (samePid : s.pids 0 = pid)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (activeOffset : (i : Row p) → row p pid i < p.batches → Int)
    (loads : ∀ i h, (s.mem p.offsets (row p pid i)).read .int = activeOffset i h)
    (i : Row p) :
    SourceValue.offsetValue p s i = offset p pid base vals activeOffset i := by
  unfold SourceValue.offsetValue SourceValue.intLoad SourceValue.active SourceValue.row
  simp only [samePid]
  change (BitVec.ofInt 32 (if decide (row p pid i < p.batches) then _ else _)).toInt = _
  by_cases active : row p pid i < p.batches
  · simpa only [offset, active, decide_true, ↓reduceIte] using
      congrArg (fun x : Int => (BitVec.ofInt 32 x).toInt) (loads i active)
  · simpa only [offset, active, decide_false, Bool.false_eq_true, ↓reduceIte] using
      congrArg (fun x : Int => (BitVec.ofInt 32 x).toInt)
        (undef_offset p pid base vals s bound i active)

theorem initial_label (vals : (addUndefined p pid base).Inputs) (s : State β)
    (samePid : s.pids 0 = pid)
    (bound : (addUndefined p pid base).UndefBinding vals s)
    (activeLabel : (i : Row p) → row p pid i < p.batches → Int)
    (loads : ∀ i h, (s.mem p.indices (row p pid i * p.strideIndexBatch)).read .int =
      activeLabel i h) (i : Row p) :
    SourceValue.initialOutput p s i = initialLabel p pid base vals activeLabel i := by
  unfold SourceValue.initialOutput SourceValue.intLoad SourceValue.active SourceValue.row
  simp only [samePid]
  change (if decide (row p pid i < p.batches) then _ else _) = _
  by_cases active : row p pid i < p.batches
  · simpa only [initialLabel, active, decide_true, ↓reduceIte] using loads i active
  · simpa only [initialLabel, active, decide_false, Bool.false_eq_true, ↓reduceIte] using
      undef_label p pid base vals s bound i active

theorem extract_binding (memory : base.Inputs) (s : State β)
    (bound : base.InputBinding memory s) :
    (addUndefined p pid base).InputBinding (extract p pid base memory s) s := by
  refine ⟨bound, ?_⟩
  intro u i _
  obtain ⟨u, hu⟩ := u
  change u < 3 at hu
  interval_cases u <;> rfl

#axiomsClean seed_value
#axiomsClean seed_domain_iff
#axiomsClean offset_value
#axiomsClean initial_label
#axiomsClean extract_binding
end
end VeriTile.Bench.Repairs.MultinomialSampling.IOUndef
