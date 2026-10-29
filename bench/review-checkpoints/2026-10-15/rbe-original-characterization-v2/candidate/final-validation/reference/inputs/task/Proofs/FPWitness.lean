import bench.tritonbench_g.rbe_triton_transform.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.RbeTritonTransform.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000

/-- One genuine pair, nonzero token position, DIM=128 from the Python helper.
This demonstrates nonempty backend execution, not a native/software bit profile. -/
def witnessParameters : Parameters :=
  ⟨"input", "output", 1, 2, 2, 2, 1, 2, 2, 1, 1, 10000, 128, 1, 2⟩

def witnessInitial {α : Type} [Inhabited α] (a b sentinel : α) : State α where
  mem := fun r o => .mk .fp16
    (if r = ("input" : RegionName) then if o = 0 then a else b else sentinel)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : Index witnessParameters := (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)

theorem witness_apart {α : Type} [Inhabited α] (a b sentinel : α) :
    WritesApart witnessParameters (witnessInitial a b sentinel) := by
  constructor
  · intro imag i j _ _ _
    exact Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) _ _
  · intro i j _ _
    have hi := Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) i lane
    have hj := Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) j lane
    subst i
    subst j
    simp [outputOffset, row, column, parts, witnessParameters, witnessInitial, lane]

theorem nonempty_backend {α : Type} [Inhabited α] (M : Algebra α) (a b sentinel : α) :
    ∃ final, FP.Structural.exec M (source witnessParameters) (witnessInitial a b sentinel) = some final ∧
      final.mem "output" 0 = .mk .fp16 (value M witnessParameters (witnessInitial a b sentinel) Bool.false lane) ∧
      final.mem "output" 1 = .mk .fp16 (value M witnessParameters (witnessInitial a b sentinel) Bool.true lane) ∧
      final.mem "input" 0 = .mk .fp16 a ∧ final.mem "output" 2 = .mk .fp16 sentinel := by
  obtain ⟨final, he, hp⟩ := correctness M witnessParameters (witnessInitial a b sentinel) (witness_apart a b sentinel)
  have hr : active witnessParameters (witnessInitial a b sentinel) Bool.false lane := by
    simp [active, row, column, parts, witnessParameters, witnessInitial, lane]
  have hi : active witnessParameters (witnessInitial a b sentinel) Bool.true lane := by
    simp [active, row, column, parts, witnessParameters, witnessInitial, lane]
  refine ⟨final, he, ?_, ?_, ?_, ?_⟩
  · simpa [outputOffset, row, column, parts, witnessParameters, witnessInitial, lane] using hp.1 Bool.false lane hr
  · simpa [outputOffset, row, column, parts, witnessParameters, witnessInitial, lane] using hp.1 Bool.true lane hi
  · have hf := hp.2 "input" 0 (Or.inl (by decide))
    simpa [witnessInitial] using hf
  · have hf := hp.2 "output" 2 (Or.inr (by
      intro imag i _
      have hi := Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) i lane
      subst i
      cases imag <;> simp [outputOffset, row, column, parts, witnessParameters, witnessInitial, lane]))
    simpa [witnessInitial] using hf
#axiomsClean nonempty_backend
end
end VeriTile.Bench.TritonBenchG.RbeTritonTransform.FPTyped
