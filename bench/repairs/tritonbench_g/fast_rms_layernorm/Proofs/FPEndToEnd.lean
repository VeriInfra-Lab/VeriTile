import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPSequence

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPEndToEnd
open VeriTile.Triton FP.Structural
open FPGridForward (Address Heap entry)
noncomputable section
variable {α : Type} [Inhabited α]

def seed (saved upstream : Heap α) : Heap α := fun a => if a.1="dx" then upstream a else saved a
def referenceSaved (M : Algebra α) (h : Call) (initial : Heap α) (row : Nat) : Cell α :=
  FPReference.savedInv M h (FPMemory.raw h (entry initial row h.rows) "x" h.cols)
/-- Independent scalar graph, including the actual saved-statistic store cast.
This reference never calls a kernel evaluator or extracts its result. -/
def referenceSeed (M : Algebra α) (h : Call) (initial upstream : Heap α) : Heap α := fun a =>
  if a.1="dx" then upstream a else
    if a.1="r" ∧ a.2<h.rows then referenceSaved M h initial a.2 else initial a
def run (M : Algebra α) (h : Call) (initial upstream : Heap α) : Option (Heap α × Heap α) := do
  let forward ← FPGridForward.run M h initial
  let backward ← FPSequence.run M h (seed forward upstream)
  return (forward,backward)

theorem complete (M : Algebra α) (h : Call) (initial upstream : Heap α) :
    ∃ forward backward,run M h initial upstream=some (forward,backward) ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,forward ("y",row.val*h.cols+j.val)=
        FPGridForward.expected M h initial row (.inl j)) ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,backward ("dx",row.val*h.cols+j.val)=
        FPGridBackward.expected M h (referenceSeed M h initial upstream) row j) ∧
      (∀ pid : Fin (Geometry.featureCount h),∀ j : FPGridWeight.Output h pid,backward (FPGridWeight.address h pid j)=
        FPGridWeight.expected M h (referenceSeed M h initial upstream) pid j) ∧
      (∀ row : Fin h.rows,forward ("r",row.val)=referenceSaved M h initial row.val) ∧
      (∀ addr,(∀ row : Fin h.rows,¬FPGridBackward.writes h row addr) →
        (∀ pid : Fin (Geometry.featureCount h),¬FPGridWeight.writes h pid addr) →
        backward addr=seed forward upstream addr) := by
  obtain ⟨forward,hforward,fo,ff⟩:=FPGridForward.complete M h initial
  obtain ⟨backward,hbackward,bo,bw,bf⟩:=FPSequence.complete M h (seed forward upstream)
  have preserve (r : RegionName) (hr : r="x" ∨ r="w") (off : Nat) : forward (r,off)=initial (r,off) := by
    apply ff
    intro row
    rintro (⟨j,eq⟩|eq)
    · have same:=congrArg Prod.fst eq
      rcases hr with rfl|rfl <;> simp at same
    · have same:=congrArg Prod.fst eq
      rcases hr with rfl|rfl <;> simp at same
  have outsideR (off : Nat) (ho : ¬off<h.rows) : forward ("r",off)=initial ("r",off) := by
    apply ff
    intro row
    rintro (⟨j,eq⟩|eq)
    · exact (by decide : ("r":RegionName)≠"y") (congrArg Prod.fst eq)
    · have value:=congrArg Prod.snd eq
      have value' : off=row.val := value
      exact ho (value'.symm ▸ row.isLt)
  have agree (r : RegionName) (hr : r="x" ∨ r="w" ∨ r="r" ∨ r="dx") (off : Nat) :
      seed forward upstream (r,off)=referenceSeed M h initial upstream (r,off) := by
    rcases hr with rfl|rfl|rfl|rfl
    · simpa [seed,referenceSeed] using preserve "x" (Or.inl rfl) off
    · simpa [seed,referenceSeed] using preserve "w" (Or.inr rfl) off
    · by_cases active : off<h.rows
      · simpa [seed,referenceSeed,active,FPGridForward.address,FPGridForward.expected,referenceSaved]
          using fo ⟨off,active⟩ (.inr ())
      · simpa [seed,referenceSeed,active] using outsideR off active
    · simp [seed,referenceSeed]
  refine ⟨forward,backward,by simp [run,hforward,hbackward],fun row j => fo row (.inl j),?_,?_,?_,bf⟩
  · intro row j
    change backward (FPGridBackward.address h row j)=_
    rw [bo]
    apply FPGridBackward.expected_stable M h row
    rintro ⟨r,off⟩ hr
    apply agree r _ off
    rcases hr with hx|hw|hr|⟨lane,eq⟩
    · exact Or.inl hx
    · exact Or.inr (Or.inl hw)
    · exact Or.inr (Or.inr (Or.inl hr))
    · exact Or.inr (Or.inr (Or.inr (congrArg Prod.fst eq)))
  · intro pid j
    rw [bw]
    apply FPGridWeight.expected_stable M h pid
    rintro ⟨r,off⟩ hr
    apply agree r _ off
    rcases hr with hx|hg|hr
    · exact Or.inl hx
    · exact Or.inr (Or.inr (Or.inr hg))
    · exact Or.inr (Or.inr (Or.inl hr))
  · intro row
    exact fo row (.inr ())

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPEndToEnd
