import bench.tritonbench_g.context_attn_fwd.Proofs.FPReferenceProgram

namespace VeriTile.Bench.Repairs.ContextAttnFwd.KernelRawGrid
open VeriTile.Triton FP.Structural FPSuccess
noncomputable section
variable {α : Type} [Inhabited α]

def run (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α) (g : Grid) :
    List (GridIndex g) → State α → Option (State α)
  | [],s => some s
  | idx::rest,s => (exec (M.withDefaultPrecision .fp32) (kernel p r .fp32 scale)
      (atIndex s idx)).bind (run p r scale M g rest)

def tile (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) {g : Grid} (idx : GridIndex g) (s : State α) : State α :=
  let launch := atIndex s idx
  scatter p r launch launch (FPReference.output p M dots scale (FPReference.inputs p r launch))

def reference (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (g : Grid) : List (GridIndex g) → State α → State α
  | [],s => s
  | idx::rest,s => reference p r scale M dots g rest (tile p r scale M dots idx s)

/-- The exact writes of the independent recurrence composition. -/
def Outside (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (g : Grid) : List (GridIndex g) → State α → RegionName → Nat → Prop
  | [],_,_,_ => True
  | idx::rest,s,rg,off =>
    (∀ i,active p r (atIndex s idx) i → rg=r.out → off=offset p r (atIndex s idx) i → False) ∧
      Outside p r scale M dots g rest (tile p r scale M dots idx s) rg off

theorem tile_congr (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) {g : Grid} (idx : GridIndex g) (s t : State α) (memory : s.mem=t.mem) :
    (tile p r scale M dots idx s).mem=(tile p r scale M dots idx t).mem := by
  have inp : FPReference.inputs p r (atIndex s idx)=FPReference.inputs p r (atIndex t idx) := by
    simp only [FPReference.inputs,atIndex,memory]
  dsimp only [tile]
  rw [inp]
  classical
  unfold scatter
  simp only [active,offset,start,length,row,atIndex,memory]
  apply masked_fold_mem_congr
  rfl

theorem reference_congr (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (g : Grid) (order : List (GridIndex g))
    (s t : State α) (memory : s.mem=t.mem) :
    (reference p r scale M dots g order s).mem=(reference p r scale M dots g order t).mem := by
  induction order generalizing s t with
  | nil => exact memory
  | cons idx rest ih => exact ih _ _ (tile_congr p r scale M dots idx s t memory)

theorem reference_frame (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (g : Grid) (order : List (GridIndex g)) (s : State α)
    (rg : RegionName) (off : Nat) (outside : Outside p r scale M dots g order s rg off) :
    (reference p r scale M dots g order s).mem rg off=s.mem rg off := by
  induction order generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    change (reference p r scale M dots g rest (tile p r scale M dots idx s)).mem rg off=_
    rw [ih _ outside.2]
    exact scatter_frame p r (atIndex s idx) (atIndex s idx) _ rg off outside.1

/-- Arbitrary raw scheduling lists execute the same complete shared source;
all output cells and the exact recurrence-write exterior frame are covered. -/
theorem complete (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : DotSupport p M) (positive : 0<p.BN) (g : Grid)
    (order : List (GridIndex g)) (s : State α) :
    ∃ final,run p r scale M g order s=some final ∧
      final.mem=(reference p r scale M dots g order s).mem ∧
      (∀ rg off,Outside p r scale M dots g order s rg off → final.mem rg off=s.mem rg off) := by
  have source : ∀ (jobs : List (GridIndex g)) (s : State α),
      ∃ final,run p r scale M g jobs s=some final ∧ final.mem=(reference p r scale M dots g jobs s).mem := by
    intro jobs
    induction jobs with
    | nil => intro s; exact ⟨s,rfl,rfl⟩
    | cons idx rest ih =>
      intro s
      obtain ⟨mid,ran,memory⟩ := FPReference.program p r scale M dots positive (atIndex s idx)
      obtain ⟨final,done,equal⟩ := ih mid
      refine ⟨final,?_,equal.trans (reference_congr p r scale M dots g rest mid (tile p r scale M dots idx s) memory)⟩
      simp only [run,ran,Option.bind_some]
      exact done
  obtain ⟨final,ran,memory⟩ := source order s
  exact ⟨final,ran,memory,fun rg off outside => by
    rw [memory]; exact reference_frame p r scale M dots g order s rg off outside⟩

#specNonCircular reference avoiding [run,FP.Structural.exec,FP.Structural.run,kernel]
#axiomsClean complete
end
end VeriTile.Bench.Repairs.ContextAttnFwd.KernelRawGrid
