import bench.tritonbench_g.matmul_triton2.IO
import bench.tritonbench_g.matmul_triton2.Proofs.KernelSafety
import bench.tritonbench_g.matmul_triton2.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.PublicIO
open VeriTile.Triton FP.Structural Revision001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 32000
variable {α : Type} [Inhabited α]

omit [Inhabited α] in
theorem initial_eq (M : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (pid : Nat) (initial : State α) :
    (IO.launch M P p pid).initial initial=GridExecution.atProgram initial pid := by
  rfl

theorem schedule_run (M : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (work : List Nat) (initial : State α) :
    runSchedule (work.map (IO.launch M P p)) initial=
      GridExecution.sequence (Core.one M P p) work initial := by
  induction work generalizing initial with
  | nil => rfl
  | cons pid work ih =>
    change (Core.one M P p pid initial).bind (runSchedule (work.map (IO.launch M P p)))=
      (Core.one M P p pid initial).bind (GridExecution.sequence (Core.one M P p) work)
    cases h : Core.one M P p pid initial <;> simp [ih]

theorem schedule_safe (M : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (hBK : 0<p.blockInner) (bounds : RegionName → Nat)
    (ha : ∀ r k, r<p.rows → k<p.innerSize → r*p.inputRow+k*p.inputInner<bounds p.input)
    (hb : ∀ k c, k<p.innerSize → c<p.columns → k*p.weightInner+c*p.weightColumn<bounds p.weight)
    (hc : ∀ r c, r<p.rows → c<p.columns → r*p.outputRow+c*p.outputColumn<bounds p.output)
    (work : List Nat) (initial : State α) :
    ScheduleSafe bounds (work.map (IO.launch M P p)) initial := by
  induction work generalizing initial with
  | nil => trivial
  | cons pid work ih =>
    refine ⟨?_,fun after _ => ih after⟩
    change TraceSafe (Core.backend M P) bounds (Core.source p)
      ((IO.launch M P p pid).initial initial)
    rw [initial_eq]
    apply KernelSafety.source_safe M P p bounds (GridExecution.atProgram initial pid) hBK
    · intro q i hr hk
      have bound := ha (Core.row p (GridExecution.atProgram initial pid) i.1)
        (q*p.blockInner+i.2.1.val) hr (by dsimp only [Core.inner] at hk; omega)
      convert bound using 1
      simp only [Core.aOffset]
      ring
    · intro q i hk hc
      have bound := hb (q*p.blockInner+i.1.val)
        (Core.column p (GridExecution.atProgram initial pid) i.2.1)
        (by dsimp only [Core.inner] at hk; omega) hc
      convert bound using 1
      simp only [Core.bOffset]
      ring
    · intro i hi
      exact hc _ _ hi.1 hi.2

/-- Compose the exact prior complete-grid execution with shared IO binding and
the separately proved actual-source access safety. Formula lowering is an
internal proof obligation, discharged for each public independent formula. -/
theorem implements (M : Algebra α) (P : Core.Primitives α) (p : Core.Parameters)
    (geometry : Geometry.Valid (Core.geometry p)) (hBK : 0<p.blockInner)
    (input : p.input≠p.output) (weight : p.weight≠p.output)
    (f : (Nat → Nat → α) → (Nat → Nat → α) → Nat → Nat → α)
    (formula : ∀ (initial : State α) (a b : Nat → Nat → α),
      (∀ r k, r<p.rows → k<p.innerSize →
        (initial.mem p.input (r*p.inputRow+k*p.inputInner)).read .real=a r k) →
      (∀ k c, k<p.innerSize → c<p.columns →
        (initial.mem p.weight (k*p.weightInner+c*p.weightColumn)).read .real=b k c) →
      ∀ r c, r<p.rows → c<p.columns → Core.logical P p initial r c=f a b r c) :
    (IO.matmul M P p).Implements (IO.formula p f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    apply schedule_safe M P p hBK bounds
    · intro r k hr hk
      exact bounded.2.1 (0 : Fin 2) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    · intro k c hk hc
      exact bounded.2.1 (1 : Fin 2) (⟨k,hk⟩,⟨c,hc⟩,PUnit.unit) trivial
    · intro r c hr hc
      exact bounded.2.2.1 (0 : Fin 1) (⟨r,hr⟩,⟨c,hc⟩,PUnit.unit) trivial
  · intro vals initial bound
    have ha : ∀ r k, r<p.rows → k<p.innerSize →
        (initial.mem p.input (r*p.inputRow+k*p.inputInner)).read .real=
          IO.matrix (vals (0 : Fin 2)) r k := by
      intro r k hr hk
      simpa [ProgramIO.InputBinding,IO.matmul,IO.inputs,IO.matrix,hr,hk,mapValue] using
        bound (0 : Fin 2) (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit) trivial
    have hb : ∀ k c, k<p.innerSize → c<p.columns →
        (initial.mem p.weight (k*p.weightInner+c*p.weightColumn)).read .real=
          IO.matrix (vals (1 : Fin 2)) k c := by
      intro k c hk hc
      simpa [ProgramIO.InputBinding,IO.matmul,IO.inputs,IO.matrix,hk,hc,mapValue] using
        bound (1 : Fin 2) (⟨k,hk⟩,⟨c,hc⟩,PUnit.unit) trivial
    obtain ⟨final,ran,output,frame⟩ := Core.grid_correct M P p geometry hBK input weight initial
    refine ⟨final,?_,?_,?_⟩
    · exact (schedule_run M P p _ initial).trans ran
    · intro port lane _
      fin_cases port
      have value := output (lane.1,lane.2.1) trivial
      dsimp only [Geometry.address,Core.geometry] at value
      rw [formula initial _ _ ha hb _ _ lane.1.isLt lane.2.1.isLt] at value
      change mapCell id (final.mem p.output
          (lane.1.val*p.outputRow+lane.2.1.val*p.outputColumn))=
        .mk .real (f (IO.matrix (vals (0 : Fin 2))) (IO.matrix (vals (1 : Fin 2)))
          lane.1.val lane.2.1.val)
      rw [value]
      rfl
    · intro region offset outside
      apply frame region offset
      by_cases other : region≠p.output
      · exact Or.inl other
      · right
        intro x _
        have miss := outside.1 (0 : Fin 1) (x.1,x.2,PUnit.unit) trivial
        exact miss.resolve_left other

#axiomsClean schedule_run
#axiomsClean schedule_safe
#axiomsClean implements
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.PublicIO
