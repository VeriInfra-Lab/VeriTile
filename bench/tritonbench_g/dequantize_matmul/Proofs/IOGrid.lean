import bench.tritonbench_g.dequantize_matmul.IO
import bench.tritonbench_g.dequantize_matmul.Proofs.AccessSafety
import bench.tritonbench_g.dequantize_matmul.Proofs.KernelWhole

namespace VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO
open VeriTile.Triton FP.Structural KernelSource KernelResult
noncomputable section
variable {α : Type} [Inhabited α]

theorem initial_eq (M : Algebra α) (p : Parameters) (version : Bool)
    (pid : Program) (s : State α) :
    (invocation M p version pid).initial s = KernelGrid.atProgram s pid := by
  simp only [Invocation.initial, invocation, bindRegisters, List.foldl_nil,
    Bool.false_eq_true, if_false, Option.getD_none, KernelGrid.atProgram]
  congr 1
  funext d
  by_cases h0 : d = 0
  · subst d; rfl
  by_cases h1 : d = 1
  · subst d; rfl
  have hd : ¬d < 2 := by omega
  simp [h0, h1, hd]

theorem invocation_run (M : Algebra α) (p : Parameters) (version : Bool)
    (pid : Program) (s : State α) :
    (invocation M p version pid).run s =
      KernelExecution.exec M p version (KernelGrid.atProgram s pid) := by
  unfold Invocation.run
  rw [initial_eq]
  rfl

theorem schedule_eq (M : Algebra α) (p : Parameters) (version : Bool)
    (jobs : List Program) (s : State α) :
    runSchedule (jobs.map (invocation M p version)) s = KernelGrid.run M p version jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid jobs ih =>
    simp only [List.map_cons, runSchedule, invocation_run, KernelGrid.run]
    cases h : KernelExecution.exec M p version (KernelGrid.atProgram s pid) <;> simp [ih]

theorem schedule_safe (M : Algebra α) (p : Parameters) (version : Bool)
    (bounds : RegionName → Nat) (values : Inputs α p)
    (hb : (sourceIO M p version).InBounds bounds values)
    (jobs : List Program) (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation M p version)) s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons pid jobs ih =>
    constructor
    · change TraceSafe M bounds (kernel version p) ((invocation M p version pid).initial s)
      apply AccessSafety.source_safe
      · intro i hi
        exact hb.2.1 ⟨0, by simp [sourceIO, inputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
      · intro i hi
        exact hb.2.1 ⟨1, by simp [sourceIO, inputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
      · intro i hi
        exact hb.2.2.1 ⟨0, by simp [sourceIO, outputs]⟩ (⟨i, hi⟩, PUnit.unit) trivial
    · intro next _
      exact ih next

theorem outside_extent (M : Algebra α) (p : Parameters) (version : Bool)
    (values : Inputs α p) (r : RegionName) (address : Nat)
    (outside : (sourceIO M p version).Outside values r address) :
    r ≠ p.output ∨ p.K*p.N ≤ address := by
  by_cases hr : r = p.output
  · right
    by_contra h
    have lt : address < p.K*p.N := by omega
    exact (outside.1 ⟨0, by simp [sourceIO, outputs]⟩ (⟨address, lt⟩, PUnit.unit)
      trivial).elim (fun h => h hr) (fun h => h rfl)
  · exact Or.inl hr

/-- Complete grid result and frame, assembled from the existing source run.
The independent scalar formula is related by a local scalar-model equality. -/
theorem implements (M : Algebra α) (p : Parameters) (hK : 0 < p.BK) (hN : 0 < p.BN)
    (sep : KernelGrid.Separate p) (f : Int → α → α)
    (scalar : ∀ q x, dequantized M q x = f q x) :
    (sourceIO M p Bool.false).Implements (result f) := by
  apply ProgramIO.Implements.intro
  · intro values bounds hb initial _
    exact schedule_safe M p Bool.false bounds values hb (KernelGrid.order p) initial
  · intro values initial hin
    obtain ⟨final, ran, output, frame⟩ := KernelGrid.correctness M p hK hN sep initial
    refine ⟨final, (schedule_eq M p Bool.false (KernelGrid.order p) initial).trans ran, ?_, ?_⟩
    · intro o j _
      fin_cases o
      change mapCell id (final.mem p.output j.1.val) =
        Cell.mk .real (f (quantized values j.1) (scale values (column p j.1)))
      rw [output j.1.val j.1.isLt]
      have qpin := hin ⟨0, by simp [sourceIO, inputs]⟩ j trivial
      have spin := hin ⟨1, by simp [sourceIO, inputs]⟩ (column p j.1, PUnit.unit) trivial
      change (initial.mem p.b j.1.val).read .int = quantized values j.1 at qpin
      change (initial.mem p.scale (j.1.val % p.N)).read .real = scale values (column p j.1) at spin
      simp only [KernelGrid.reference, qpin, spin, scalar, mapCell, mapValue, id_eq]
    · intro r address outside
      exact frame r address (outside_extent M p Bool.false values r address outside)

/-- Both actual full grids return the same raw state; the common layout gives
all active output observations and exact exterior frames. -/
theorem equivalent (M : Algebra α) (p : Parameters) (hK : 0 < p.BK) (hN : 0 < p.BN)
    (sep : KernelGrid.Separate p) :
    (sourceIO M p Bool.false).Equivalent (variant M p) := by
  apply ProgramIO.EquivalentOn.intro
  · intro values _ bounds hb initial _
    exact schedule_safe M p Bool.false bounds values hb (KernelGrid.order p) initial
  · intro values _ bounds hb initial _
    exact schedule_safe M p Bool.true bounds values hb (KernelGrid.order p) initial
  · intro values _ bounds hb initial hin
    obtain ⟨final, ran, _, frame⟩ := KernelGrid.correctness M p hK hN sep initial
    refine ⟨final, final, (schedule_eq M p Bool.false (KernelGrid.order p) initial).trans ran,
      ?_, ?_, ?_, ?_⟩
    · exact (schedule_eq M p Bool.true (KernelGrid.order p) initial).trans
        ((KernelWhole.same_run M p (KernelGrid.order p) initial).symm.trans ran)
    · intro o j _
      rfl
    all_goals
      intro r address outside
      exact frame r address (outside_extent M p Bool.false values r address outside)

#axiomsClean schedule_eq
#axiomsClean schedule_safe
#axiomsClean implements
#axiomsClean equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO
