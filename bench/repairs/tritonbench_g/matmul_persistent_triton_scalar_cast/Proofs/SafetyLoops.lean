import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetySteps
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetyAccess

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SourceSafety
open VeriTile.Triton FP.Structural Core
open VeriTile.Bench.Repairs.MatmulPersistentTriton
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GridExecution (atProgram)
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem signed_safe_of_nat_invariant (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (stop : Nat) (stride : Nat)
    (body : List ComputeStmt) (P : Nat → State α → Prop)
    (bodySafe : ∀ q, q < stop → ∀ t, P q t →
      Safety.traceSafeList M bounds body (t.setReg idx .int [] (fun _ => (q : Int))))
    (preserved : ∀ q, q < stop → ∀ t, P q t → ∀ u,
      run M body (t.setReg idx .int [] (fun _ => (q : Int))) = some u → P (q + stride) u)
    (start : Nat) (s : State α) (initial : P start s) :
    Safety.rangeIntSafe M bounds idx start stop stride body s := by
  apply Safety.rangeIntSafe_of_invariant M bounds idx stop stride body
    (fun q t => 0 ≤ q ∧ P q.toNat t)
  · intro q hq t inv
    have eq : (q.toNat : Int) = q := Int.toNat_of_nonneg inv.1
    simpa only [eq] using bodySafe q.toNat (by omega) t inv.2
  · intro q hq t inv u ran
    have eq : (q.toNat : Int) = q := Int.toNat_of_nonneg inv.1
    refine ⟨by omega, ?_⟩
    have result := preserved q.toNat (by omega) t inv.2 u (by simpa only [eq] using ran)
    simpa only [Int.toNat_add inv.1 (Int.natCast_nonneg _),Int.toNat_natCast] using result
  · simpa using And.intro (Int.natCast_nonneg start) initial

/-- Bounds for exactly the source's clamped loads and masked output stores. -/
structure TileBounds (p : Core.Parameters) (bounds : RegionName → Nat) (tile : Nat) : Prop where
  a : ∀ block (i : AIndex p),p.problem.k block i.2.1.val<p.problem.K → aOffset p tile block i<bounds "A"
  b : ∀ block (i : BIndex p),p.problem.k block i.1.val<p.problem.K → bOffset p tile block i<bounds "B"
  c : ∀ i : CIndex,active p tile i → cOffset p tile i<bounds "C"

theorem inner_loop_safe (P : Source.Arithmetic α α) (p : Core.Parameters)
    (initial current : State α) (tile : Nat) (h : InnerInvariant P p initial tile 0 current)
    (bounds : RegionName → Nat) (hb : TileBounds p bounds tile) :
    Safety.rangeIntSafe (backend P) bounds "ki" 0 p.problem.kTiles 1 (Core.inner p) current := by
  apply signed_safe_of_nat_invariant (backend P) bounds "ki" p.problem.kTiles 1 (Core.inner p)
    (fun q t => InnerInvariant P p initial tile q t)
  · intro q _ t inv
    exact inner_safe P p initial t tile q inv bounds (hb.a q) (hb.b q)
  · intro q _ t inv u ran
    rw [bodyExecution P p initial t tile q inv] at ran
    cases Option.some.inj ran
    exact bodyInvariant P p initial t tile q inv
  · exact h

theorem inner_statement_safe (P : Source.Arithmetic α α) (p : Core.Parameters)
    (current : State α) (pid : Nat) (h : Registers p pid current) (bounds : RegionName → Nat)
    (safe : Safety.rangeIntSafe (backend P) bounds "ki" 0 p.problem.kTiles 1 (Core.inner p) current) :
    Safety.stepSafe (backend P) bounds (innerStatement p) current := by
  cases hp : p.problem.profile <;>
    simpa [innerStatement,Core.inner,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,hp,ComputeKernel.surfaceBody,
      Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,h.count] using safe

theorem outer_safe (P : Source.Arithmetic α α) (p : Core.Parameters)
    (initial : State α) (pid ordinal : Nat) (regs : Registers p pid initial)
    (tileBound : p.problem.tile pid ordinal<p.problem.tiles) (bounds : RegionName → Nat)
    (hb : TileBounds p bounds (p.problem.tile pid ordinal)) :
    Safety.traceSafeList (backend P) bounds (outer p)
      (initial.setReg "tile_ordinal" .int [] (fun _ => (ordinal : Int))) ∧
    ∃ final,run (backend P) (outer p)
      (initial.setReg "tile_ordinal" .int [] (fun _ => (ordinal : Int)))=some final ∧ Registers p pid final := by
  let setup := tilePreludeExecution P p pid ordinal initial regs
  have invariant := tile_initial P p pid ordinal initial regs tileBound
  have registers := tile_registers P p pid ordinal initial regs
  obtain ⟨last,ran,inv,reg⟩ := innerLoop P p initial setup.val pid (p.problem.tile pid ordinal) 0
    (Nat.zero_le _) invariant registers
  have stepRan : step (backend P) (innerStatement p) setup.val=some last := by
    simp only [Int.natCast_zero] at ran
    rw [dynamicInner P p setup.val pid registers,ran]
  constructor
  · rw [outer_decomposition,Safety.traceSafeList_append]
    refine ⟨tile_prelude_safe P p bounds _,?_⟩
    intro t ht
    rw [setup.property] at ht
    cases Option.some.inj ht
    exact Safety.traceSafeList_cons_of_step
      (inner_statement_safe P p setup.val pid registers bounds
        (inner_loop_safe P p initial setup.val _ invariant bounds hb)) stepRan
      (suffix_safe P p initial last _ inv bounds hb.c)
  · refine ⟨suffixResult P p initial last (p.problem.tile pid ordinal),?_,
      suffix_registers P p initial last pid (p.problem.tile pid ordinal) reg⟩
    rw [outer_decomposition,run_append,setup.property,Option.bind_some,run,stepRan]
    exact suffixExecution P p initial last _ inv

theorem program_safe (P : Source.Arithmetic α α) (p : Core.Parameters) (initial : State α)
    (pid : Nat) (sms : 0<p.problem.SMS) (pidBound : pid<p.problem.grid)
    (bounds : RegionName → Nat) (hb : ∀ tile,tile<p.problem.tiles → TileBounds p bounds tile) :
    TraceSafe (backend P) bounds (source p) (atProgram initial pid) := by
  let prepared := preludeExecution P p (atProgram initial pid)
  have regs : Registers p pid prepared.val := by
    simpa [atProgram] using (prelude_invariant P p (atProgram initial pid)).1
  have bodySafe (ordinal : Nat) (ho : ordinal<p.problem.work pid) (current : State α)
      (ready : Registers p pid current) :=
    let tileBound : p.problem.tile pid ordinal<p.problem.tiles :=
      (Source.tiles_exact p.problem sms _).mp
        ((Source.tile_mem p.problem _).mpr ⟨pid,pidBound,ordinal,ho,rfl⟩)
    outer_safe P p current pid ordinal ready tileBound bounds (hb _ tileBound)
  have outerSafe : Safety.rangeIntSafe (backend P) bounds "tile_ordinal" 0 (p.problem.work pid) 1
      (outer p) prepared.val := by
    apply signed_safe_of_nat_invariant (backend P) bounds "tile_ordinal" (p.problem.work pid) 1
      (outer p) (fun _ t => Registers p pid t)
    · intro q hq t ready
      exact (bodySafe q hq t ready).1
    · intro q hq t ready u ran
      obtain ⟨v,hv,rv⟩ := (bodySafe q hq t ready).2
      rw [hv] at ran
      cases Option.some.inj ran
      exact rv
    · exact regs
  change Safety.traceSafeList (backend P) bounds (source p).surfaceBody (atProgram initial pid)
  rw [source_decomposition,Safety.traceSafeList_append]
  refine ⟨prelude_safe P p bounds _,?_⟩
  intro t ht
  rw [prepared.property] at ht
  cases Option.some.inj ht
  apply Safety.traceSafeList_cons_intro
  · cases hp : p.problem.profile <;>
      simpa [outerStatement,outer,source,TypedSource.kernel,TypedSource.fp16,TypedSource.f8e4,hp,ComputeKernel.surfaceBody,
        Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,regs.work] using outerSafe
  · intro t _
    exact Safety.traceSafeList_nil _ _ _

#axiomsClean program_safe
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.SourceSafety
