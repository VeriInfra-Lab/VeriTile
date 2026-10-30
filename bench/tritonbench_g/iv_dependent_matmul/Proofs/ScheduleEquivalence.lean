import bench.tritonbench_g.iv_dependent_matmul.Programs

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.ScheduleEquivalence
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul FPEquiv
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Pointer staging changes no allocation or physical-address requirement. -/
def layout (g : Kernel.Geometry) (L : IO.Layout g) (schedule : Kernels.Schedule) :
    IO.Layout (IO.reschedule g schedule) :=
  ⟨L.freshA,L.freshB,L.baseA,L.baseB,L.baseC,L.physicalFreshA,L.physicalFreshB,
    L.alignA,L.alignB,L.alignC,L.inputAddressable,L.weightAddressable,L.outputAddressable⟩

def launchable (g : Kernel.Geometry) (l : Index64.Launchable g.request)
    (schedule : Kernels.Schedule) : Index64.Launchable (IO.reschedule g schedule).request :=
  ⟨l.rowsBlock,l.columnsBlock,l.innerBlock,l.grid⟩

theorem variant_io (M : Algebra Word) (g : Kernel.Geometry) (s : Kernels.Schedule) :
    (IO.scheduledVariant M .accumulator g s).toProgramIO =
      IO.matmulIO M .accumulator (IO.reschedule g s) := by
  simp [IO.scheduledVariant, ProgramIO.Variant.toProgramIO, IO.matmulIO,
    IO.reschedule, IO.inputs, IO.outputs]
  refine ⟨?_, rfl, ?_, rfl⟩
  · funext i vals j
    fin_cases i <;> rfl
  · funext o vals j
    fin_cases o
    rfl

/-- The independent full-K target contains no pointer-stage selector. -/
theorem formula_reschedule (profile : Target.NativeProfile) (g : Kernel.Geometry)
    (s : Kernels.Schedule) :
    FPEquiv.formula profile g = FPEquiv.formula profile (IO.reschedule g s) := by
  funext vals o j
  fin_cases o
  change Target.native profile.reference g.inner g.blockInner (IO.matrixA g vals) (IO.matrixB g vals) _ _ =
    Target.native profile.reference g.inner g.blockInner (IO.matrixA (IO.reschedule g s) vals)
      (IO.matrixB (IO.reschedule g s) vals) _ _
  have ha : IO.matrixA g vals = IO.matrixA (IO.reschedule g s) vals := by
    funext i k
    rfl
  have hb : IO.matrixB g vals = IO.matrixB (IO.reschedule g s) vals := by
    funext k c
    rfl
  rw [ha,hb]

/-- Each of the actual five source schedules implements the same independent
native recurrence. Reuse both complete native proofs with exact input cells,
successful grids, access safety and both exterior frames. -/
theorem equivalent (rest : Algebra Word) (profile : Target.NativeProfile)
    (g : Kernel.Geometry) (schedule : Kernels.Schedule) (L : IO.Layout g)
    (l : Index64.Launchable g.request) :
    original rest profile g ≡[[]] pointerScheduled rest profile g schedule := by
  apply ProgramIO.Implementation.of_equivalentCells
  apply ProgramIO.EquivalentOnCells.of_implementsOnCells
    (f := formula profile g)
  · exact IOResults.native rest profile g L l
  · have proof : (IO.matmulIO (backend rest profile) .accumulator (IO.reschedule g schedule)).ImplementsCells
        (FPEquiv.formula profile (IO.reschedule g schedule)) :=
      IOResults.native rest profile (IO.reschedule g schedule)
        (layout g L schedule) (launchable g l schedule)
    rw [formula_reschedule profile g schedule]
    intro vals _ bounds bounded initial inputs
    have bounded' : (IO.matmulIO (backend rest profile) .accumulator
        (IO.reschedule g schedule)).InBounds bounds vals := by
      refine ⟨bounded.1, ?_, ?_, ?_⟩
      · intro i j active
        fin_cases i
        · exact bounded.2.1 (0 : Fin 2) j active
        · exact bounded.2.1 (1 : Fin 2) j active
      · intro o j active
        fin_cases o
        exact bounded.2.2.1 (0 : Fin 1) j active
      · intro k
        exact k.elim0
    have inputs' : (IO.matmulIO (backend rest profile) .accumulator
        (IO.reschedule g schedule)).InputCells vals initial := by
      intro i j active
      fin_cases i
      · exact inputs (0 : Fin 2) j active
      · exact inputs (1 : Fin 2) j active
    obtain ⟨safe, final, ran, outputs, frame⟩ :=
      proof vals trivial bounds bounded' initial inputs'
    refine ⟨safe, final, ran, ?_, ?_⟩
    · intro o j active
      fin_cases o
      exact outputs (0 : Fin 1) j active
    · intro region off outside
      apply frame region off
      constructor
      · intro o j active
        fin_cases o
        exact outside.1 (0 : Fin 1) j active
      · intro k
        exact k.elim0

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.ScheduleEquivalence
