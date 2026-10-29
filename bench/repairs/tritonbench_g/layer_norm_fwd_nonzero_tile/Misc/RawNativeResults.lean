import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Source
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.NativePrimitives
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.StorageExecutionReturn

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section

/-- Complete shared quoted-source execution, independent ordered source-FP
reference, and exact store complement, under actual active-output geometry. -/
theorem raw_quoted_fp {α : Type} [Inhabited α] (A : Algebra α) (cmp : TypedSource.Comparisons A)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (initial : State α) (width : widthCheck N=Bool.true)
    (sep : Domain.ReadWriteSeparated f r) (geometry : NativeGeometry.Geometry f r stride M N) :
    ∃ final, Nonempty (TypedSource.GridRun A f r stride M N eps p (tile N) (tile_arange N) initial final) ∧
      TypedSource.observe f r stride M N final=
        TypedSource.oracle A cmp f r stride M N eps p (tile N) ((widthCheck_iff N).mp width) initial ∧
      (∀ region offset, (∀ row : Fin M, ¬Contract.written f r stride M N row.val region offset) →
        final.mem region offset=initial.mem region offset) := by
  obtain ⟨final,run,post⟩ := TypedSource.source_grid_post A cmp f r stride M N eps p
    (tile N) (tile_arange N) initial sep geometry
  exact ⟨final,run,TypedSource.observe_eq_oracle A cmp f r stride M N eps p (tile N)
    ((widthCheck_iff N).mp width) initial final sep post,post.2⟩

/-- Native arithmetic is instantiated only at local opcodes, complete
owner-specific reduction trees and requested store conversions. The artifact
binds opcodes, ordered operands, source sites and masks recursively; it has no
whole-output or successful-execution premise. This is the declared compiler
boundary, not a proof of the installed compiler or equality with Real. -/
theorem fp_equiv {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (_selectedTile : block=tile c.N)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (initial : StorageMachine.Memory c) (template : State α) :
    TypedSource.observe c.flags KernelEntry.regions c.strides c.M c.N
      (TypedPhysical.pull (PhysicalDomain.layout c.allocations) (StorageMachine.execute c artifact initial) template)=
      StorageCompiler.reference c.profile c.ops c.casts artifact.graph.schedule c.flags
        KernelEntry.regions c.strides c.M c.N c.eps c.p
        (TypedPhysical.pull (PhysicalDomain.layout c.allocations) initial template) := by
  rw [StorageMachine.execute_observe,StorageCompiler.Artifact.observe_eq]

/-- Statistics are read back from actual native memory and equal the
owner-zero independent source expressions, with their FP32 store cast. -/
theorem fp_rstd {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (selectedTile : block=tile c.N)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (initial : StorageMachine.Memory c) (template : State α) (row : Fin c.M) :
    TypedPhysical.read (PhysicalDomain.layout c.allocations) (StorageMachine.execute c artifact initial)
      (KernelEntry.regions.rstd,row.val)=
      Cell.mk .real (c.ops.cast32 ((NativeParameters.rstd artifact.graph.schedule.reductions c.flags
        (StorageMachine.snapshot c initial) c.M c.N c.eps c.p row.val 0).eval c.ops)) := by
  exact congrArg (fun result => result.rstd row) (fp_equiv c selectedTile artifact initial template)

theorem fp_mean {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (selectedTile : block=tile c.N)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (initial : StorageMachine.Memory c) (template : State α) (row : Fin c.M)
    (layerNorm : c.flags.rms=Bool.false) :
    TypedPhysical.read (PhysicalDomain.layout c.allocations) (StorageMachine.execute c artifact initial)
      (KernelEntry.regions.mean,row.val)=
      Cell.mk .real (c.ops.cast32 ((NativeParameters.mean artifact.graph.schedule.reductions c.flags
        (StorageMachine.snapshot c initial) c.M c.N c.p row.val 0).eval c.ops)) := by
  have h := congrArg (fun result => result.mean.map (fun value => value row))
    (fp_equiv c selectedTile artifact initial template)
  simpa [TypedSource.observe,TypedPhysical.pull,StorageCompiler.reference,StorageCompiler.castResult,
    NativeSchedule.oracle,layerNorm] using h

/-- Zero columns still evaluate the scalar denominator zero. No NaN-bit or
exact-Real equation is postulated at this boundary. -/
theorem native_zero_mean (ops : NativeRow.Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (inputs : NativeRow.Inputs α) (M row owner : Nat) (p : α) :
    (NativeParameters.mean red f inputs M 0 p row owner).eval ops=
      ops.div ((NativeRow.sumExpr (red owner).mean
        (fun j => NativeParameters.combined f inputs M 0 p row j.val)).eval ops) (ops.literal 0) := by
  simp [NativeParameters.mean,NativeRow.Expr.eval]

theorem native_zero_variance (ops : NativeRow.Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (inputs : NativeRow.Inputs α) (M row owner : Nat) (p : α) :
    (NativeParameters.variance red f inputs M 0 p row owner).eval ops=
      ops.div ((NativeRow.squareExpr (red owner).variance
        (fun _ => NativeRow.Expr.literal 0)).eval ops) (ops.literal 0) := by
  simp [NativeParameters.variance,NativeRow.Expr.eval]

theorem native_zero_rstd (ops : NativeRow.Ops α) (red : Nat → Whole.Reductions block)
    (f : Flags) (inputs : NativeRow.Inputs α) (M row owner : Nat) (eps p : α) :
    (NativeParameters.rstd red f inputs M 0 eps p row owner).eval ops=
      ops.div (ops.literal 1) (ops.sqrt (ops.add eps
        (ops.div ((NativeRow.squareExpr (red owner).variance
          (fun _ => NativeRow.Expr.literal 0)).eval ops) (ops.literal 0)))) := by
  simp [NativeParameters.rstd,NativeParameters.variance,NativeRow.Expr.eval]

/-- Empty rows still store these exact native expressions. The masked
variance leaves are zero, and division by the scalar zero remains explicit;
the padded input stream may still include rowscale and dropout operations. -/
theorem fp_zero_statistics {α : Type} [Inhabited α] (c : StorageMachine.Config α block)
    (selectedTile : block=tile c.N) (zero : c.N=0)
    (artifact : StorageCompiler.Artifact c.profile c.flags c.N block)
    (initial : StorageMachine.Memory c) (template : State α) (row : Fin c.M) :
    TypedPhysical.read (PhysicalDomain.layout c.allocations) (StorageMachine.execute c artifact initial)
      (KernelEntry.regions.rstd,row.val)=
      Cell.mk .real (c.ops.cast32 (c.ops.div (c.ops.literal 1) (c.ops.sqrt (c.ops.add c.eps
        (c.ops.div ((NativeRow.squareExpr (artifact.graph.schedule.reductions 0).variance
          (fun _ => NativeRow.Expr.literal 0)).eval c.ops) (c.ops.literal 0)))))) ∧
    (c.flags.rms=Bool.false →
      TypedPhysical.read (PhysicalDomain.layout c.allocations) (StorageMachine.execute c artifact initial)
        (KernelEntry.regions.mean,row.val)=
        Cell.mk .real (c.ops.cast32 (c.ops.div
          ((NativeRow.sumExpr (artifact.graph.schedule.reductions 0).mean (fun j =>
            NativeParameters.combined c.flags (StorageMachine.snapshot c initial)
              c.M 0 c.p row.val j.val)).eval c.ops) (c.ops.literal 0)))) := by
  constructor
  · rw [fp_rstd c selectedTile artifact initial template row]
    generalize artifact.graph.schedule.reductions=reductions
    rw [zero,native_zero_rstd]
  · intro layerNorm
    rw [fp_mean c selectedTile artifact initial template row layerNorm]
    generalize artifact.graph.schedule.reductions=reductions
    rw [zero,native_zero_mean]

#axiomsClean raw_quoted_fp
#axiomsClean fp_equiv
#axiomsClean fp_rstd
#axiomsClean fp_mean
#axiomsClean native_zero_mean
#axiomsClean native_zero_variance
#axiomsClean native_zero_rstd
#axiomsClean fp_zero_statistics
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile
