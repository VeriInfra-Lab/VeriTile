import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Materialized
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.KernelRewrite

namespace VeriTile.Bench.Repairs.BmmChunkBwd.KernelInterface
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
open NativeGraph DirectFP
noncomputable section
set_option maxHeartbeats 6000000

/-- Execute the actual quoted source or its integer-index candidate on raw state. -/
def run {α : Type} [Inhabited α] (M : Algebra α) (h : HostCall) (choice : Fin 9)
    (candidate : Bool) : List HostFP.Pid → State α → Option (State α)
  | [], s => some s
  | q :: qs, s => do
      let next ← FP.Structural.exec (engine M)
        (if candidate then Kernels.indexCommuted h (tuning choice)
          else Kernels.fp32Source h (tuning choice))
        (HostFP.launch h (tuning choice) q s)
      run M h choice candidate qs next

theorem source_eq {α : Type} [Inhabited α] (M : Algebra α) (h : HostCall)
    (choice : Fin 9) (ha : h.aStorage = .fp32) (hd : h.doutStorage = .fp32)
    (qs : List HostFP.Pid) (s : State α) :
    run M h choice Bool.false qs s = HostFP.run M h (tuning choice) Bool.false qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [run, HostFP.run, Bool.false_eq_true, ↓reduceIte,
      Kernels.fp32Source_eq h (tuning choice) ha hd]
    congr 1
    funext next
    exact ih next

theorem versions_eq {α : Type} [Inhabited α] (M : Algebra α) (h : HostCall)
    (choice : Fin 9) (qs : List HostFP.Pid) (s : State α) :
    run M h choice Bool.false qs s = run M h choice Bool.true qs s := by
  induction qs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [run, Bool.false_eq_true, ↓reduceIte]
    rw [KernelRewrite.execution M]
    congr 1
    funext next
    exact ih next

/-- The complete quoted-source/candidate/native relation with every output,
exact source frame and physical exterior frame. Metadata here is only the
existing raw signed-INT64 geometry bound record, not Tensor metadata. -/
theorem native (rest : Algebra Expr) (h : HostCall) (choice : Fin 9)
    (precision : FPGrid.DefaultPrecision h)
    (controls : NativeControl.Metadata h (tuning choice))
    (a : FPFlat.Placement) (alloc : NativeAddress.Allocation a) (abstract : FlatAlloc)
    (layout : HostContract.Layout h (tuning choice) abstract) (extent : a.extent = abstract.extent)
    (tag : FPRoles.ResidualTag h) (fresh : FPPhysical.FreshStorage h a)
    (s : State Expr) (good : Good s) :
    ∃ final,
      run (FPDefault.backend (algebra rest) (dot choice)) h choice Bool.false
        (HostFP.grid h (tuning choice)) (FPFlat.lift h a s) = some final ∧
      run (FPDefault.backend (algebra rest) (dot choice)) h choice Bool.true
        (HostFP.grid h (tuning choice)) (FPFlat.lift h a s) = some final ∧
      (NativeExecution.grid (algebra rest) (dot choice) h (tuning choice) a alloc
        (HostFP.grid h (tuning choice)) s).mem = (FPFlat.publish a h.Out s final).mem ∧
      FPGridOutput.Result (algebra rest) (dot choice) h (tuning choice) (FPFlat.lift h a s) final ∧
      Good (NativeExecution.grid (algebra rest) (dot choice) h (tuning choice) a alloc
        (HostFP.grid h (tuning choice)) s) ∧
      (∀ r off, ¬ FPFlat.scratchCell a h.Out r off →
        (NativeExecution.grid (algebra rest) (dot choice) h (tuning choice) a alloc
          (HostFP.grid h (tuning choice)) s).mem r off = s.mem r off) := by
  obtain ⟨final, executed, stores⟩ := NativeExecution.source_grid (algebra rest) (dot choice)
    h choice precision controls a alloc abstract layout extent tag fresh s
    (fun off _ => (good _ _).1)
  obtain ⟨other, executed', result⟩ := FPGridOutput.complete (algebra rest) (dot choice)
    h (tuning choice) precision abstract layout (FPFlat.lift h a s)
  have same : other = final := Option.some.inj (executed'.symm.trans executed)
  subst other
  have source : run (FPDefault.backend (algebra rest) (dot choice)) h choice Bool.false
      (HostFP.grid h (tuning choice)) (FPFlat.lift h a s) = some final := by
    rw [source_eq _ _ _ precision.1 precision.2.1]
    exact executed
  refine ⟨final, source, ?_, stores, result,
    grid_good rest h choice a alloc _ s good, ?_⟩
  · rw [← versions_eq]; exact source
  · intro r off outside
    rw [stores]
    exact FPFlat.publish_frame a h.Out s final r off outside

#axiomsClean native
#specNonCircular FPGridOutput.Result avoiding [run, NativeExecution.grid, Kernels.fp32Source]
end
end VeriTile.Bench.Repairs.BmmChunkBwd.KernelInterface
