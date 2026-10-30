import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPCurrent
open VeriTile.Triton Host Contract Storage FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Source-order backward formulas use the actual saved X/W/r at entry. No
claim that stale r is the variance of newly mutated X is made. -/
def Result (M : Algebra α) (zero : Cell α) (ctx : Context (Cell α)) (g : Upstream (Cell α)) (actual : FPGridForward.Heap α) : Prop :=
  if ctx.call.rows=0 then ∀ j : Fin ctx.call.cols,actual ("dw",j.val)=zero
  else
    (∀ row : Fin ctx.call.rows,∀ j,
      actual (FPGridBackward.address ctx.call row j)=
        FPGridBackward.expected M ctx.call (seed ctx g) row j) ∧
    (∀ pid : Fin (Geometry.featureCount ctx.call),∀ j,
      actual (FPGridWeight.address ctx.call pid j)=
        FPGridWeight.expected M ctx.call (seed ctx g) pid j)

theorem backward_current {β : Type} (M : Algebra α) (zero : Cell α) (ctx : Context (Cell α))
    (g : Upstream (Cell α)) (metadata : TensorMetadata) (admitted : BackwardAdmitted ctx g metadata)
    (live : Caller) (encode : (Cell α) → Fin 4 → β) (entry : Heap β)
    (current : HostStorage.SavedAtEntry ctx encode entry) (p : BackwardPrepared ctx live) :
    ∃ actual,CheckedHost.backward (FPGridWeight.run M) (FPGridBackward.run M) zero
        ctx g metadata live encode entry (.ok p) (.ok ()) (.ok ()) (.ok ())=
      finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual)) ∧
      Result M zero ctx g actual := by
  rw [CheckedHost.backward_admitted _ _ _ ctx g metadata admitted]
  have valid:=backward_validated ctx g metadata admitted
  by_cases empty : ctx.call.rows=0
  · refine ⟨zeroWeight ctx.call zero (seed ctx g),Host.backward_empty _ _ _ _ _ _ _ _ _ valid empty (.ok ()) (.ok ()),?_⟩
    simp [Result,empty]
  · obtain ⟨actual,run,dx,dw,_⟩:=FPSequence.complete M ctx.call (seed ctx g)
    change (FPGridWeight.run M ctx.call (seed ctx g)).bind (FPGridBackward.run M ctx.call)=some actual at run
    cases weightRun : FPGridWeight.run M ctx.call (seed ctx g) with
    | none => simp [weightRun] at run
    | some weighted =>
      simp only [weightRun,Option.bind_some] at run
      refine ⟨actual,Host.backward_success _ _ _ _ _ _ _ _ _ valid empty weighted actual weightRun run (.ok ()),?_⟩
      simpa [Result,empty] using And.intro dx dw

#specNonCircular Result avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#axiomsClean backward_current
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPCurrent
