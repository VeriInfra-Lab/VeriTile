import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.CheckedHost

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RealCurrent
open VeriTile.Triton Host Contract Storage
noncomputable section

/-- Source-order backward formulas use the actual saved X/W/r at entry. No
claim that stale r is the variance of newly mutated X is made. -/
def Result (ctx : Context (MemCell)) (g : Upstream (MemCell)) (actual : RealGridForward.Heap) : Prop :=
  if ctx.call.rows=0 then ∀ j : Fin ctx.call.cols,actual ("dw",j.val)=(.real 0)
  else
    (∀ row : Fin ctx.call.rows,∀ j,
      RealGridForward.read (actual (RealGridBackward.address ctx.call row j))=
        RealGridBackward.expected ctx.call (seed ctx g) row j) ∧
    (∀ pid : Fin (Geometry.featureCount ctx.call),∀ j,
      RealGridForward.read (actual (RealGridWeight.address ctx.call pid j))=
        RealGridWeight.expected ctx.call (seed ctx g) pid j)

theorem backward_current {β : Type} (ctx : Context (MemCell))
    (g : Upstream (MemCell)) (metadata : TensorMetadata) (admitted : BackwardAdmitted ctx g metadata)
    (live : Caller) (encode : (MemCell) → Fin 4 → β) (entry : Heap β)
    (current : HostStorage.SavedAtEntry ctx encode entry) (p : BackwardPrepared ctx live) :
    ∃ actual,CheckedHost.backward (RealGridWeight.run ) (RealGridBackward.run ) (.real 0)
        ctx g metadata live encode entry (.ok p) (.ok ()) (.ok ()) (.ok ())=
      finish p.toPrepared encode entry actual (.ok (gradientTensors ctx p actual)) ∧
      Result ctx g actual := by
  rw [CheckedHost.backward_admitted _ _ _ ctx g metadata admitted]
  have valid:=backward_validated ctx g metadata admitted
  by_cases empty : ctx.call.rows=0
  · refine ⟨zeroWeight ctx.call (.real 0) (seed ctx g),Host.backward_empty _ _ _ _ _ _ _ _ _ valid empty (.ok ()) (.ok ()),?_⟩
    simp [Result,empty]
  · obtain ⟨actual,run,dx,dw,_⟩:=RealSequence.complete ctx.call (seed ctx g)
    change (RealGridWeight.run ctx.call (seed ctx g)).bind (RealGridBackward.run ctx.call)=some actual at run
    cases weightRun : RealGridWeight.run ctx.call (seed ctx g) with
    | none => simp [weightRun] at run
    | some weighted =>
      simp only [weightRun,Option.bind_some] at run
      refine ⟨actual,Host.backward_success _ _ _ _ _ _ _ _ _ valid empty weighted actual weightRun run (.ok ()),?_⟩
      simpa [Result,empty] using And.intro dx dw

#specNonCircular Result avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#axiomsClean backward_current
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RealCurrent
