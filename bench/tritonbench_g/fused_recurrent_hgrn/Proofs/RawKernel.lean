import bench.tritonbench_g.fused_recurrent_hgrn.Proofs.Whole

namespace VeriTile.Bench.Repairs.FusedRecurrentHgrn.RawKernel
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- Both source grids, every active result and the separate phase frames.
The interface contains only raw state, input arrays, geometry and flags. -/
structure Result (p : Ops α) (input : Input α) (BH T D BDF BDB : Nat)
    (U S V : Bool) (initial f b : State α) : Prop where
  forwardExecuted : Execution.gridRun p (forward T D BDF U S).surfaceBody
    (Execution.programs BH D BDF) initial=some f
  backwardExecuted : Execution.gridRun p (backward T D BDB U V).surfaceBody
    (Execution.programs BH D BDB) f=some b
  output : Execution.Saved p b input U BH T D
  finalState : S=Bool.true → ∀ bh, bh<BH → ∀ d, d<D →
    b.mem "ht" (stateAddr bh D d)=.mk .real (state p U input bh d T)
  gradients : Execution.Gradients p b input U V BH T D
  forwardFrame : ∀ r a, ¬Execution.ForwardRegion BH T D S r a → f.mem r a=initial.mem r a
  backwardFrame : ∀ r a, ¬Execution.BackwardRegion BH T D r a → b.mem r a=f.mem r a

theorem complete (p : Ops α) (input : Input α) (BH T D BDF BDB : Nat)
    (hF : 0<BDF) (hB : 0<BDB) (U S V : Bool) (initial : State α)
    (reads : Execution.Inputs initial input T D) :
    ∃ f b, Result p input BH T D BDF BDB U S V initial f b := by
  obtain ⟨f,hf,hI,hO,hH,hff⟩ := Execution.forward_grid p input BH T D BDF hF U S initial reads
  obtain ⟨b,hb,_,hO,hG,hbf⟩ := Execution.backward_grid p input BH T D BDB hB U V f hI hO
  refine ⟨f,b,hf,hb,hO,?_,hG,hff,hbf⟩
  intro hs bh hh d hd
  rw [hbf "ht" _ (by simp [Execution.BackwardRegion])]
  exact hH hs bh hh d hd

#axiomsClean complete
end
end VeriTile.Bench.Repairs.FusedRecurrentHgrn.RawKernel
