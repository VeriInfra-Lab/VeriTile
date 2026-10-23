import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostContract

namespace VeriTile.Bench.Repairs.FastCeLoss.Checked
open Source Host
open CrossEntropy1.Host (Step)
noncomputable section

theorem input_contract {σ : Type} {live : σ → Nat → Prop} {fm bm args upstream native integers}
    (forward : ForwardContract fm args integers) (backward : BackwardContract bm args upstream)
    (bindings : Bindings live args upstream native) : InputContract live args upstream native integers := by
  refine ⟨forward.positive,forward.logitsShape,forward.labelsShape,backward.shape,
    forward.logitsDType,forward.labelsDType,backward.dtype,forward.device,backward.device,
    forward.logitsContiguous,forward.labelsContiguous,forward.logitsResolved,forward.labelsResolved,
    backward.resolved,bindings.logitsBounds,bindings.labelsBounds,bindings.upstreamBounds,
    bindings.logitsLive,bindings.labelsLive,bindings.upstreamLive,forward.distinct,backward.distinct,
    forward.labelsValid,bindings.capBinding,bindings.scaleBinding,?_⟩
  cases h : args.items with
  | none => simpa only [h] using forward.denominator
  | some value =>
    have denominator := forward.denominator
    rw [h] at denominator
    exact denominator.2.2

theorem whole_admitted {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (fm : ForwardMetadata) (bm : BackwardMetadata)
    (args : Arguments) (upstream : Descriptor) (initial : World α σ)
    (forward : ForwardContract fm args initial.integer) (backward : BackwardContract bm args upstream) :
    whole a scalars runtime stream fm bm args upstream initial=
      Host.whole a scalars runtime stream args upstream initial := by
  simp only [whole,if_pos forward,if_pos backward]
  rfl

theorem whole_success {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (stream : Nat) (fm : ForwardMetadata) (bm : BackwardMetadata)
    (args : Arguments) (upstream : Descriptor) (initial final : World α σ) (result : WholeReturn) :
    whole a scalars runtime stream fm bm args upstream initial=.ok result final ↔
      ForwardContract fm args initial.integer ∧ BackwardContract bm args upstream ∧
      Host.whole a scalars runtime stream args upstream initial=.ok result final := by
  classical
  by_cases hf : ForwardContract fm args initial.integer
  · by_cases hb : BackwardContract bm args upstream
    · rw [whole_admitted a scalars runtime stream fm bm args upstream initial hf hb]
      simp only [hf,hb,true_and]
    · simp only [whole,hf,hb,↓reduceIte,false_and,true_and,iff_false]
      cases h : Host.forward a scalars runtime stream args initial <;> simp [bindAction,h]
  · simp [whole,hf]

theorem native_grids {fm args integers} (forward : ForwardContract fm args integers) :
    args.shape.rows≤2^31-1 ∧ args.shape.blocks≤65535 ∧ args.shape.chunks≤65535 := by
  refine ⟨forward.rowGrid,forward.columnGrid,?_⟩
  change Source.ceilDiv args.vocab 65536≤65535
  have bound := forward.columnGrid
  unfold Source.ceilDiv at bound ⊢
  split_ifs at bound ⊢ <;> omega

#axiomsClean input_contract
#axiomsClean whole_success
#axiomsClean native_grids
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Checked
