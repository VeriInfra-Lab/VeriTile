import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Input
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Contract

namespace VeriTile.Bench.Repairs.FastCeLoss.Correct
open Source Host
noncomputable section
set_option maxHeartbeats 5000000

/-!
Independent Real correctness of the repaired whole wrapper/custom-autograd call.
For contiguous raw/resolved FP32 logits, int64 labels, positive vocabulary and
valid/ignored targets, the loss is the logical-vocabulary cross entropy mean.
Negative nonzero scale/softcap values retain their source branches. The actual
count supplies the default positive denominator; explicit positive Python
scalars allow empty/all-ignored inputs. The external 0-D FP32 upstream is also
raw/resolved, with arbitrary stored sign. Tensor denominators are outside this
normal host-scalar API. Ordinary allocation/view/runtime success is explicit.
The public scalar cell, in-place full gradient, exact custom return, public
storage alias, untouched labels and caller source-memory frame are conclusions.
Real arithmetic is a separate interpretation; no native Real-accuracy claim.
-/

def items (args : Arguments) : Option ℝ := args.items.bind pythonValue

 theorem scalar_meaning {σ : Type} {live : σ → Nat → Prop} {args upstream native integers}
    (input : InputContract live args upstream native integers) :
    args.items.map realScalars.apply=(items args).map (fun x => (some x : WithBot ℝ)) := by
  have decode (value : PythonScalar) : realScalars.apply value=pythonValue value := by cases value <;> rfl
  cases h : args.items with
  | none => simp [items,h]; rfl
  | some value =>
    have present := input.denominator
    rw [h] at present
    obtain ⟨real,meaning,positive⟩ := present
    simp only [items,h,Option.map_some,Option.bind_some,decode,meaning]
    rfl

structure Specification {σ : Type} (live : σ → Nat → Prop) (args : Arguments)
    (initial final : World (WithBot ℝ) σ) (result : WholeReturn)
    (x : Nat → Nat → ℝ) (labels : Nat → Option (Fin args.shape.vocab)) (u : ℝ) : Prop where
  scalar : final.memory (addr result.forward.scalar.tensor 0)=
    some (Math.loss args.options.softcap args.options.scaling
      (Math.denominator (fun row : Fin args.shape.rows => labels row.val) (items args))
      (fun row : Fin args.shape.rows => labels row.val) (fun row col => x row.val col.val))
  gradient : ∀ row : Fin args.shape.rows,∀ col : Fin args.shape.vocab,
    final.memory (addr args.logits.tensor (row.val*args.shape.vocab+col.val))=
      some (Math.gradient args.options.softcap args.options.scaling
        (Math.denominator (fun row : Fin args.shape.rows => labels row.val) (items args)) u
        (fun row : Fin args.shape.rows => labels row.val) (fun row col => x row.val col.val) row col)
  callerFrame : ∀ point,live initial.native point.1 →
    (point.1≠args.logits.tensor.storage ∨ point.2<args.logits.tensor.offset ∨
      args.logits.tensor.offset+args.shape.rows*args.shape.vocab≤point.2) → final.memory point=initial.memory point
  integerFrame : ∀ point,live initial.native point.1 → final.integer point=initial.integer point
  customTuple : result.customBackward=[some result.forward.context.logits.tensor,none,none,none]
  gradientStorage : result.publicGradient.tensor.storage=args.logits.tensor.storage
  gradientOffset : result.publicGradient.tensor.offset=args.logits.tensor.offset
  scalarShape : result.forward.scalar.tensor.shape=[]
  gradientShape : result.publicGradient.tensor.shape=[args.batch,args.sequence,args.vocab]

 theorem correct {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop) (stream : Nat)
    (args : Arguments) (upstream : Descriptor) (initial final : World (WithBot ℝ) σ) (result : WholeReturn)
    (library : Library runtime live) (input : InputContract live args upstream initial.native initial.integer)
    (x : Nat → Nat → ℝ) (labels : Nat → Option (Fin args.shape.vocab)) (u : ℝ)
    (logits : ∀ row<args.shape.rows,∀ col<args.shape.vocab,
      initial.memory (addr args.logits.tensor (row*args.shape.vocab+col))=some (x row col))
    (targets : ∀ row<args.shape.rows,labelValues args.labels initial row=RealProof.labelCode (labels row))
    (upstreamValue : initial.memory (addr upstream.tensor 0)=some u)
    (execution : Host.whole RealProof.arithmetic realScalars runtime stream args upstream initial=.ok result final) :
    Specification live args initial final result x labels u := by
  obtain ⟨e⟩ := complete_execution RealProof.arithmetic realScalars runtime live stream args upstream initial final result
    library input.positive input.logitsLive input.labelsLive execution
  refine ⟨e.loss_real x labels (items args) (scalar_meaning input) targets logits,
    e.gradient_real input.upstreamLive x labels (items args) u (scalar_meaning input) upstreamValue targets logits,
    e.caller_frame,e.integer_frame,?_,e.returned_alias.2.1,e.returned_alias.2.2,
    (e.returned_shapes library).1,(e.returned_shapes library).2⟩
  rw [e.backward.resultExact]

/-- Successful checked forward and scalar-upstream hook establish the complete
mathematical input domain; ordinary descriptor/allocator binding remains explicit. -/
theorem checked_correct {σ : Type} (runtime : Runtime σ) (live : σ → Nat → Prop) (stream : Nat)
    (fm : Checked.ForwardMetadata) (bm : Checked.BackwardMetadata)
    (args : Arguments) (upstream : Descriptor) (initial final : World (WithBot ℝ) σ) (result : WholeReturn)
    (library : Library runtime live) (bindings : Checked.Bindings live args upstream initial.native)
    (x : Nat → Nat → ℝ) (labels : Nat → Option (Fin args.shape.vocab)) (u : ℝ)
    (logits : ∀ row<args.shape.rows,∀ col<args.shape.vocab,
      initial.memory (addr args.logits.tensor (row*args.shape.vocab+col))=some (x row col))
    (targets : ∀ row<args.shape.rows,labelValues args.labels initial row=RealProof.labelCode (labels row))
    (upstreamValue : initial.memory (addr upstream.tensor 0)=some u)
    (execution : Checked.whole RealProof.arithmetic realScalars runtime stream fm bm args upstream initial=.ok result final) :
    Specification live args initial final result x labels u := by
  obtain ⟨forward,backward,source⟩ := (Checked.whole_success RealProof.arithmetic realScalars runtime stream fm bm
    args upstream initial final result).mp execution
  exact correct runtime live stream args upstream initial final result library
    (Checked.input_contract forward backward bindings) x labels u logits targets upstreamValue source

#axiomsClean correct
#axiomsClean checked_correct
#specNonCircular Math.loss avoiding [Source.forwardGrid,Source.backwardGrid,Host.whole]
#specNonCircular Math.gradient avoiding [Source.forwardGrid,Source.backwardGrid,Host.whole]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Correct
