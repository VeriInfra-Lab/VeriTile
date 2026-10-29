import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostWhole
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.RealAPI

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API Complete
noncomputable section
set_option maxHeartbeats 5000000

variable {α σ : Type} {a : Arithmetic α} {scalars : ScalarInterpreter α}
  {runtime : Runtime σ} {live : σ → Nat → Prop} {stream : Nat} {args : Arguments}
  {upstream : Descriptor} {initial final : World α σ} {result : WholeReturn}

 theorem WholeExecution.logits_address (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (index : Nat) : addr e.forward.logits.tensor index=addr args.logits.tensor index :=
  Prod.ext e.forwardFacts.logitsStorage (by simp only [addr,e.forwardFacts.logitsOffset])

 theorem WholeExecution.labels_address (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (index : Nat) : addr e.forward.labels.tensor index=addr args.labels.tensor index :=
  Prod.ext e.forwardFacts.labelsStorage (by simp only [addr,e.forwardFacts.labelsOffset])

 theorem WholeExecution.label_values (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    e.forward.labelsAtEntry=labelValues args.labels initial := by
  funext row
  change initial.integer (addr e.forward.labels.tensor row)=initial.integer (addr args.labels.tensor row)
  rw [e.labels_address]

 theorem WholeExecution.scalar_descriptor (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    result.forward.scalar=e.forward.scalar := by
  have one : result.forward=e.saved := congrArg WholeReturn.forward e.backward.resultExact
  have two : e.saved.scalar=e.forward.scalar := congrArg ForwardReturn.scalar e.forward.resultExact
  exact (congrArg ForwardReturn.scalar one).trans two

 theorem WholeExecution.returned_shapes (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (laws : Library runtime live) : result.forward.scalar.tensor.shape=[] ∧
      result.publicGradient.tensor.shape=[args.batch,args.sequence,args.vocab] := by
  have scalar := laws.allocate _ _ _ _ _ _ (allocate_memory runtime _ _ _ _ _ _ e.forward.scalarAllocation).2.2.2
  have gradient := laws.view _ _ _ _ _ (view_memory runtime _ _ _ _ _ e.backward.pullback).2.2.2
  have gradientEq : result.publicGradient=e.backward.gradient := congrArg WholeReturn.publicGradient e.backward.resultExact
  constructor
  · rw [e.scalar_descriptor]
    exact scalar.1
  · rw [gradientEq,gradient.1,e.args_eq]
    rfl

 theorem WholeExecution.loss_real {scalars : ScalarInterpreter (WithBot ℝ)}
    {initial final : World (WithBot ℝ) σ}
    (e : WholeExecution RealProof.arithmetic scalars runtime live stream args upstream initial final result)
    (x : Nat → Nat → ℝ) (logical : Nat → Option (Fin args.shape.vocab)) (items : Option ℝ)
    (scalarMeaning : args.items.map scalars.apply=items.map (fun x => (some x : WithBot ℝ)))
    (codes : ∀ row<args.shape.rows,labelValues args.labels initial row=RealProof.labelCode (logical row))
    (inputs : ∀ row<args.shape.rows,∀ col<args.shape.vocab,
      initial.memory (addr args.logits.tensor (row*args.shape.vocab+col))=some (x row col)) :
    final.memory (addr result.forward.scalar.tensor 0)=
      some (Math.loss args.options.softcap args.options.scaling
        (Math.denominator (fun row : Fin args.shape.rows => logical row.val) items)
        (fun row : Fin args.shape.rows => logical row.val) (fun row col => x row.val col.val)) := by
  rw [e.scalar_descriptor,e.scalar_output,scalarMeaning]
  have inputViews : ∀ row<args.shape.rows,∀ col<args.shape.vocab,
      initial.memory (addr (e.forward.buffers e.backward.expanded).logits (row*args.shape.vocab+col))=some (x row col) := by
    intro row hr col hc
    change initial.memory (addr e.forward.logits.tensor _)=_
    rw [e.logits_address]
    exact inputs row hr col hc
  have codeViews : ∀ row<args.shape.rows,e.forward.labelsAtEntry row=RealProof.labelCode (logical row) := by
    rw [e.label_values]
    exact codes
  have h := RealProof.scalar_graph_real args.options args.shape (e.forward.buffers e.backward.expanded)
      e.forward.labelsAtEntry initial.memory (e.forwardFacts.layout _) x logical items codeViews inputViews
  rw [RealProof.logical_count args.shape e.forward.labelsAtEntry logical codeViews] at h
  exact h

 theorem WholeExecution.gradient_real {scalars : ScalarInterpreter (WithBot ℝ)}
    {initial final : World (WithBot ℝ) σ}
    (e : WholeExecution RealProof.arithmetic scalars runtime live stream args upstream initial final result)
    (upstreamLive : live initial.native upstream.tensor.storage)
    (x : Nat → Nat → ℝ) (logical : Nat → Option (Fin args.shape.vocab)) (items : Option ℝ) (u : ℝ)
    (scalarMeaning : args.items.map scalars.apply=items.map (fun x => (some x : WithBot ℝ)))
    (upstreamMeaning : initial.memory (addr upstream.tensor 0)=some u)
    (codes : ∀ row<args.shape.rows,labelValues args.labels initial row=RealProof.labelCode (logical row))
    (inputs : ∀ row<args.shape.rows,∀ col<args.shape.vocab,
      initial.memory (addr args.logits.tensor (row*args.shape.vocab+col))=some (x row col))
    (row : Fin args.shape.rows) (col : Fin args.shape.vocab) :
    final.memory (addr args.logits.tensor (row.val*args.shape.vocab+col.val))=
      some (Math.gradient args.options.softcap args.options.scaling
        (Math.denominator (fun row : Fin args.shape.rows => logical row.val) items) u
        (fun row : Fin args.shape.rows => logical row.val) (fun row col => x row.val col.val) row col) := by
  have h := e.gradient_output upstreamLive row.val col.val row.isLt col.isLt
  rw [e.logits_address,scalarMeaning,upstreamMeaning] at h
  rw [h]
  apply RealProof.gradient_graph_real args.options args.shape _ _ _ (e.forwardFacts.layout _) x logical items u
  · rw [e.label_values]
    exact codes
  · intro r hr c hc
    change initial.memory (addr e.forward.logits.tensor _)=_
    rw [e.logits_address]
    exact inputs r hr c hc

#axiomsClean WholeExecution.loss_real
#axiomsClean WholeExecution.gradient_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
