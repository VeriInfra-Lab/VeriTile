import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Allocations

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API Grid Complete
noncomputable section
set_option maxHeartbeats 5000000

structure WholeExecution {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (live : σ → Nat → Prop) (stream : Nat) (args : Arguments)
    (upstream : Descriptor) (initial final : World α σ) (result : WholeReturn) where
  saved : ForwardReturn
  middle : World α σ
  forward : ForwardExecution a scalars runtime stream args initial middle saved
  backward : BackwardExecution a scalars runtime stream saved upstream middle final result
  forwardFacts : ForwardAllocationFacts forward live
  backwardFacts : BackwardAllocationFacts backward live

 theorem complete_execution {α σ : Type} (a : Arithmetic α) (scalars : ScalarInterpreter α)
    (runtime : Runtime σ) (live : σ → Nat → Prop) (stream : Nat) (args : Arguments)
    (upstream : Descriptor) (initial final : World α σ) (result : WholeReturn)
    (laws : Library runtime live) (positive : 0<args.vocab)
    (logitsLive : live initial.native args.logits.tensor.storage)
    (labelsLive : live initial.native args.labels.tensor.storage)
    (success : Host.whole a scalars runtime stream args upstream initial=.ok result final) :
    Nonempty (WholeExecution a scalars runtime live stream args upstream initial final result) := by
  obtain ⟨saved,middle,hf,hb⟩ := whole_success a scalars runtime stream args upstream initial final result success
  obtain ⟨f⟩ := forward_success a scalars runtime stream args initial middle saved hf
  obtain ⟨b⟩ := backward_success a scalars runtime stream saved upstream middle final result hb
  have ff := f.allocation_facts laws positive logitsLive labelsLive
  have xeq : saved.context.logits=f.logits := congrArg (fun r : ForwardReturn => r.context.logits) f.resultExact
  have leq : saved.context.lse=f.lse := congrArg (fun r : ForwardReturn => r.context.lse) f.resultExact
  have bf := b.allocation_facts laws (by rw [xeq]; exact ff.logitsLive) (by rw [leq]; exact ff.lseLive)
  exact ⟨⟨saved,middle,f,b,ff,bf⟩⟩

variable {α σ : Type} {a : Arithmetic α} {scalars : ScalarInterpreter α}
  {runtime : Runtime σ} {live : σ → Nat → Prop} {stream : Nat} {args : Arguments}
  {upstream : Descriptor} {initial final : World α σ} {result : WholeReturn}

 theorem WholeExecution.args_eq (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    e.saved.args=args := congrArg ForwardReturn.args e.forward.resultExact
 theorem WholeExecution.logits_eq (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    e.saved.context.logits=e.forward.logits := congrArg (fun r : ForwardReturn => r.context.logits) e.forward.resultExact
 theorem WholeExecution.lse_eq (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    e.saved.context.lse=e.forward.lse := congrArg (fun r : ForwardReturn => r.context.lse) e.forward.resultExact

 theorem WholeExecution.backward_separated (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    BackwardSeparated (returnedBuffers e.saved e.backward.expanded) := by
  constructor
  · change e.saved.context.logits.tensor.storage≠e.saved.context.lse.tensor.storage
    rw [e.logits_eq,e.lse_eq]
    have layout := e.forwardFacts.layout e.backward.expanded
    by_cases single : args.shape.chunks=1
    · have same := congrArg Tensor.storage (layout.singleAlias single)
      change e.forward.lse.tensor.storage=e.forward.chunks.tensor.storage at same
      rw [same]
      exact layout.separated.chunksLogits.symm
    · exact (layout.chunkFresh single).lseLogits.symm
  · change e.saved.context.logits.tensor.storage≠e.backward.expanded.tensor.storage
    rw [e.backwardFacts.expandedStorage]
    exact e.backwardFacts.dividedLogits

 theorem WholeExecution.forward_frame (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (point : Address) (present : live initial.native point.1) : e.middle.memory point=initial.memory point := by
  have fresh := e.forwardFacts.fresh point.1 present
  rw [e.forward.saved_memory e.backward.expanded point fresh.2.2.2.1.symm fresh.2.2.2.2.1.symm,
    API.actual_forward_eq a args.options args.shape (e.forward.buffers e.backward.expanded)
      e.forward.labelsAtEntry initial.memory (e.forwardFacts.layout _)]
  exact Complete.forward_frame a args.options args.shape (e.forward.buffers e.backward.expanded)
    e.forward.labelsAtEntry initial.memory (e.forwardFacts.layout _) point fresh.2.1.symm fresh.1.symm fresh.2.2.1.symm

 theorem WholeExecution.gradient_output (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (upstreamLive : live initial.native upstream.tensor.storage)
    (row col : Nat) (hr : row<args.shape.rows) (hc : col<args.shape.vocab) :
    final.memory (addr e.forward.logits.tensor (row*args.shape.vocab+col))=
      gradientGraph a args.options args.shape (e.forward.buffers e.backward.expanded) e.forward.labelsAtEntry
        (args.items.map scalars.apply) (initial.memory (addr upstream.tensor 0)) initial.memory row col := by
  have h := whole_gradient e.forward e.backward (e.forwardFacts.layout _) e.backward_separated
    e.forwardFacts.sumLogits e.forwardFacts.scalarLogits e.forwardFacts.sumLSE e.forwardFacts.scalarLSE
    e.backwardFacts.dividedLogits e.backwardFacts.dividedLSE e.forwardFacts.countFresh
    (fun row _ => e.backwardFacts.expansion row) row col hr hc
  rw [e.forward_frame (addr upstream.tensor 0) upstreamLive] at h
  exact h

 theorem WholeExecution.scalar_output (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    final.memory (addr e.forward.scalar.tensor 0)=
      scalarGraph a args.options args.shape (e.forward.buffers e.backward.expanded) e.forward.labelsAtEntry
        (args.items.map scalars.apply) initial.memory := by
  have outside : e.forward.scalar.tensor.storage≠e.saved.context.logits.tensor.storage := by
    rw [e.logits_eq]
    exact e.forwardFacts.scalarLogits.symm
  have fresh := e.backwardFacts.fresh _ e.forwardFacts.scalarLive
  rw [e.backward.frame e.backward_separated (by rw [e.args_eq]; exact (e.forwardFacts.layout e.backward.expanded).positive)
    (addr e.forward.scalar.tensor 0) fresh.symm (Or.inl outside)]
  exact e.forward.scalar_output e.backward.expanded (e.forwardFacts.layout _)

 theorem WholeExecution.caller_frame (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (point : Address) (present : live initial.native point.1)
    (outside : point.1≠args.logits.tensor.storage ∨ point.2<args.logits.tensor.offset ∨
      args.logits.tensor.offset+args.shape.rows*args.shape.vocab≤point.2) :
    final.memory point=initial.memory point := by
  have fresh := e.backwardFacts.fresh point.1 (e.forwardFacts.preserves _ present)
  have outsideSaved : point.1≠e.saved.context.logits.tensor.storage ∨ point.2<e.saved.context.logits.tensor.offset ∨
      e.saved.context.logits.tensor.offset+e.saved.args.shape.rows*e.saved.args.shape.vocab≤point.2 := by
    rw [e.logits_eq,e.args_eq,e.forwardFacts.logitsStorage,e.forwardFacts.logitsOffset]
    exact outside
  rw [e.backward.frame e.backward_separated (by rw [e.args_eq]; exact (e.forwardFacts.layout e.backward.expanded).positive)
    point fresh.symm outsideSaved,e.forward_frame point present]

 theorem WholeExecution.integer_frame (e : WholeExecution a scalars runtime live stream args upstream initial final result)
    (point : Address) (present : live initial.native point.1) : final.integer point=initial.integer point := by
  rw [e.backward.final_memory.2,e.forward.integer_memory]
  cases h : e.forward.count with
  | none => rfl
  | some count =>
    have fresh := (e.forwardFacts.fresh point.1 present).2.2.2.2.2 count (by simp [h])
    have addresses : point≠addr count.tensor 0 := by intro eq; exact fresh (congrArg Prod.fst eq).symm
    simp only [countMemory,writeInteger,if_neg addresses]

 theorem WholeExecution.returned_alias (e : WholeExecution a scalars runtime live stream args upstream initial final result) :
    result.customBackward=[some e.forward.logits.tensor,none,none,none] ∧
    result.publicGradient.tensor.storage=args.logits.tensor.storage ∧
    result.publicGradient.tensor.offset=args.logits.tensor.offset := by
  have tupleEq : result.customBackward=[some e.saved.context.logits.tensor,none,none,none] :=
    congrArg WholeReturn.customBackward e.backward.resultExact
  have gradientEq : result.publicGradient=e.backward.gradient := congrArg WholeReturn.publicGradient e.backward.resultExact
  rw [e.logits_eq] at tupleEq
  refine ⟨tupleEq,?_,?_⟩
  · rw [gradientEq,e.backwardFacts.gradientStorage,e.logits_eq,e.forwardFacts.logitsStorage]
  · rw [gradientEq,e.backwardFacts.gradientOffset,e.logits_eq,e.forwardFacts.logitsOffset]

#axiomsClean complete_execution
#axiomsClean WholeExecution.gradient_output
#axiomsClean WholeExecution.scalar_output
#axiomsClean WholeExecution.caller_frame
#axiomsClean WholeExecution.integer_frame
#axiomsClean WholeExecution.returned_alias
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
