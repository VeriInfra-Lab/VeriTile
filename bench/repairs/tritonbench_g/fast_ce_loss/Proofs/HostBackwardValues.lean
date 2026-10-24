import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostValues

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API HostStages Complete Grid
noncomputable section
set_option maxHeartbeats 5000000

variable {α σ : Type} {a : Arithmetic α} {scalars : ScalarInterpreter α}
  {runtime : Runtime σ} {stream : Nat} {saved : ForwardReturn} {upstream : Descriptor}
  {initial final : World α σ} {result : WholeReturn}

def BackwardExecution.dividedMemory (e : BackwardExecution a scalars runtime stream saved upstream initial final result) : Memory α :=
  writeRows e.divided.tensor 1 (fun _ => a.upstreamDiv (a.load (initial.memory (addr upstream.tensor 0)))
    (denominatorValue a scalars saved.args saved.count initial)) initial.memory

theorem BackwardExecution.prepared_memory (e : BackwardExecution a scalars runtime stream saved upstream initial final result) :
    e.prepared.memory=e.dividedMemory ∧ e.prepared.integer=initial.integer := by
  have ha := allocate_memory runtime _ _ _ _ _ _ e.allocation
  have hd := perform_memory runtime _ _ _ _ _ _ e.division
  have he := expand_memory runtime _ _ _ _ _ e.expansion
  have hp := perform_memory runtime _ _ _ _ _ _ e.expansionEvent
  constructor
  · rw [hp.1,he.1,hd.1,ha.1]
    rfl
  · exact hp.2.1.trans (he.2.1.trans (hd.2.1.trans ha.2.1))

theorem BackwardExecution.final_memory (e : BackwardExecution a scalars runtime stream saved upstream initial final result) :
    final.memory=backwardGrid a saved.context.options saved.args.shape (returnedBuffers saved e.expanded)
      (labelValues saved.context.labels initial) e.dividedMemory ∧ final.integer=initial.integer := by
  have hk := perform_memory runtime _ _ _ _ _ _ e.kernel
  have hv := view_memory runtime _ _ _ _ _ e.pullback
  have hr := perform_memory runtime _ _ _ _ _ _ e.returned
  constructor
  · rw [hr.1,hv.1,hk.1,e.prepared_memory.1]
    have labels : labelValues saved.context.labels e.prepared=labelValues saved.context.labels initial := by
      unfold labelValues
      rw [e.prepared_memory.2]
    rw [labels]
    rfl
  · exact hr.2.1.trans (hv.2.1.trans (hk.2.1.trans e.prepared_memory.2))

theorem BackwardExecution.gradient_output (e : BackwardExecution a scalars runtime stream saved upstream initial final result)
    (sep : BackwardSeparated (returnedBuffers saved e.expanded)) (positive : 0<saved.args.shape.vocab)
    (dividedLogits : saved.context.logits.tensor.storage≠e.divided.tensor.storage)
    (dividedLSE : saved.context.lse.tensor.storage≠e.divided.tensor.storage)
    (expansion : ∀ row<saved.args.shape.rows,
      addr e.expanded.tensor (row*(e.expanded.strides.headD 0).toNat)=addr e.divided.tensor 0)
    (row col : Nat) (hr : row<saved.args.shape.rows) (hc : col<saved.args.shape.vocab) :
    final.memory (addr saved.context.logits.tensor (row*saved.args.shape.vocab+col))=
      backwardScalar a saved.context.options (labelValues saved.context.labels initial row) col
        (initial.memory (addr saved.context.logits.tensor (row*saved.args.shape.vocab+col)))
        (initial.memory (addr saved.context.lse.tensor row))
        (a.upstreamDiv (a.load (initial.memory (addr upstream.tensor 0)))
          (denominatorValue a scalars saved.args saved.count initial)) := by
  rw [e.final_memory.1]
  have hb := Grid.backward_output a saved.context.options saved.args.shape (returnedBuffers saved e.expanded)
    (labelValues saved.context.labels initial) e.dividedMemory sep positive row col hr hc
  apply hb.trans
  change backwardScalar a _ _ _ (e.dividedMemory (addr saved.context.logits.tensor _))
    (e.dividedMemory (addr saved.context.lse.tensor _))
    (e.dividedMemory (addr e.expanded.tensor (row*(e.expanded.strides.headD 0).toNat)))=_
  rw [expansion row hr]
  unfold BackwardExecution.dividedMemory
  rw [write_other _ _ _ _ _ _ dividedLogits,write_other _ _ _ _ _ _ dividedLSE,
    write_here _ _ _ _ _ (by decide)]

 theorem BackwardExecution.frame (e : BackwardExecution a scalars runtime stream saved upstream initial final result)
    (sep : BackwardSeparated (returnedBuffers saved e.expanded)) (positive : 0<saved.args.shape.vocab)
    (point : Address) (divisionOutside : point.1≠e.divided.tensor.storage)
    (outside : point.1≠saved.context.logits.tensor.storage ∨ point.2<saved.context.logits.tensor.offset ∨
      saved.context.logits.tensor.offset+saved.args.shape.rows*saved.args.shape.vocab≤point.2) :
    final.memory point=initial.memory point := by
  rw [e.final_memory.1,Grid.backward_frame a saved.context.options saved.args.shape
    (returnedBuffers saved e.expanded) _ _ sep positive point outside]
  have outsideDivision : ¬window e.divided.tensor 1 point := by intro h; exact divisionOutside h.1
  simp only [BackwardExecution.dividedMemory,writeRows,if_neg outsideDivision]

 theorem ForwardExecution.final_denominator {args : Arguments} {first middle : World α σ}
    (e : ForwardExecution a scalars runtime stream args first middle saved) :
    denominatorValue a scalars saved.args saved.count middle=
      Source.denominator a args.shape e.labelsAtEntry (args.items.map scalars.apply) := by
  have hr := perform_memory runtime _ _ _ _ _ _ e.returned
  have hd := perform_memory runtime _ _ _ _ _ _ e.division
  have same : middle.integer=e.scalarAllocated.integer := hr.2.1.trans hd.2.1
  have argsEq : saved.args=args := congrArg ForwardReturn.args e.resultExact
  have countEq : saved.count=e.count := congrArg ForwardReturn.count e.resultExact
  rw [argsEq,countEq]
  unfold denominatorValue
  rw [same]
  exact e.denominator

 theorem ForwardExecution.final_labels {args : Arguments} {first middle : World α σ}
    (e : ForwardExecution a scalars runtime stream args first middle saved)
    (countFresh : ∀ count∈e.count,count.tensor.storage≠e.labels.tensor.storage) :
    labelValues saved.context.labels middle=e.labelsAtEntry := by
  have labelsEq : saved.context.labels=e.labels := congrArg (fun r : ForwardReturn => r.context.labels) e.resultExact
  rw [labelsEq]
  unfold labelValues
  rw [e.integer_memory]
  funext row
  cases h : e.count with
  | none => rfl
  | some count =>
    have different := countFresh count (by simp [h])
    have addresses : addr e.labels.tensor row≠addr count.tensor 0 := by
      intro same
      exact different (congrArg Prod.fst same.symm)
    simp only [h,countMemory,writeInteger,if_neg addresses]
    rfl

/-- Entire actual forward and backward source runs share saved state. Every
saved LSE is derived from the same forward entry, never supplied as a premise. -/
theorem whole_gradient {args : Arguments} {first middle : World α σ}
    (forward : ForwardExecution a scalars runtime stream args first middle saved)
    (backward : BackwardExecution a scalars runtime stream saved upstream middle final result)
    (layout : ForwardLayout args.shape (forward.buffers backward.expanded))
    (sep : BackwardSeparated (returnedBuffers saved backward.expanded))
    (sumLogits : forward.logits.tensor.storage≠forward.sum.tensor.storage)
    (scalarLogits : forward.logits.tensor.storage≠forward.scalar.tensor.storage)
    (sumLSE : forward.lse.tensor.storage≠forward.sum.tensor.storage)
    (scalarLSE : forward.lse.tensor.storage≠forward.scalar.tensor.storage)
    (dividedLogits : saved.context.logits.tensor.storage≠backward.divided.tensor.storage)
    (dividedLSE : saved.context.lse.tensor.storage≠backward.divided.tensor.storage)
    (countFresh : ∀ count∈forward.count,count.tensor.storage≠forward.labels.tensor.storage)
    (expansion : ∀ row<saved.args.shape.rows,
      addr backward.expanded.tensor (row*(backward.expanded.strides.headD 0).toNat)=addr backward.divided.tensor 0)
    (row col : Nat) (hr : row<args.shape.rows) (hc : col<args.shape.vocab) :
    final.memory (addr forward.logits.tensor (row*args.shape.vocab+col))=
      gradientGraph a args.options args.shape (forward.buffers backward.expanded) forward.labelsAtEntry
        (args.items.map scalars.apply) (middle.memory (addr upstream.tensor 0)) first.memory row col := by
  have argsEq : saved.args=args := congrArg ForwardReturn.args forward.resultExact
  have logitsEq : saved.context.logits=forward.logits := congrArg (fun r : ForwardReturn => r.context.logits) forward.resultExact
  have lseEq : saved.context.lse=forward.lse := congrArg (fun r : ForwardReturn => r.context.lse) forward.resultExact
  have optionsEq : saved.context.options=args.options := congrArg (fun r : ForwardReturn => r.context.options) forward.resultExact
  have h := backward.gradient_output sep (by simpa only [argsEq] using layout.positive)
    dividedLogits dividedLSE expansion row col (by simpa only [argsEq] using hr) (by simpa only [argsEq] using hc)
  rw [forward.final_labels countFresh,forward.final_denominator,argsEq,logitsEq,lseEq,optionsEq] at h
  have hx := forward.saved_memory backward.expanded (addr forward.logits.tensor (row*args.shape.vocab+col)) sumLogits scalarLogits
  have hl := forward.saved_memory backward.expanded (addr forward.lse.tensor row) sumLSE scalarLSE
  rw [API.actual_forward_eq a args.options args.shape (forward.buffers backward.expanded)
    forward.labelsAtEntry first.memory layout] at hx hl
  have hx' := Complete.logits_saved a args.options args.shape (forward.buffers backward.expanded)
    forward.labelsAtEntry first.memory layout (row*args.shape.vocab+col)
  have hl' := Complete.forward_lse a args.options args.shape (forward.buffers backward.expanded)
    forward.labelsAtEntry first.memory layout row hr
  have hxFull := hx.trans hx'
  have hlFull := hl.trans hl'
  rw [hxFull,hlFull] at h
  exact h

#axiomsClean BackwardExecution.gradient_output
#axiomsClean BackwardExecution.frame
#axiomsClean ForwardExecution.final_labels
#axiomsClean whole_gradient
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
