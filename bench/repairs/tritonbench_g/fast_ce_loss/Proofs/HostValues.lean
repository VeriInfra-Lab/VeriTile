import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostForwardExecution

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API HostStages Complete
noncomputable section
set_option maxHeartbeats 5000000

variable {α σ : Type} {a : Arithmetic α} {scalars : ScalarInterpreter α}
  {runtime : Runtime σ} {stream : Nat} {args : Arguments} {initial final : World α σ} {result : ForwardReturn}

def ForwardExecution.buffers (e : ForwardExecution a scalars runtime stream args initial final result)
    (upstream : Descriptor) : Buffers :=
  ⟨e.logits.tensor,e.labels.tensor,e.losses.tensor,e.chunks.tensor,e.lse.tensor,e.scalar.tensor,
    upstream.tensor,(upstream.strides.headD 0).toNat⟩

def ForwardExecution.labelsAtEntry (e : ForwardExecution a scalars runtime stream args initial final result) : Nat → Int :=
  labelValues e.labels initial

theorem ForwardExecution.prefix (e : ForwardExecution a scalars runtime stream args initial final result) :
    e.chunksState.memory=initial.memory ∧ e.chunksState.integer=initial.integer := by
  have hx := view_memory runtime _ _ _ _ _ e.logitsView
  have hy := view_memory runtime _ _ _ _ _ e.labelsView
  have hl := allocate_memory runtime _ _ _ _ _ _ e.lossesAllocation
  have hc := allocate_memory runtime _ _ _ _ _ _ e.chunksAllocation
  exact ⟨hc.1.trans (hl.1.trans (hy.1.trans hx.1)),hc.2.1.trans (hl.2.1.trans (hy.2.1.trans hx.2.1))⟩

theorem ForwardExecution.labels_eq (e : ForwardExecution a scalars runtime stream args initial final result) :
    labelValues e.labels e.chunksState=e.labelsAtEntry := by
  unfold labelValues ForwardExecution.labelsAtEntry
  rw [e.prefix.2]
  rfl

theorem ForwardExecution.finished_memory (e : ForwardExecution a scalars runtime stream args initial final result)
    (upstream : Descriptor) :
    e.finished.memory=API.actualForward a args.options args.shape (e.buffers upstream) e.labelsAtEntry initial.memory ∧
      e.finished.integer=initial.integer := by
  have hk := perform_memory runtime _ _ _ _ _ _ e.kernel
  constructor
  · rw [e.finish.memory.1,hk.1,e.prefix.1,e.labels_eq]
    rfl
  · exact e.finish.memory.2.trans (hk.2.1.trans e.prefix.2)

theorem ForwardExecution.before_mean (e : ForwardExecution a scalars runtime stream args initial final result)
    (upstream : Descriptor) :
    e.sumAllocated.memory=API.actualForward a args.options args.shape (e.buffers upstream) e.labelsAtEntry initial.memory ∧
      e.sumAllocated.integer=countMemory args.shape e.labelsAtEntry e.count initial.integer := by
  have hs := perform_memory runtime _ _ _ _ _ _ e.save
  have ha := allocate_memory runtime _ _ _ _ _ _ e.sumAllocation
  constructor
  · exact ha.1.trans (e.counting.memory.1.trans (hs.1.trans (e.finished_memory upstream).1))
  · rw [ha.2.1,e.counting.memory.2,hs.2.1,(e.finished_memory upstream).2,e.labels_eq]
    rfl

theorem CountExecution.denominator {logits : Descriptor} {label : Nat → Int}
    {start counted later : World α σ} {count : Option Descriptor}
    (e : CountExecution runtime stream args logits label start count counted)
    (same : later.integer=counted.integer) :
    denominatorValue a scalars args count later=
      Source.denominator a args.shape label (args.items.map scalars.apply) := by
  cases e with
  | explicit item items => simp [denominatorValue,Source.denominator,items]
  | counted items tensor allocated counted allocation operation =>
    have ha := allocate_memory runtime _ _ _ _ _ _ allocation
    have ho := perform_memory runtime _ _ _ _ _ _ operation
    simp only [denominatorValue,items]
    rw [same,ho.2.1]
    simp [writeInteger,Source.denominator]

theorem ForwardExecution.denominator (e : ForwardExecution a scalars runtime stream args initial final result) :
    denominatorValue a scalars args e.count e.scalarAllocated=
      Source.denominator a args.shape e.labelsAtEntry (args.items.map scalars.apply) := by
  have hsumAlloc := allocate_memory runtime _ _ _ _ _ _ e.sumAllocation
  have hsum := perform_memory runtime _ _ _ _ _ _ e.summation
  have hscalar := allocate_memory runtime _ _ _ _ _ _ e.scalarAllocation
  have same : e.scalarAllocated.integer=e.counted.integer := hscalar.2.1.trans (hsum.2.1.trans hsumAlloc.2.1)
  rw [e.counting.denominator same,e.labels_eq]

theorem ForwardExecution.scalar_output (e : ForwardExecution a scalars runtime stream args initial final result)
    (upstream : Descriptor) (layout : ForwardLayout args.shape (e.buffers upstream)) :
    final.memory (addr e.scalar.tensor 0)=
      scalarGraph a args.options args.shape (e.buffers upstream) e.labelsAtEntry (args.items.map scalars.apply) initial.memory := by
  have hret := perform_memory runtime _ _ _ _ _ _ e.returned
  have hdiv := perform_memory runtime _ _ _ _ _ _ e.division
  have hscalar := allocate_memory runtime _ _ _ _ _ _ e.scalarAllocation
  have hsum := perform_memory runtime _ _ _ _ _ _ e.summation
  rw [hret.1]
  simp only [id_eq]
  rw [hdiv.1,write_here _ _ _ _ _ (by decide),hscalar.1,hsum.1,
    write_here _ _ _ _ _ (by decide),e.denominator,(e.before_mean upstream).1,
    API.actual_forward_eq a args.options args.shape (e.buffers upstream) e.labelsAtEntry initial.memory layout]
  exact Complete.scalar_loss a args.options args.shape (e.buffers upstream) e.labelsAtEntry _ initial.memory layout

/-- Later mean temporaries are ordinary fresh allocations, so the exact saved
LSE and logits from the custom forward remain available to the backward. -/
theorem ForwardExecution.saved_memory (e : ForwardExecution a scalars runtime stream args initial final result)
    (upstream : Descriptor) (point : Address)
    (sumOutside : point.1≠e.sum.tensor.storage) (scalarOutside : point.1≠e.scalar.tensor.storage) :
    final.memory point=API.actualForward a args.options args.shape (e.buffers upstream) e.labelsAtEntry initial.memory point := by
  have hret := perform_memory runtime _ _ _ _ _ _ e.returned
  have hdiv := perform_memory runtime _ _ _ _ _ _ e.division
  have hscalar := allocate_memory runtime _ _ _ _ _ _ e.scalarAllocation
  have hsum := perform_memory runtime _ _ _ _ _ _ e.summation
  have outsideSum : ¬window e.sum.tensor 1 point := by intro h; exact sumOutside h.1
  have outsideScalar : ¬window e.scalar.tensor 1 point := by intro h; exact scalarOutside h.1
  rw [hret.1]
  simp only [id_eq]
  rw [hdiv.1]
  simp only [writeRows,if_neg outsideScalar]
  rw [hscalar.1,hsum.1]
  simp only [writeRows,if_neg outsideSum]
  rw [(e.before_mean upstream).1]

 theorem ForwardExecution.integer_memory (e : ForwardExecution a scalars runtime stream args initial final result) :
    final.integer=countMemory args.shape e.labelsAtEntry e.count initial.integer := by
  have hret := perform_memory runtime _ _ _ _ _ _ e.returned
  have hdiv := perform_memory runtime _ _ _ _ _ _ e.division
  have hscalar := allocate_memory runtime _ _ _ _ _ _ e.scalarAllocation
  have hsum := perform_memory runtime _ _ _ _ _ _ e.summation
  exact hret.2.1.trans (hdiv.2.1.trans (hscalar.2.1.trans (hsum.2.1.trans (e.before_mean e.scalar).2)))

#axiomsClean ForwardExecution.finished_memory
#axiomsClean ForwardExecution.scalar_output
#axiomsClean ForwardExecution.saved_memory
#axiomsClean ForwardExecution.integer_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
