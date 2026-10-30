import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.Library
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.HostBackwardValues

namespace VeriTile.Bench.Repairs.FastCeLoss.Host
open Source API Grid Complete
noncomputable section
set_option maxHeartbeats 5000000

variable {α σ : Type} {runtime : Runtime σ} {live : σ → Nat → Prop}

 theorem Library.view_storage (laws : Library runtime live) {input shape before after output}
    (success : Host.view (α:=α) runtime input shape before=.ok output after) :
    output.tensor.storage=input.tensor.storage :=
  (laws.view _ _ _ _ _ (view_memory runtime _ _ _ _ _ success).2.2.2).2.1

 theorem Library.view_offset (laws : Library runtime live) {input shape before after output}
    (success : Host.view (α:=α) runtime input shape before=.ok output after) :
    output.tensor.offset=input.tensor.offset :=
  (laws.view _ _ _ _ _ (view_memory runtime _ _ _ _ _ success).2.2.2).2.2.1

 theorem FinishExecution.lse_live {a : Arithmetic α} {stream s logits labels losses chunks label before after lse}
    (e : FinishExecution a runtime stream s logits labels losses chunks label before lse after)
    (laws : Library runtime live) (chunksLive : live before.native chunks.tensor.storage) :
    live after.native lse.tensor.storage := by
  cases e with
  | single same => exact chunksLive
  | chunked different lse allocated combined added final allocation combine addition mask =>
    exact ((laws.perform_preserves combine).trans ((laws.perform_preserves addition).trans
      (laws.perform_preserves mask))) _ (laws.allocate_live allocation)

 theorem FinishExecution.single_alias {a : Arithmetic α} {stream s logits labels losses chunks label before after lse}
    (e : FinishExecution a runtime stream s logits labels losses chunks label before lse after)
    (single : s.chunks=1) : lse=chunks := by
  cases e with
  | single same => rfl
  | chunked different => exact False.elim (different single)

 theorem FinishExecution.lse_fresh {a : Arithmetic α} {stream s logits labels losses chunks label before after lse}
    (e : FinishExecution a runtime stream s logits labels losses chunks label before lse after)
    (laws : Library runtime live) (storage : Nat) (present : live before.native storage)
    (chunksFresh : chunks.tensor.storage≠storage) : lse.tensor.storage≠storage := by
  cases e with
  | single same => exact chunksFresh
  | chunked different lse allocated combined added final allocation combine addition mask =>
    exact laws.allocate_fresh allocation storage present

 theorem FinishExecution.chunk_fresh {a : Arithmetic α} {stream s logits labels losses chunks label before after lse}
    (e : FinishExecution a runtime stream s logits labels losses chunks label before lse after)
    (laws : Library runtime live) (chunked : s.chunks≠1) (storage : Nat)
    (present : live before.native storage) : lse.tensor.storage≠storage := by
  cases e with
  | single same => exact False.elim (chunked same)
  | chunked different lse allocated combined added final allocation combine addition mask =>
    exact laws.allocate_fresh allocation storage present

 theorem CountExecution.fresh {stream args logits label before after count}
    (e : CountExecution (α:=α) runtime stream args logits label before count after)
    (laws : Library runtime live) (storage : Nat) (present : live before.native storage) :
    ∀ tensor∈count,tensor.tensor.storage≠storage := by
  cases e with
  | explicit item items => simp
  | counted items tensor allocated final allocation operation =>
    intro other member
    have same : tensor=other := by simpa using member
    subst other
    exact laws.allocate_fresh allocation storage present

structure ForwardAllocationFacts {a : Arithmetic α} {scalars : ScalarInterpreter α} {stream args initial final result}
    (e : ForwardExecution a scalars runtime stream args initial final result) (live : σ → Nat → Prop) : Prop where
  logitsStorage : e.logits.tensor.storage=args.logits.tensor.storage
  logitsOffset : e.logits.tensor.offset=args.logits.tensor.offset
  labelsStorage : e.labels.tensor.storage=args.labels.tensor.storage
  labelsOffset : e.labels.tensor.offset=args.labels.tensor.offset
  layout : ∀ upstream,ForwardLayout args.shape (e.buffers upstream)
  sumLogits : e.logits.tensor.storage≠e.sum.tensor.storage
  scalarLogits : e.logits.tensor.storage≠e.scalar.tensor.storage
  sumLSE : e.lse.tensor.storage≠e.sum.tensor.storage
  scalarLSE : e.lse.tensor.storage≠e.scalar.tensor.storage
  countFresh : ∀ count∈e.count,count.tensor.storage≠e.labels.tensor.storage
  preserves : Preserves live initial.native final.native
  logitsLive : live final.native e.logits.tensor.storage
  lseLive : live final.native e.lse.tensor.storage
  scalarLive : live final.native e.scalar.tensor.storage
  fresh : ∀ storage,live initial.native storage →
    e.losses.tensor.storage≠storage ∧ e.chunks.tensor.storage≠storage ∧
    e.lse.tensor.storage≠storage ∧ e.sum.tensor.storage≠storage ∧ e.scalar.tensor.storage≠storage ∧
    ∀ count∈e.count,count.tensor.storage≠storage

 theorem ForwardExecution.allocation_facts {a : Arithmetic α} {scalars : ScalarInterpreter α} {stream args initial final result}
    (e : ForwardExecution a scalars runtime stream args initial final result)
    (laws : Library runtime live) (positive : 0<args.vocab)
    (logitsLive : live initial.native args.logits.tensor.storage)
    (labelsLive : live initial.native args.labels.tensor.storage) : ForwardAllocationFacts e live := by
  have px := laws.view_preserves e.logitsView
  have py := laws.view_preserves e.labelsView
  have pl := laws.allocate_preserves e.lossesAllocation
  have pc := laws.allocate_preserves e.chunksAllocation
  have pk := laws.perform_preserves e.kernel
  have pf := e.finish.preserves laws
  have ps := laws.perform_preserves e.save
  have pn := e.counting.preserves laws
  have pa := laws.allocate_preserves e.sumAllocation
  have pm := laws.perform_preserves e.summation
  have pb := laws.allocate_preserves e.scalarAllocation
  have pd := laws.perform_preserves e.division
  have pr := laws.perform_preserves e.returned
  have xeq := laws.view_storage e.logitsView
  have yeq := laws.view_storage e.labelsView
  have x0 : live initial.native e.logits.tensor.storage := by rw [xeq]; exact logitsLive
  have y0 : live initial.native e.labels.tensor.storage := by rw [yeq]; exact labelsLive
  have pre := px.trans py
  have toLaunch := pre.trans (pl.trans (pc.trans pk))
  have toSaved := toLaunch.trans (pf.trans ps)
  have toCounted := toSaved.trans pn
  have toSummed := toCounted.trans (pa.trans pm)
  have tail := pb.trans (pd.trans pr)
  have all := toSummed.trans tail
  have lossLive := laws.allocate_live e.lossesAllocation
  have chunksLive := laws.allocate_live e.chunksAllocation
  have chunksLosses := laws.allocate_fresh e.chunksAllocation _ lossLive
  have lossesLogits := laws.allocate_fresh e.lossesAllocation _ (pre _ x0)
  have chunksLogits := laws.allocate_fresh e.chunksAllocation _ ((pre.trans pl) _ x0)
  have sep : ∀ upstream,ForwardSeparated (e.buffers upstream) := fun _ => ⟨chunksLosses,chunksLogits,lossesLogits⟩
  have lseLive := e.finish.lse_live laws (pk _ chunksLive)
  have sumLogits := laws.allocate_fresh e.sumAllocation _ (toCounted _ x0)
  have scalarLogits := laws.allocate_fresh e.scalarAllocation _ (toSummed _ x0)
  have lseCounted := (ps.trans pn) _ lseLive
  have sumLSE := laws.allocate_fresh e.sumAllocation _ lseCounted
  have scalarLSE := laws.allocate_fresh e.scalarAllocation _ ((pa.trans pm) _ lseCounted)
  refine ⟨xeq,laws.view_offset e.logitsView,yeq,laws.view_offset e.labelsView,?_,sumLogits.symm,
    scalarLogits.symm,sumLSE.symm,scalarLSE.symm,?_,all,all _ x0,?_,?_,?_⟩
  · intro upstream
    apply setup_layout
    refine ⟨positive,rfl,sep upstream,?_,?_⟩
    · intro single
      have same := e.finish.single_alias single
      exact congrArg Descriptor.tensor same
    · intro chunked
      refine ⟨sep upstream,?_,?_,?_⟩
      · exact e.finish.chunk_fresh laws chunked _ (pk _ chunksLive)
      · exact e.finish.chunk_fresh laws chunked _ ((pc.trans pk) _ lossLive)
      · exact e.finish.chunk_fresh laws chunked _ (toLaunch _ x0)
  · exact e.counting.fresh laws _ (toSaved _ y0)
  · exact ((ps.trans pn).trans ((pa.trans pm).trans tail)) _ lseLive
  · exact (pd.trans pr) _ (laws.allocate_live e.scalarAllocation)
  · intro storage present
    have lf := laws.allocate_fresh e.lossesAllocation storage (pre _ present)
    have cf := laws.allocate_fresh e.chunksAllocation storage ((pre.trans pl) _ present)
    exact ⟨lf,cf,e.finish.lse_fresh laws storage (toLaunch _ present) cf,
      laws.allocate_fresh e.sumAllocation storage (toCounted _ present),
      laws.allocate_fresh e.scalarAllocation storage (toSummed _ present),
      e.counting.fresh laws storage (toSaved _ present)⟩

structure BackwardAllocationFacts {a : Arithmetic α} {scalars : ScalarInterpreter α} {stream saved upstream initial final result}
    (e : BackwardExecution a scalars runtime stream saved upstream initial final result) (live : σ → Nat → Prop) : Prop where
  dividedLogits : saved.context.logits.tensor.storage≠e.divided.tensor.storage
  dividedLSE : saved.context.lse.tensor.storage≠e.divided.tensor.storage
  expandedStorage : e.expanded.tensor.storage=e.divided.tensor.storage
  expansion : ∀ row,addr e.expanded.tensor (row*(e.expanded.strides.headD 0).toNat)=addr e.divided.tensor 0
  gradientStorage : e.gradient.tensor.storage=saved.context.logits.tensor.storage
  gradientOffset : e.gradient.tensor.offset=saved.context.logits.tensor.offset
  preserves : Preserves live initial.native final.native
  fresh : ∀ storage,live initial.native storage → e.divided.tensor.storage≠storage

 theorem BackwardExecution.allocation_facts {a : Arithmetic α} {scalars : ScalarInterpreter α} {stream saved upstream initial final result}
    (e : BackwardExecution a scalars runtime stream saved upstream initial final result)
    (laws : Library runtime live)
    (logitsLive : live initial.native saved.context.logits.tensor.storage)
    (lseLive : live initial.native saved.context.lse.tensor.storage) : BackwardAllocationFacts e live := by
  have expanded := laws.expand _ _ _ _ _ (expand_memory runtime _ _ _ _ _ e.expansion).2.2.2
  exact ⟨(laws.allocate_fresh e.allocation _ logitsLive).symm,
    (laws.allocate_fresh e.allocation _ lseLive).symm,expanded.2.1,
    laws.expand_coordinate e.expansion,laws.view_storage e.pullback,laws.view_offset e.pullback,
    (laws.allocate_preserves e.allocation).trans ((laws.perform_preserves e.division).trans
      ((laws.expand_preserves e.expansion).trans ((laws.perform_preserves e.expansionEvent).trans
        ((laws.perform_preserves e.kernel).trans ((laws.view_preserves e.pullback).trans
          (laws.perform_preserves e.returned)))))),laws.allocate_fresh e.allocation⟩

#axiomsClean ForwardExecution.allocation_facts
#axiomsClean BackwardExecution.allocation_facts
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastCeLoss.Host
