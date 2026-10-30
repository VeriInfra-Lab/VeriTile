import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageView

namespace VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
variable {α Host : Type} {caller : Address → Prop}

/-- All eight prepared tensor observations agree with native addressed memory,
including original-handle fallback, optional scalars, both seed halves and
combined-mask halves. No output-value equation is a premise. -/
theorem prepared_observe_native [Inhabited α]
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hl : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller))
    (q : Request) (initial current : HostState α Host) (p : Prepared) (args : LaunchArguments)
    (hp : RequestPresent store q initial)
    (h : LaunchStateBound store tensor q initial p args current) (hv : ValidatedFacts q p.validated)
    (hi : InputsDense p.inputs) (ho : OutputsDense p.outputs)
    (c : StorageMachine.Config α block) (hc : Placement store q p args current c)
    (benchmarks : List (StorageMachine.DispatchEvent c.profile c.flags c.N block))
    (selected : StorageCompiler.Artifact c.profile c.flags c.N block) (template : State α) :
    let native := {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}
    preparedObserve store native p (HostReturn.seedCount (p.flags q) p.validated.M)=
      TypedSource.observe (p.flags q) preparedRegions (argumentStrides args) p.validated.M p.validated.N
        (TypedPhysical.pull (PhysicalDomain.layout c.allocations) native.memory template) := by
  let native := {current with memory:=(StorageMachine.dispatch c benchmarks selected current.memory).memory}
  have hread := prepared_value_readback store hm hh tensor q initial current native p args h hv hi ho
    c hc.allocations rfl
  have hy1 := optionAllocated_presence h.outputs.2.1
  have hmean := optionAllocated_presence h.outputs.2.2.2.1
  have hseeds := optionAllocated_presence h.outputs.2.2.2.2.2.1
  have hmask := optionAllocated_presence h.outputs.2.2.2.2.2.2
  have hdrop : p.outputs.mask.isSome=Bool.true → q.dropout=Bool.true := by
    intro he
    rw [he] at hmask
    cases hd : q.dropout
    · have hf : Bool.true=Bool.false := by simpa only [hd,Bool.and_false] using hmask
      cases hf
    · rfl
  apply TypedSource.Result.ext
  · funext i j
    exact hread i.val j.val i.isLt .y trivial j.isLt p.outputs.y rfl
  · cases he : p.outputs.y1 with
    | none =>
      have hf : (p.flags q).w1=Bool.false := by simpa [he,Prepared.flags] using hy1.symm
      simp [preparedObserve,TypedSource.observe,he,hf]
    | some value =>
      have hf : (p.flags q).w1=Bool.true := by simpa [he,Prepared.flags] using hy1.symm
      simp only [preparedObserve,TypedSource.observe,he,hf,ite_true,Option.map_some]
      apply congrArg some
      funext i j
      exact hread i.val j.val i.isLt .y1 hf j.isLt value he
  · cases he : p.outputs.mean with
    | none =>
      have hf : (p.flags q).rms=Bool.true := by
        simpa [he,Prepared.flags] using congrArg Bool.not hmean.symm
      simp [preparedObserve,TypedSource.observe,he,hf]
    | some value =>
      have hf : (p.flags q).rms=Bool.false := by
        simpa [he,Prepared.flags] using congrArg Bool.not hmean.symm
      simp only [preparedObserve,TypedSource.observe,he,hf,Bool.false_eq_true,ite_false,Option.map_some]
      apply congrArg some
      funext i
      exact hread i.val 0 i.isLt .mean hf trivial value he
  · funext i
    exact hread i.val 0 i.isLt .rstd trivial trivial p.outputs.rstd rfl
  · cases he : p.outputs.residualOut with
    | none =>
      have hf : (p.flags q).residualOut=Bool.false := by simp [Prepared.flags,he]
      simp only [preparedObserve,TypedSource.observe,he,hf,Bool.false_eq_true,ite_false,Option.getD_none]
      funext i j
      exact dispatch_fallback_readback store hm hh hl tensor q initial current p args hp h hv hi ho c hc
        benchmarks selected i.val j.val i.isLt j.isLt
    | some value =>
      have hf : (p.flags q).residualOut=Bool.true := by simp [Prepared.flags,he]
      simp only [preparedObserve,TypedSource.observe,he,hf,ite_true,Option.getD_some]
      funext i j
      exact hread i.val j.val i.isLt .residualOut hf j.isLt value he
  · cases he : p.outputs.seeds with
    | none =>
      have hf : (p.flags q).dropout=Bool.false := by simpa [he,Prepared.flags] using hseeds.symm
      simp [preparedObserve,TypedSource.observe,he,hf]
    | some value =>
      have hf : (p.flags q).dropout=Bool.true := by simpa [he,Prepared.flags] using hseeds.symm
      simp only [preparedObserve,TypedSource.observe,he,hf,ite_true,Option.map_some]
      apply congrArg some
      funext i
      by_cases hir : i.val<p.validated.M
      · exact hread i.val 0 hir .seeds hf trivial value he
      · have hx : (p.flags q).x1=Bool.true := by
          cases hx : (p.flags q).x1
          · have hb := i.isLt
            simp [HostReturn.seedCount,hx] at hb
            exact False.elim (hir hb)
          · rfl
        have hb : i.val<2*p.validated.M := by simpa [HostReturn.seedCount,hx] using i.isLt
        have hrow : i.val-p.validated.M<p.validated.M := by omega
        have heq : p.validated.M+(i.val-p.validated.M)=i.val := by omega
        have hr := hread (i.val-p.validated.M) 0 hrow .seeds1 ⟨hx,hf⟩ trivial value he
        simpa only [roleIndex,LayerNormFwd.Address.region,LayerNormFwd.Address.offset,heq,readVector] using hr
  · cases he : p.outputs.mask with
    | none =>
      have hf : (p.flags q).dropoutMask=Bool.false := by simp [Prepared.flags,he]
      simp [preparedObserve,TypedSource.observe,he,hf]
    | some value =>
      have hm : (p.flags q).dropoutMask=Bool.true := by simp [Prepared.flags,he]
      have hd : (p.flags q).dropout=Bool.true := hdrop (by simp [he])
      simp only [preparedObserve,TypedSource.observe,he,hm,hd,Bool.true_and,ite_true,Option.map_some]
      apply congrArg some
      funext i j
      exact hread i.val j.val i.isLt .mask ⟨hd,hm⟩ j.isLt value he
  · cases hx : p.inputs.x1.isSome with
    | «false» =>
      have hf : (p.flags q).x1=Bool.false := hx
      simp [preparedObserve,TypedSource.observe,hx,hf]
    | «true» =>
      cases he : p.outputs.mask with
      | none =>
        have hf : (p.flags q).dropoutMask=Bool.false := by simp [Prepared.flags,he]
        simp [preparedObserve,TypedSource.observe,hx,he,hf]
      | some value =>
        have hm : (p.flags q).dropoutMask=Bool.true := by simp [Prepared.flags,he]
        have hd : (p.flags q).dropout=Bool.true := hdrop (by simp [he])
        have hf : (p.flags q).x1=Bool.true := hx
        simp only [preparedObserve,TypedSource.observe,hx,he,hm,hd,hf,Bool.true_and,ite_true,Option.map_some]
        apply congrArg some
        funext i j
        exact hread i.val j.val i.isLt .mask1 ⟨hf,hd,hm⟩ j.isLt value he

#axiomsClean prepared_observe_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.WrapperStorage
