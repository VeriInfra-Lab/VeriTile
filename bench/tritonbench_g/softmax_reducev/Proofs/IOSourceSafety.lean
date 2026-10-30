import bench.tritonbench_g.softmax_reducev.Proofs.FPContract
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.IOSourceSafety
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual active source addresses. The public IO proof derives these bounds
from its logical input and output windows; no numerical law is involved. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  lengths : s.pids 0 < bounds p.lengths.name
  starts : s.pids 0 < bounds p.starts.name
  locations : ∀ start (i : Fin p.tokens), active p s start i →
    locationBase p s + (start+i.val)*p.locationSequenceStride < bounds p.locations.name
  logics : ∀ start (i : Fin p.tokens), active p s start i →
    logicOffset p s start i < bounds p.logics
  values : ∀ start (i : TileIndex [p.tokens,p.dimensions]), active p s start i.1 →
    valueOffset p s start i < bounds p.values
  output : ∀ i : Index p, outputOffset p s i < bounds p.output

theorem prelude_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList M bounds (prelude p) s := by
  simp [prelude,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,hb.lengths,hb.starts]

theorem body_safe (M : Algebra α) (p : Parameters) (s t : State α)
    (start : Nat) (a : Accumulator α p.dimensions) (h : Ready p s t a)
    (hN : 0 < p.tokens) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList M bounds (loopBody p)
      (t.setReg "start_n" .nat [] (fun _ => start)) := by
  simp (maxSteps := 500000) [loopBody,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLe,natLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,h.mem,h.pids,h.undef,h.batch,h.head,
    h.seqlen,h.start,h.lanes,h.dimensions,h.locbase,h.valueptr,h.maximum,h.normalizer,h.output,hN]
  refine ⟨?_,?_,?_⟩
  · intro i hi
    exact hb.locations start i hi
  · intro i hi
    exact hb.logics start i hi
  · intro i d hi
    simpa [valueOffset,valuePointer,location,active,hi,bop,Region.cast] using
      hb.values start (i,d,()) hi

theorem suffix_safe (M : Algebra α) (p : Parameters) (s t : State α)
    (a : Accumulator α p.dimensions) (h : Ready p s t a)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList M bounds (suffix p) t := by
  simp [suffix,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    ComputeDType.eraseDType,numeric,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,h.batch,h.head,h.dimensions,h.output,h.normalizer]
  constructor
  · intro d
    exact hb.output (d,())
  · split <;> trivial

theorem range_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) (hN : 0 < p.tokens)
    (cur : Nat) (t : State α) (a : Accumulator α p.dimensions) (h : Ready p s t a) :
    Safety.rangeSafe M bounds "start_n" cur (sequenceLength p s) p.tokens (loopBody p) t := by
  rw [Safety.rangeSafe_unfold]
  simp only [Nat.ne_of_gt hN,↓reduceIte]
  split
  · refine ⟨body_safe M p s t cur a h hN bounds hb,?_⟩
    obtain ⟨u,hu,hr⟩ := body_run M p s t cur a h hN
    rw [hu]
    exact range_safe M p s bounds hb hN (cur+p.tokens) u _ hr
  · trivial
termination_by sequenceLength p s-cur
decreasing_by omega

theorem source_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) (hN : 0 < p.tokens) :
    TraceSafe M bounds (source p) s := by
  change Safety.traceSafeList M bounds _ s
  rw [source_split]
  simp only [List.append_assoc,List.singleton_append]
  rw [Safety.traceSafeList_append]
  refine ⟨prelude_safe M p s bounds hb,?_⟩
  intro t ht
  obtain ⟨u,hu,hr⟩ := prefix_run M p s
  have eq : u=t := Option.some.inj (hu.symm.trans ht)
  subst u
  obtain ⟨v,hv,vr⟩ := loop_run M p s t 0 (sequenceLength p s) p.tokens _ hr hN
  apply Safety.traceSafeList_cons_of_step (t := v)
  · simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,
      hr.seqlen,Option.bind,Option.bind_some,true_and]
    exact range_safe M p s bounds hb hN 0 t _ hr
  · simp only [step,evalOp_unfold,hr.seqlen,Option.bind,Option.bind_some]
    exact hv
  · exact suffix_safe M p s v _ vr bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.IOSourceSafety
