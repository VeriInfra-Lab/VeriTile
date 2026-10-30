import bench.repairs.tritonbench_g.token_attn_reduceV.Contract
import VeriTile.Triton.Float.StructuralSafety

/-! Safety of the actual metadata loads, dynamic gather loop and unmasked
output store. Bounds are later derived from the public logical IO windows. -/
namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

structure Bounds (p : Parameters) (s : State A) (bounds : RegionName → Nat) : Prop where
  length : s.pids 0*p.lengthStride < bounds (Region.cast p.length)
  start : s.pids 0*p.startStride < bounds (Region.cast p.start)
  request : s.pids 0*p.requestStride < bounds (Region.cast p.request)
  probability : p.block≠0 → ∀ n, n<count p s → probOffset p s n < bounds p.prob
  tokens : p.block≠0 → ∀ n, n<count p s → gatherOffset p s n < bounds (Region.cast p.tokens)
  values : p.block≠0 → ∀ n, n<count p s → ∀ d : Fin p.width,
    valueOffset p s d.val + (s.mem (Region.cast p.tokens) (gatherOffset p s n)).read .nat*p.valueBatchStride < bounds p.value
  output : ∀ d : Fin p.width, outputOffset p s (d,PUnit.unit) < bounds p.output

theorem prelude_safe (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (s : State A) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (withSum M R) bounds (prelude p) s := by
  simp [prelude,kernel,source,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,integralFloorDiv,
    withSum,Region.cast,State.setReg]
  exact ⟨hb.length,hb.start,hb.request⟩

theorem body_safe (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (s t : State A) (n : Nat) (a : TileIndex [p.width] → A)
    (ready : Ready p s t a) (bounds : RegionName → Nat) (hb : Bounds p s bounds) (nonzero : p.block≠0) :
    Safety.traceSafeList (withSum M R) bounds (body p)
      (t.setReg "start_n" .nat [] (fun _ => n)) := by
  simp [body,kernel,source,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,numeric,bop,numericLt,natLt,
    Region.cast,withSum,State.setReg,ready.mem,ready.pids,ready.undef,ready.batch,ready.head,
    ready.offsN,ready.offsD,ready.length,ready.prob,ready.gather,ready.value,ready.accumulator]
  refine ⟨?_,?_,?_⟩
  · intro i hi
    change probOffset p s i.val+n*p.probTokenStride < bounds p.prob
    have eq : probOffset p s i.val+n*p.probTokenStride = probOffset p s (n+i.val) := by
      unfold probOffset; ring
    rw [eq]
    exact hb.probability nonzero (n+i.val) hi
  · intro i hi
    change gatherOffset p s i.val+n*p.requestTokenStride < bounds (Region.cast p.tokens)
    have eq : gatherOffset p s i.val+n*p.requestTokenStride = gatherOffset p s (n+i.val) := by
      unfold gatherOffset; ring
    rw [eq]
    exact hb.tokens nonzero (n+i.val) hi
  · intro i d hi
    have eq : gatherOffset p s i.val+n*p.requestTokenStride = gatherOffset p s (n+i.val) := by
      unfold gatherOffset; ring
    simpa only [hi,if_true,gather_index_toNat,eq] using hb.values nonzero (n+i.val) hi d

theorem postlude_safe (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (s t : State A) (a : TileIndex [p.width] → A) (ready : Ready p s t a)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (withSum M R) bounds (postlude p) t := by
  simp [postlude,kernel,source,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,Option.bind,ComputeDType.eraseDType,
    FloatDType.toTileDType,FloatDType.numericDType,ofFloat,toFloat,numeric,bop,Region.cast,withSum,
    State.setReg,ready.mem,ready.pids,ready.undef,ready.batch,ready.head,ready.offsN,ready.offsD,
    ready.length,ready.prob,ready.gather,ready.value,ready.accumulator]
  exact hb.output

theorem range_safe (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (s t : State A) (start stop : Nat) (a : TileIndex [p.width] → A)
    (ready : Ready p s t a) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.rangeSafe (withSum M R) bounds "start_n" start stop p.block (body p) t := by
  rw [Safety.rangeSafe_unfold]
  by_cases zero : p.block=0
  · simp [zero]
  · simp only [zero,if_false]
    by_cases active : start<stop
    · simp only [active,if_true]
      refine ⟨body_safe M R p s t start a ready bounds hb zero,?_⟩
      obtain ⟨u,ran,next⟩ := body_run M R p s t start a ready
      rw [ran]
      exact range_safe M R p s u (start+p.block) stop (update M R p s start a) next bounds hb
    · simp [active]
termination_by stop-start
decreasing_by omega

theorem source_safe (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (s : State A) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe (withSum M R) bounds (kernel p) s := by
  unfold TraceSafe
  rw [source_split,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe M R p s bounds hb,?_⟩
  intro t ran
  obtain ⟨t',preludeRun,ready⟩ := prelude_run M R p s
  have eq : t=t' := Option.some.inj (ran.symm.trans preludeRun)
  subst t'
  rw [List.singleton_append,Safety.traceSafeList_unfold]
  refine ⟨?_,?_⟩
  · simpa only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,ready.length,
      true_and,Option.some.injEq] using range_safe M R p s t 0 (count p s) (initial M p) ready bounds hb
  · obtain ⟨u,loop,next⟩ := range_run M R p s t 0 (count p s) (initial M p) ready
    simp only [step,evalOp_unfold,ready.length,bind,pure,Option.bind,loop]
    exact postlude_safe M R p s u _ next bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.SourceSafety
