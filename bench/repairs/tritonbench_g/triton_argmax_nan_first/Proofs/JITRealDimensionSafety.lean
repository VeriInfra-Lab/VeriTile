import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.JITRealDimension
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealDimensionSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

theorem body_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0 < BN)
    (current : Fin BM → Math.Value × Nat) (ready : JITRealDimension.Ready M N K BM row inner current s)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i j, i < M → j < N → i*N*K+j*K+inner < bounds X) :
    Safety.traceSafeList (TypedReal.backend rest) bounds (JITRealDimension.body X O BM BN index64)
      (s.setReg "start_n" .int [] (fun _ => (cur : Int))) := by
  obtain ⟨hM,hN,hK,hk,hm,hv,hi⟩ := ready
  simp [JITRealDimension.body, JITRealDimension.rangeBody, JITSource.dimension, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
    TypedReal.backend, hM, hN, hK, hk, hm, hv, hi, TypedReal.indexed, Reduction.orderedEq,
    TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
    TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
    natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax, Option.bind,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  try simp only [← apply_ite (fun n : Nat => (n : Int)), ← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast]
  constructor
  · intro i j hm hn
    exact inputBounds _ _ hm hn
  · intro i hm hn
    exact inputBounds _ _ hm hn

theorem prefix_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (bounds : RegionName → Nat) :
    Safety.traceSafeList (TypedReal.backend rest) bounds
      ((JITSource.dimension X O BM BN index64).surfaceBody.take 7) (JITRealDimension.arguments s M N K base) := by
  cases index64 <;>
    simp [JITSource.dimension, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold,
      TypedReal.backend, JITRealDimension.arguments, ofFloat, toFloat, bop, numeric, numericLt, natLt, IntegerExpr.sameShape,
      integralFloorDiv, integralMod, ComputeDType.eraseDType, Option.bind,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]

theorem tail_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner : Nat) (index64 : Bool)
    (current : Fin BM → Math.Value × Nat) (ready : JITRealDimension.Ready M N K BM row inner current s)
    (bounds : RegionName → Nat)
    (outputBounds : ∀ i : Fin BM, row+i.val < M → (row+i.val)*K+inner < bounds (Region.cast O)) :
    Safety.traceSafeList (TypedReal.backend rest) bounds
      ((JITSource.dimension X O BM BN index64).surfaceBody.drop 8) s := by
  obtain ⟨hM,hN,hK,hk,hm,hv,hi⟩ := ready
  simp [JITSource.dimension, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    run, step, evalExpr, evalComputeOp, evalOp_unfold,
    TypedReal.backend, hM, hN, hK, hk, hm, hv, hi, ofFloat, toFloat, bop, numeric,
    numericLt, natLt, ComputeDType.eraseDType, Option.bind, store, Region.cast,-Nat.cast_add,-Nat.cast_mul,← Int.natCast_add,← Int.natCast_mul]
  intro i hm
  exact outputBounds i hm

theorem range_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0 < BN)
    (current : Fin BM → Math.Value × Nat) (ready : JITRealDimension.Ready M N K BM row inner current s)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i j, i < M → j < N → i*N*K+j*K+inner < bounds X) :
    Safety.rangeIntSafe (TypedReal.backend rest) bounds "start_n" (cur : Int) (N : Int) BN
      (JITRealDimension.body X O BM BN index64) s := by
  rw [Safety.rangeIntSafe]
  simp only [Nat.ne_of_gt positive, if_false]
  split
  · rename_i active
    refine ⟨body_safe rest s X O M N K BM BN base row inner cur index64 positive
      current ready bounds inputBounds, ?_⟩
    obtain ⟨t, ran, _, next⟩ := JITRealDimension.body_run rest s X O M N K BM BN base row inner cur
      index64 positive current ready
    rw [ran]
    simpa only [← Int.natCast_add] using range_safe rest t X O M N K BM BN base row inner (cur+BN) index64 positive
      _ next bounds inputBounds
  · trivial
termination_by N-cur
decreasing_by omega

theorem source_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (positive : 0 < BN)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i j, i < M → j < N →
      i*N*K+j*K+(s.pids 0+base)%K < bounds X)
    (outputBounds : ∀ i : Fin BM, ((s.pids 0+base)/K)*BM+i.val < M →
      (((s.pids 0+base)/K)*BM+i.val)*K+(s.pids 0+base)%K < bounds (Region.cast O)) :
    TraceSafe (TypedReal.backend rest) bounds (JITSource.dimension X O BM BN index64) (JITRealDimension.arguments s M N K base) := by
  obtain ⟨a,ha,ma,ra⟩ := JITRealDimension.prefix_run rest s X O M N K BM BN base index64
  obtain ⟨b,hb,mb,rb⟩ := JITRealDimension.range_run rest s a ma X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) 0 index64 positive _ ra
  let code := (JITSource.dimension X O BM BN index64).surfaceBody
  change Safety.traceSafeList (TypedReal.backend rest) bounds code (JITRealDimension.arguments s M N K base)
  rw [←List.take_append_drop 7 code, Safety.traceSafeList_append]
  refine ⟨prefix_safe rest s X O M N K BM BN base index64 bounds, ?_⟩
  intro initial ran
  have eq : initial = a := Option.some.inj (ran.symm.trans ha)
  subst initial
  have splitCode : code.drop 7 = [.forRangeInt "start_n" (.constInt 0) (.ref .int [] "N") (.constNat BN) (JITRealDimension.body X O BM BN index64)] ++ code.drop 8 := rfl
  rw [splitCode, Safety.traceSafeList_append]
  have bound := ra.2.1
  constructor
  · simp only [Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeOp_unfold,evalOp_unfold,bound,true_and]
    refine ⟨range_safe rest a X O M N K BM BN base (((s.pids 0+base)/K)*BM)
      ((s.pids 0+base)%K) 0 index64 positive _ ra bounds inputBounds, ?_⟩
    cases step (TypedReal.backend rest) (.forRangeInt "start_n" (.constInt 0) (.ref .int [] "N") (.constNat BN)
      (JITRealDimension.body X O BM BN index64)) a <;> trivial
  · intro last ran
    have eq : last = b := by
      apply Option.some.inj
      simpa only [run, step, evalOp_unfold, bound, bind, pure, Option.bind, ← Int.natCast_zero, hb, Option.bind_some] using ran.symm
    subst last
    exact tail_safe rest b X O M N K BM BN base (((s.pids 0+base)/K)*BM)
      ((s.pids 0+base)%K) index64 _ rb bounds outputBounds

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITRealDimensionSafety
