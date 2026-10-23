import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.TypedReal
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealDimensionSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem body_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0 < BN)
    (current : Fin BM → Math.Value × Nat) (ready : TypedReal.Ready BM row inner current s)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i j, i < M → j < N → i*N*K+j*K+inner < bounds X) :
    Safety.traceSafeList (TypedReal.backend rest) bounds (Typed.body X O M N K BM BN base index64)
      (s.setReg "start_n" .nat [] (fun _ => cur)) := by
  obtain ⟨hk,hm,hv,hi⟩ := ready
  simp [Typed.body, Typed.rangeBody, TypedSource.dimension, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
    TypedReal.backend, State.setReg, hk, hm, hv, hi, TypedReal.indexed, Reduction.orderedEq,
    TileShape.axisDim, positive, Nat.ne_of_gt positive, TileShape.insertAxisIndex,
    TileShape.eraseAxis, ofFloat, toFloat, bop, numeric, numericLt, numericEq,
    natLt, TileShape.discreteMaximum, TileShape.discreteIntArgMax, Option.bind]
  constructor
  · intro i j hm hn
    exact inputBounds _ _ hm hn
  · intro i hm hn
    exact inputBounds _ _ hm hn

theorem prefix_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base : Nat) (index64 : Bool) (bounds : RegionName → Nat) :
    Safety.traceSafeList (TypedReal.backend rest) bounds
      ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody.take 7) s := by
  cases index64 <;>
    simp [TypedSource.dimension, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      run, step, evalExpr, evalComputeOp, evalOp_unfold,
      TypedReal.backend, State.setReg, ofFloat, toFloat, bop, numeric,
      integralFloorDiv, integralMod, ComputeDType.eraseDType, Option.bind]

theorem tail_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner : Nat) (index64 : Bool)
    (current : Fin BM → Math.Value × Nat) (ready : TypedReal.Ready BM row inner current s)
    (bounds : RegionName → Nat)
    (outputBounds : ∀ i : Fin BM, row+i.val < M → (row+i.val)*K+inner < bounds (Region.cast O)) :
    Safety.traceSafeList (TypedReal.backend rest) bounds
      ((TypedSource.dimension X O M N K BM BN base index64).surfaceBody.drop 8) s := by
  obtain ⟨hk,hm,hv,hi⟩ := ready
  simp [TypedSource.dimension, ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
    Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
    Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
    run, step, evalExpr, evalComputeOp, evalOp_unfold,
    TypedReal.backend, State.setReg, hk, hm, hv, hi, ofFloat, toFloat, bop, numeric,
    natLt, ComputeDType.eraseDType, Option.bind, store, Region.cast]
  intro i hm
  exact outputBounds i hm

theorem range_safe (rest : Algebra Math.Value) (s : State Math.Value) (X : RegionName) (O : Region .int)
    (M N K BM BN base row inner cur : Nat) (index64 : Bool) (positive : 0 < BN)
    (current : Fin BM → Math.Value × Nat) (ready : TypedReal.Ready BM row inner current s)
    (bounds : RegionName → Nat)
    (inputBounds : ∀ i j, i < M → j < N → i*N*K+j*K+inner < bounds X) :
    Safety.rangeSafe (TypedReal.backend rest) bounds "start_n" cur N BN
      (Typed.body X O M N K BM BN base index64) s := by
  rw [Safety.rangeSafe]
  simp only [Nat.ne_of_gt positive, if_false]
  split
  · rename_i active
    refine ⟨body_safe rest s X O M N K BM BN base row inner cur index64 positive
      current ready bounds inputBounds, ?_⟩
    obtain ⟨t, ran, _, next⟩ := TypedReal.body_run rest s X O M N K BM BN base row inner cur
      index64 positive current ready
    rw [ran]
    exact range_safe rest t X O M N K BM BN base row inner (cur+BN) index64 positive
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
    TraceSafe (TypedReal.backend rest) bounds (TypedSource.dimension X O M N K BM BN base index64) s := by
  obtain ⟨a,ha,ma,ra⟩ := TypedReal.prefix_run rest s X O M N K BM BN base index64
  obtain ⟨b,hb,mb,rb⟩ := TypedReal.range_run rest s a ma X O M N K BM BN base
    (((s.pids 0+base)/K)*BM) ((s.pids 0+base)%K) 0 index64 positive _ ra
  let code := (TypedSource.dimension X O M N K BM BN base index64).surfaceBody
  change Safety.traceSafeList (TypedReal.backend rest) bounds code s
  rw [←List.take_append_drop 7 code, Safety.traceSafeList_append]
  refine ⟨prefix_safe rest s X O M N K BM BN base index64 bounds, ?_⟩
  intro initial ran
  have eq : initial = a := Option.some.inj (ran.symm.trans ha)
  subst initial
  have splitCode : code.drop 7 = [.forRange "start_n" 0 N BN (Typed.body X O M N K BM BN base index64)] ++ code.drop 8 := rfl
  rw [splitCode, Safety.traceSafeList_append]
  constructor
  · simp only [Safety.traceSafeList_unfold, Safety.stepSafe_unfold]
    refine ⟨range_safe rest a X O M N K BM BN base (((s.pids 0+base)/K)*BM)
      ((s.pids 0+base)%K) 0 index64 positive _ ra bounds inputBounds, ?_⟩
    cases step (TypedReal.backend rest) (.forRange "start_n" 0 N BN
      (Typed.body X O M N K BM BN base index64)) a <;> trivial
  · intro last ran
    have eq : last = b := by
      apply Option.some.inj
      simpa only [run, step, hb, Option.bind_some] using ran.symm
    subst last
    exact tail_safe rest b X O M N K BM BN base (((s.pids 0+base)/K)*BM)
      ((s.pids 0+base)%K) index64 _ rb bounds outputBounds

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.RealDimensionSafety
