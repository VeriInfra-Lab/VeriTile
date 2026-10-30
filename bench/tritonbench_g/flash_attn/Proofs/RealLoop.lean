import bench.tritonbench_g.flash_attn.Proofs.RealStep

namespace VeriTile.Bench.Repairs.FlashAttn.KernelReal
open VeriTile Triton FP.Structural ActualSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
attribute [local irreducible] Reduction.plan Reduction.Tree.eval

theorem foldRange_invariant (update : Nat→β→β) (limit : Nat) (I : Nat→β→Prop)
    (step : ∀ n a,I n a→n*64<limit→I (n+1) (update (n*64) a))
    (n : Nat) (a : β) (hn : n≤(limit+63)/64) (ha : I n a) :
    I ((limit+63)/64) (foldRange update (n*64) limit 64 a) := by
  by_cases hc : n*64<limit
  · rw [foldRange]
    simp only [show ¬(64:Nat)=0 by decide,if_false,hc,if_true]
    rw [show n*64+64=(n+1)*64 by omega]
    exact foldRange_invariant update limit I step (n+1) (update (n*64) a) (by omega) (step n a ha hc)
  · have he : n=(limit+63)/64 := by omega
    rw [foldRange]
    simp only [show ¬(64:Nat)=0 by decide,if_false,hc]
    exact he ▸ ha
termination_by limit-n*64
decreasing_by omega

theorem accumulated_row (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (pid head : Nat) (i : Fin 128) (col : Fin x.shape.dim) (active : pid*128+i.val<x.shape.length) :
    RowEq x (ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)) i col
      (OnlineMath.running (Spec.pairs x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val)) := by
  let I:=fun n a=>RowEq x a i col (OnlineMath.running
    (Real.prefixPairs x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val n))
  have start : I 0 (initialAcc algebra (parameters x)) := by
    simp [I,RowEq,initialAcc,algebra,binary,zero,WithBot.realAdd,Real.prefixPairs,Real.prefixKeys,OnlineMath.running]
    rfl
  have step : ∀ n a,I n a→n*64<ActualSource.stop (parameters x) (state x m pid head)→
      I (n+1) (update algebra dot (parameters x) (state x m pid head) (some scale) (n*64) a) := by
    intro n a ha _
    dsimp [I] at ha ⊢
    rw [Real.prefix_succ]
    apply update_row x hs m scale pid head n i col active a _ ha
    rw [Real.maximum_append,←Real.prefix_succ]
    exact OnlineMath.maximum_finite _ (Real.prefix_nonempty x m scale _ _ _ _ (n+1) hs.2.2.1 (by omega))
  have final:=foldRange_invariant (update algebra dot (parameters x) (state x m pid head) (some scale))
    (ActualSource.stop (parameters x) (state x m pid head)) I step 0 (initialAcc algebra (parameters x)) (by omega) start
  have rowpid : (pid*128+i.val)/128=pid := by omega
  have count : (ActualSource.stop (parameters x) (state x m pid head)+63)/64=rounds x.shape ((pid*128+i.val)/128) := by
    simp only [ActualSource.stop,parameters,state,stateAt,if_pos rfl,rounds,FlashAttn.stop,ceil,rowpid]
    congr 1
  dsimp [I] at final
  rw [count,Real.prefix_full x m scale _ _ _ _ active] at final
  exact final

theorem output_math (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (pid head : Nat) (i : Fin 128) (col : Fin x.shape.dim) (active : pid*128+i.val<x.shape.length) :
    ActualSource.outputValue algebra (parameters x)
      (ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)) (i,col,())=
      some (Spec.output x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val) := by
  have row:=accumulated_row x hs m scale pid head i col active
  have old:=Real.output_real x hs m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val active
  rw [FlashAttn.outputValue,Real.accumulated_real x hs m scale _ _ _ _ active] at old
  change WithBot.realDiv
    ((ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)).output (i,col,()))
    ((ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)).normalizer i) = _
  rw [row.2.2,row.2.1]
  exact old

theorem dim_positive (x : Layout) (hs : x.shape.Valid) : 0<x.shape.dim := by
  have hd : x.shape.dim=16 ∨ x.shape.dim=32 ∨ x.shape.dim=64 ∨ x.shape.dim=128 := by simpa using hs.2.2.2
  omega

theorem log_math (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (pid head : Nat) (i : Fin 128) (active : pid*128+i.val<x.shape.length) :
    ActualSource.logValue algebra (parameters x)
      (ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)) (i,())=
      some (Spec.lse x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val)) := by
  have row:=accumulated_row x hs m scale pid head i ⟨0,dim_positive x hs⟩ active
  have old:=Real.log_real x hs m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) active
  rw [FlashAttn.logValue,Real.accumulated_real x hs m scale _ _ _ _ active] at old
  change WithBot.realAdd
    ((ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)).maximum i)
    (WithBot.realLog2 ((ActualSource.accumulated algebra dot (parameters x) (state x m pid head) (some scale)).normalizer i)) = _
  rw [row.1,row.2.1]
  exact old

#axiomsClean foldRange_invariant
#axiomsClean accumulated_row
#axiomsClean output_math
#axiomsClean log_math
end
end VeriTile.Bench.Repairs.FlashAttn.KernelReal
