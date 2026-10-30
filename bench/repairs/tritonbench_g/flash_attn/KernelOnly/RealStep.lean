import bench.repairs.tritonbench_g.flash_attn.KernelOnly.RealScores

namespace VeriTile.Bench.Repairs.FlashAttn.KernelReal
open VeriTile Triton FP.Structural ActualSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000
attribute [local irreducible] Reduction.plan Reduction.Tree.eval

/-- One active row of the source tile, related to an independent prefix sum. -/
def RowEq (x : Layout) (a : ActualSource.Acc (parameters x) R) (i : Fin 128)
    (col : Fin x.shape.dim) (st : R×ℝ×ℝ) : Prop :=
  a.maximum i=st.1 ∧ a.normalizer i=some st.2.1 ∧ a.output (i,col,())=some st.2.2

theorem reduce_scores (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (pid head block : Nat) (i : Fin 128) (col : Nat) (active : pid*128+i.val<x.shape.length) :
    algebra.reduceMax (some .fp32) ⟨1,by simp⟩ Bool.false
      (scores algebra dot (parameters x) (state x m pid head) (some scale) (block*64)) (i,())=
      OnlineMath.maximum (Real.blockPairs x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col block) := by
  change reduce max (shape:=[128,64]) ⟨1,by simp⟩ Bool.false
    (scores algebra dot (parameters x) (state x m pid head) (some scale) (block*64)) (i,())=_
  rw [reduce_row]
  rw [show (Reduction.plan 0).eval max
      (fun j=>scores algebra dot (parameters x) (state x m pid head) (some scale) (block*64)
        (i,⟨j%64,Nat.mod_lt _ (by decide)⟩,()))=
      (Reduction.plan 0).eval Real.ops.max
        (fun j=>FlashAttn.score x Real.ops (Real.lift m) (some scale)
          (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) (block*64+j)) by
    apply reduction_congr
    intro j hj
    rw [score_value x m scale pid head (block*64) _ active,
      Real.score_real x hs m scale _ _ _ _ active]
    simp [Nat.mod_eq_of_lt hj]]
  exact Real.max_block x hs m scale _ _ _ col block active 0

#check reduce_scores

theorem update_row (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (pid head block : Nat) (i : Fin 128) (col : Fin x.shape.dim)
    (active : pid*128+i.val<x.shape.length) (a : ActualSource.Acc (parameters x) R)
    (xs : List (ℝ×ℝ)) (before : RowEq x a i col (OnlineMath.running xs))
    (finite : ∃ z : ℝ,OnlineMath.maximum xs ⊔ OnlineMath.maximum
      (Real.blockPairs x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val block)=some z) :
    RowEq x (update algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a) i col
      (OnlineMath.running (xs++Real.blockPairs x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val block)) := by
  obtain ⟨z,hz⟩:=finite
  let blk:=Real.blockPairs x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) col.val block
  let alpha:= (WithBot.realExp2 (WithBot.realSub (OnlineMath.maximum xs) (some z))).unbotD 0
  let prob:=fun j=>if visible x.shape (pid*128+i.val) (block*64+j) then
    pow2 (Spec.score x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) (block*64+j)-z) else 0
  have mx : maximum algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a i=some z := by
    unfold maximum
    rw [reduce_scores x hs m scale pid head block i col.val active,before.1]
    simpa [algebra,binary,OnlineMath.running_max] using hz
  have hp : ∀ j : Fin 64,probability algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a (i,j,())=
      some (prob j.val) := by
    intro j
    unfold probability
    rw [mx,score_value x m scale pid head (block*64) (i,j,()) active]
    exact Real.probability_real _ _ _
  have ha : rescale algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a i=some alpha := by
    unfold rescale
    rw [mx,before.1]
    change Real.exp2 (WithBot.realSub (OnlineMath.running xs).1 (some z))=_
    rw [Real.exp_sub,OnlineMath.running_max]
  have sumP : ((List.range 64).map prob).sum=(blk.map (fun sv=>pow2 (sv.1-z))).sum := by
    have hh:=Real.sum_mask ((List.range 64).map (fun j=>block*64+j)) (visible x.shape (pid*128+i.val))
      (fun k=>pow2 (Spec.score x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) k-z))
    simpa [blk,Real.blockPairs,Real.blockKeys,List.map_map,Function.comp_def,prob] using hh
  have sumV : RealSum.sumReal (fun j=>prob j*(if block*64+j<x.shape.length then
      logical m .v x.v (head/x.shape.heads) (head%x.shape.heads) (block*64+j) col.val else 0)) 64=
      (blk.map (fun sv=>pow2 (sv.1-z)*sv.2)).sum := by
    rw [RealSum.sumReal_list]
    have eq : (fun j=>prob j*(if block*64+j<x.shape.length then
        logical m .v x.v (head/x.shape.heads) (head%x.shape.heads) (block*64+j) col.val else 0))=
        (fun j=>if visible x.shape (pid*128+i.val) (block*64+j) then
          pow2 (Spec.score x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) (block*64+j)-z)*
            logical m .v x.v (head/x.shape.heads) (head%x.shape.heads) (block*64+j) col.val else 0) := by
      funext j
      by_cases vis : visible x.shape (pid*128+i.val) (block*64+j)=Bool.true
      · have hk : block*64+j<x.shape.length := by
          have vh : block*64+j<x.shape.length ∧
              (x.shape.causal=Bool.false ∨ block*64+j≤pid*128+i.val) := by simpa [visible] using vis
          exact vh.1
        simp [prob,vis,hk]
      · simp [prob,vis]
    rw [eq]
    have hh:=Real.sum_mask ((List.range 64).map (fun j=>block*64+j)) (visible x.shape (pid*128+i.val))
      (fun k=>pow2 (Spec.score x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.val) k-z)*
        logical m .v x.v (head/x.shape.heads) (head%x.shape.heads) k col.val)
    simpa [blk,Real.blockPairs,Real.blockKeys,List.map_map,Function.comp_def] using hh
  have den : algebra.reduceSum (some .fp32) ⟨1,by simp⟩ Bool.false
      (probability algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a) (i,())=
      some ((blk.map (fun sv=>pow2 (sv.1-z))).sum) := by
    change reduce WithBot.realAdd (shape:=[128,64]) ⟨1,by simp⟩ Bool.false
      (probability algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a) (i,())=_
    rw [reduce_row]
    have eq : (Reduction.plan 0).eval WithBot.realAdd
        (fun j=>probability algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a
          (i,⟨j%64,Nat.mod_lt _ (by decide)⟩,()))=
        (Reduction.plan 0).eval Real.ops.add (fun j=>some (prob j)) := by
      apply reduction_congr
      intro j hj
      rw [hp]
      simp [Nat.mod_eq_of_lt hj]
    rw [eq,Real.reduce_sum,sumP]
  have pv : contribution algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a (i,col,())=
      some ((blk.map (fun sv=>pow2 (sv.1-z)*sv.2)).sum) := by
    unfold contribution dot
    simp only [ofFloat,toFloat,FloatDType.toTileDType,id_eq,dotAt]
    change some (RealSum.sumReal _ 64)=_
    rw [←sumV]
    congr 1
    apply RealSum.sum_congr
    intro j hj
    simp only [dif_pos hj]
    change (probability algebra dot (parameters x) (state x m pid head) (some scale) (block*64) a (i,⟨j,hj⟩,())).unbotD 0 *
      (ActualSource.value algebra (parameters x) (state x m pid head) (block*64) (⟨j,hj⟩,col,())).unbotD 0=_
    rw [hp,value_value]
    rfl
  have math:=OnlineMath.running_append_block xs blk
  dsimp only at math
  rw [show OnlineMath.maximum xs ⊔ OnlineMath.maximum blk=some z from hz] at math
  rw [RowEq,←math]
  simp only [update,mx,ha,den,pv,before.2.1,before.2.2]
  simp [algebra,binary,zero,WithBot.realAdd,WithBot.realMul,alpha]

#axiomsClean reduce_scores
#axiomsClean update_row
end
end VeriTile.Bench.Repairs.FlashAttn.KernelReal
