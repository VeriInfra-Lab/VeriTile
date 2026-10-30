import bench.tritonbench_g.flash_attn.Proofs.RealBackend
-- Preserve the historical Bool spellings after importing the shared Triton DSL.
local notation "true" => Bool.true
local notation "false" => Bool.false
namespace VeriTile.Bench.Repairs.FlashAttn.Real
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

def prefixKeys (s : Shape) (row count : Nat) : List Nat :=
  (List.range (count*64)).filter (visible s row)
def blockKeys (s : Shape) (row block : Nat) : List Nat :=
  ((List.range 64).map (fun j=>block*64+j)).filter (visible s row)
def prefixPairs (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col count : Nat) :=
  (prefixKeys x.shape row count).map (fun k=>(Spec.score x m scale b h row k,logical m .v x.v b h k col))
def blockPairs (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col block : Nat) :=
  (blockKeys x.shape row block).map (fun k=>(Spec.score x m scale b h row k,logical m .v x.v b h k col))

theorem prefix_succ (x : Layout) (m : Memory ℝ) (scale : ℝ) (b h row col n : Nat) :
    prefixPairs x m scale b h row col (n+1)=
      prefixPairs x m scale b h row col n++blockPairs x m scale b h row col n := by
  simp [prefixPairs,prefixKeys,blockPairs,blockKeys,Nat.add_mul,List.range_add,List.filter_append,List.map_append]

def embed (st : R × ℝ × ℝ) : Acc R := ⟨fun _=>st.1,fun _=>some st.2.1,some st.2.2⟩

theorem max_mask (xs : List Nat) (pred : Nat → Bool) (s v : Nat → ℝ) :
    (xs.map (fun k=>if pred k then some (s k) else (none : R))).foldr (fun (a b : R)=>a⊔b) ⊥=
      OnlineMath.maximum ((xs.filter pred).map (fun k=>(s k,v k))) := by
  induction xs with
  | nil=>rfl
  | cons k ks ih=>
    simp only [OnlineMath.maximum,List.map_map,Function.comp_def] at ih ⊢
    cases hk : pred k <;> simp only [List.map_cons,List.foldr_cons,List.filter_cons,hk,
      Bool.false_eq_true,ite_false,ite_true,List.map_cons,List.foldr_cons]
    · change (⊥ : R) ⊔ _=_
      rw [bot_sup_eq]
      exact ih
    · simpa only [List.foldr_cons] using congrArg (fun z : R=>(s k : R) ⊔ z) ih

theorem sum_mask (xs : List Nat) (pred : Nat → Bool) (f : Nat → ℝ) :
    (xs.map (fun k=>if pred k then f k else 0)).sum=((xs.filter pred).map f).sum := by
  induction xs with
  | nil=>rfl
  | cons k ks ih=>cases hk : pred k <;> simp [hk,ih]

theorem max_block (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row col block : Nat) (hr : row<x.shape.length) (o : Fin 4) :
    (Reduction.plan o).eval ops.max (fun j=>score x ops (lift m) (some scale) b h row (block*64+j))=
      OnlineMath.maximum (blockPairs x m scale b h row col block) := by
  rw [reduce_max]
  simp_rw [score_real x hs m scale b h row _ hr]
  have hm:=max_mask ((List.range 64).map (fun j=>block*64+j)) (visible x.shape row)
    (Spec.score x m scale b h row) (fun k=>logical m .v x.v b h k col)
  simpa [blockPairs,blockKeys,List.map_map,Function.comp_def] using hm

theorem exp_sub (x y : R) :
    exp2 (WithBot.realSub x y)=some ((WithBot.realExp2 (WithBot.realSub x y)).unbotD 0) := by
  cases x <;> cases y <;> simp [exp2,WithBot.realSub,WithBot.realExp2,pow2,mul_comm]

theorem probability_real (s z : ℝ) (p : Bool) :
    exp2 (WithBot.realSub (if p then some s else none) (some z))=
      some (if p then pow2 (s-z) else 0) := by
  cases p <;> simp [exp2,WithBot.realSub]

theorem block_eq (x : Layout) (hs : x.shape.Valid) (m : Memory ℝ) (scale : ℝ)
    (b h row col block : Nat) (hr : row<x.shape.length) (xs : List (ℝ × ℝ))
    (finite : ∃ z : ℝ, OnlineMath.maximum xs ⊔ OnlineMath.maximum (blockPairs x m scale b h row col block)=some z) :
    blockStep x ops (lift m) (some scale) b h row col (embed (OnlineMath.running xs)) block=
      embed (OnlineMath.running (xs++blockPairs x m scale b h row col block)) := by
  obtain ⟨z,hz⟩:=finite
  let blk:=blockPairs x m scale b h row col block
  let α:= (WithBot.realExp2 (WithBot.realSub (OnlineMath.maximum xs) (some z))).unbotD 0
  let prob:=fun j=>if visible x.shape row (block*64+j) then
    pow2 (Spec.score x m scale b h row (block*64+j)-z) else 0
  have mx : (fun o=>ops.max ((embed (OnlineMath.running xs)).maximum o)
      ((Reduction.plan o).eval ops.max (fun j=>score x ops (lift m) (some scale) b h row (block*64+j))))=
      (fun _=>some z) := by
    funext o
    rw [max_block x hs m scale b h row col block hr o]
    simpa [embed,ops,OnlineMath.running_max] using hz
  have ha : ops.exp2 (ops.sub (OnlineMath.running xs).1 (some z))=some α := by
    change exp2 (WithBot.realSub _ _)=_
    rw [exp_sub,OnlineMath.running_max]
  have hp : (fun j=>ops.exp2 (ops.sub (score x ops (lift m) (some scale) b h row (block*64+j)) (some z)))=
      (fun j=>some (prob j)) := by
    funext j
    rw [score_real x hs m scale b h row _ hr]
    exact probability_real _ _ _
  have sumP : ((List.range 64).map prob).sum=(blk.map (fun sv=>pow2 (sv.1-z))).sum := by
    have hh:=sum_mask ((List.range 64).map (fun j=>block*64+j)) (visible x.shape row)
      (fun k=>pow2 (Spec.score x m scale b h row k-z))
    simpa [blk,blockPairs,blockKeys,List.map_map,Function.comp_def,prob] using hh
  have sumV : RealSum.sumReal (fun j=>prob j*(if block*64+j<x.shape.length then
      logical m .v x.v b h (block*64+j) col else 0)) 64=
      (blk.map (fun sv=>pow2 (sv.1-z)*sv.2)).sum := by
    rw [RealSum.sumReal_list]
    have eq : (fun j=>prob j*(if block*64+j<x.shape.length then logical m .v x.v b h (block*64+j) col else 0))=
        (fun j=>if visible x.shape row (block*64+j) then
          pow2 (Spec.score x m scale b h row (block*64+j)-z)*logical m .v x.v b h (block*64+j) col else 0) := by
      funext j
      by_cases vis : visible x.shape row (block*64+j)=true
      · have hk : block*64+j<x.shape.length := by
          have hh : block*64+j<x.shape.length ∧ (x.shape.causal=false ∨ block*64+j≤row) := by simpa [visible] using vis
          exact hh.1
        simp [prob,vis,hk]
      · simp [prob,vis]
    rw [eq]
    have hh:=sum_mask ((List.range 64).map (fun j=>block*64+j)) (visible x.shape row)
      (fun k=>pow2 (Spec.score x m scale b h row k-z)*logical m .v x.v b h k col)
    simpa [blk,blockPairs,blockKeys,List.map_map,Function.comp_def] using hh
  have hv : value x ops (lift m) b h=fun k d=>some (if k<x.shape.length then logical m .v x.v b h k d else 0) :=
    funext (fun k=>funext (value_real x m b h k))
  have update:=OnlineMath.running_append_block xs blk
  dsimp only at update
  rw [show OnlineMath.maximum xs ⊔ OnlineMath.maximum blk=some z from hz] at update
  have converted:=congrArg embed update
  rw [←converted]
  have mxAt := fun o=>congrFun mx o
  have hpAt := fun j=>congrFun hp j
  have cast : ∀ a : R, ops.widen (ops.narrow a)=a := fun _=>rfl
  have mul : ∀ a b : ℝ, ops.mul (some a) (some b)=some (a*b) := fun _ _=>rfl
  have fma : ∀ a b c : ℝ, ops.fma (some a) (some b) (some c)=some (a*b+c) := fun _ _ _=>rfl
  have den : ∀ o,(Reduction.plan o).eval ops.add (fun j=>some (prob j))=
      (some ((blk.map (fun sv=>pow2 (sv.1-z))).sum) : R) := by
    intro o
    exact (reduce_sum o prob).trans (congrArg (fun z : ℝ=>(some z : R)) sumP)
  simp only [blockStep,mxAt]
  simp only [embed,ha,hpAt,cast,hv,den]
  simp only [show ops.zero=(some 0 : R) from rfl,fma,mul,mul_zero,zero_add]
  rw [dot_some]
  simp only [show 64/8*8=64 from rfl,sumV]
  rfl

#axiomsClean block_eq
end
end VeriTile.Bench.Repairs.FlashAttn.Real
