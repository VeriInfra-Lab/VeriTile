import bench.tritonbench_g.softmax_reducev.Proofs.FPContract

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.RealModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Range (foldRange)
open FPValue
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

/-- Exact real interpretation of source operations. The reductions are explicit
finite supremum and finite sum; there is no reduction-result parameter. -/
def reductionInputs {α : Type} {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : TileIndex shape → α) (i : TileIndex (TileShape.reduceShape shape ax keep)) :
    Fin (TileShape.axisDim shape ax) → α :=
  match keep with
  | Bool.false => fun k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun k => xs (TileShape.replaceAxisIndex shape ax i k)

def model : Algebra (WithBot ℝ) where
  literal := fun _ _ x => ↑x
  negInf := ⊥
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => a ⊔ b
    | .pow => ↑(a.unbotD 0 ^ b.unbotD 0)
  unary := fun _ op x => match op with
    | .exp => WithBot.realExp x
    | _ => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => ↑(n : ℝ)
  fromInt := fun _ n => ↑(n : ℝ)
  fp32Bits := fun _ => ↑(0 : ℝ)
  fp32Load := id
  reduceMax := fun _ {_} ax keep xs i => Finset.univ.sup (reductionInputs ax keep xs i)
  reduceSum := fun _ {_} ax keep xs i => ↑(∑ k, (reductionInputs ax keep xs i k).unbotD 0)

/-- Negative infinity denotes zero softmax mass, independent of block scheduling. -/
def mass (q : WithBot ℝ) : ℝ := q.elim 0 Real.exp

/-- Independent, unshifted attention definition over every logical key.
The public normalized theorem requires at least one finite key. -/
def target (S : Nat) (q : Nat → WithBot ℝ) (v : Nat → ℝ) : ℝ :=
  (∑ k ∈ Finset.range S, mass (q k) * v k) /
  (∑ k ∈ Finset.range S, mass (q k))

/-- Normalization has positive mass exactly where a finite active logit exists. -/
theorem mass_nonneg (q : WithBot ℝ) : 0 ≤ mass q := by
  cases q with
  | bot => exact le_rfl
  | coe a => exact le_of_lt (Real.exp_pos a)

theorem total_mass_pos (S : Nat) (q : Nat → WithBot ℝ)
    (positive : ∃ k, ∃ r : ℝ, k < S ∧ q k=↑r) :
    0 < ∑ k ∈ Finset.range S, mass (q k) := by
  obtain ⟨k,r,hk,hq⟩ := positive
  apply Finset.sum_pos' (fun _ _ => mass_nonneg _)
  exact ⟨k,Finset.mem_range.mpr hk,by rw [hq]; exact Real.exp_pos r⟩

theorem target_constant (S : Nat) (q : Nat → WithBot ℝ) (value : ℝ)
    (positive : ∃ k, ∃ r : ℝ, k < S ∧ q k=↑r) : target S q (fun _ => value)=value := by
  unfold target
  rw [←Finset.sum_mul, mul_div_cancel_left₀ value (ne_of_gt (total_mass_pos S q positive))]

/-- The new maximum does not alter any maximum above the exact FP32 floor. -/
theorem floor_unchanged (q : ℝ) (finiteRange : -340282346638528859811704183484516925440 ≤ q) :
    (↑q : WithBot ℝ) ⊔ ↑(-340282346638528859811704183484516925440 : ℝ)=↑q := by
  exact sup_eq_left.mpr (WithBot.coe_le_coe.mpr finiteRange)

def weight (S : Nat) (q : Nat → WithBot ℝ) (k : Nat) : ℝ := if k < S then mass (q k) else 0

def prefixSum (S : Nat) (f : Nat → ℝ) (cur : Nat) : ℝ :=
  ∑ k ∈ Finset.range cur, if k < S then f k else 0

def block (S N : Nat) (f : Nat → ℝ) (cur : Nat) : ℝ :=
  ∑ i : Fin N, if cur+i.val < S then f (cur+i.val) else 0

theorem prefix_step (S N : Nat) (f : Nat → ℝ) (cur : Nat) :
    prefixSum S f (cur+N)=prefixSum S f cur+block S N f cur := by
  simp only [prefixSum,block,Finset.sum_range_add]
  congr 1
  exact (Fin.sum_univ_eq_sum_range (fun k => if cur+k < S then f (cur+k) else 0) N).symm

theorem prefix_done (S : Nat) (f : Nat → ℝ) (cur : Nat) (done : S ≤ cur) :
    prefixSum S f cur=∑ k ∈ Finset.range S, f k := by
  have h := Finset.sum_subset (s₁:=Finset.range S) (s₂:=Finset.range cur)
    (f:=fun k => if k < S then f k else 0) (Finset.range_mono done)
    (by intro k hk hn; simp only [Finset.mem_range] at hn; simp [hn])
  rw [prefixSum,←h]
  exact Finset.sum_congr rfl (fun k hk => if_pos (Finset.mem_range.mp hk))

def κ (m : WithBot ℝ) : ℝ := m.elim 0 (fun r => Real.exp (-r))

def qlane (S N : Nat) (q : Nat → WithBot ℝ) (cur : Nat) (i : Fin N) : WithBot ℝ :=
  if cur+i.val < S then q (cur+i.val) else ⊥

def nextMax (S N : Nat) (q : Nat → WithBot ℝ) (cur : Nat) (m : WithBot ℝ) : WithBot ℝ :=
  (Finset.univ.sup (qlane S N q cur) ⊔ m) ⊔
    (↑(-340282346638528859811704183484516925440 : ℝ) : WithBot ℝ)

theorem nextMax_finite (S N : Nat) (q : Nat → WithBot ℝ) (cur : Nat) (m : WithBot ℝ)
    (hN : 0 < N) (reached : cur < S) : ∃ r : ℝ, nextMax S N q cur m=↑r := by
  have le : (↑(-340282346638528859811704183484516925440 : ℝ) : WithBot ℝ) ≤
      nextMax S N q cur m := le_sup_right
  have nonbot : nextMax S N q cur m≠⊥ := by
    intro h
    rw [h] at le
    exact WithBot.coe_ne_bot (le_antisymm le bot_le)
  obtain ⟨r,hr⟩ := WithBot.ne_bot_iff_exists.mp nonbot
  exact ⟨r,hr.symm⟩

theorem rescale (m : WithBot ℝ) (r P : ℝ) (empty : m=⊥ → P=0) :
    κ m * P * (WithBot.realExp (WithBot.realSub m (↑r))).unbotD 0 =
      Real.exp (-r)*P := by
  cases m with
  | bot => simp [κ,empty rfl]
  | coe a =>
    simp only [κ,WithBot.realSub_coe_coe,WithBot.realExp_coe,WithBot.unbotD_coe]
    change Real.exp (-a)*P*Real.exp (a-r)=Real.exp (-r)*P
    have h : Real.exp (-a)*Real.exp (a-r)=Real.exp (-r) := by
      rw [←Real.exp_add]; congr 1; ring
    calc Real.exp (-a) * P * Real.exp (a-r) = (Real.exp (-a)*Real.exp (a-r))*P := by ring
         _ = Real.exp (-r)*P := by rw [h]

theorem shifted (S N : Nat) (q : Nat → WithBot ℝ) (cur : Nat) (r : ℝ) (i : Fin N) :
    WithBot.realExp (WithBot.realSub (qlane S N q cur i) ↑r)=
      ↑(if cur+i.val < S then Real.exp (-r)*mass (q (cur+i.val)) else 0) := by
  by_cases active : cur+i.val < S
  · simp only [qlane,if_pos active]
    cases hq : q (cur+i.val) with
    | bot =>
      change WithBot.realExp (WithBot.realSub ⊥ ↑r) = ↑(Real.exp (-r)*0)
      rw [WithBot.realSub_bot_left,WithBot.realExp_bot,mul_zero]
    | coe a =>
      change (↑(Real.exp (a-r)) : WithBot ℝ) = ↑(Real.exp (-r)*Real.exp a)
      congr 1
      rw [←Real.exp_add]; congr 1; ring
  · rw [qlane,if_neg active,WithBot.realSub_bot_left,WithBot.realExp_bot,if_neg active]

/-- A single source block, including its exact partial-tail masks. -/
def step (S N D : Nat) (q : Nat → WithBot ℝ) (v : Nat → Fin D → ℝ)
    (cur : Nat) (old : Accumulator (WithBot ℝ) D) : Accumulator (WithBot ℝ) D :=
  let m := nextMax S N q cur old.maximum
  let scale := WithBot.realExp (WithBot.realSub old.maximum m)
  let p := fun i => WithBot.realExp (WithBot.realSub (qlane S N q cur i) m)
  { maximum := m
    normalizer := WithBot.realAdd (WithBot.realMul old.normalizer scale)
      (↑(∑ i : Fin N, (p i).unbotD 0))
    output := fun d => WithBot.realAdd (WithBot.realMul (old.output d) scale)
      (↑(∑ i : Fin N, (WithBot.realMul (p i)
        (↑(if cur+i.val < S then v (cur+i.val) d.1 else 0))).unbotD 0)) }

def Good (S D : Nat) (q : Nat → WithBot ℝ) (v : Nat → Fin D → ℝ) (cur : Nat)
    (a : Accumulator (WithBot ℝ) D) : Prop :=
  a.normalizer=↑(κ a.maximum * prefixSum S (fun k => mass (q k)) cur) ∧
  (∀ d, a.output d=↑(κ a.maximum * prefixSum S (fun k => mass (q k)*v k d.1) cur)) ∧
  (cur=0 ∨ a.maximum≠⊥)

theorem exp_sub_coe (m : WithBot ℝ) (r : ℝ) :
    WithBot.realExp (WithBot.realSub m ↑r)=
      ↑((WithBot.realExp (WithBot.realSub m ↑r)).unbotD 0) := by
  cases m with
  | bot => rw [WithBot.realSub_bot_left,WithBot.realExp_bot]; rfl
  | coe a => rw [WithBot.realSub_coe_coe,WithBot.realExp_coe]; rfl

theorem shifted_sum (S N : Nat) (q : Nat → WithBot ℝ) (cur : Nat) (r : ℝ) :
    (∑ i : Fin N, (WithBot.realExp (WithBot.realSub (qlane S N q cur i) ↑r)).unbotD 0)=
      Real.exp (-r)*block S N (fun k => mass (q k)) cur := by
  simp_rw [shifted,WithBot.unbotD_coe]
  rw [block,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  split_ifs <;> simp

theorem shifted_products (S N D : Nat) (q : Nat → WithBot ℝ) (v : Nat → Fin D → ℝ)
    (cur : Nat) (r : ℝ) (d : Fin D) :
    (∑ i : Fin N, (WithBot.realMul
      (WithBot.realExp (WithBot.realSub (qlane S N q cur i) ↑r))
      (↑(if cur+i.val < S then v (cur+i.val) d else 0))).unbotD 0)=
      Real.exp (-r)*block S N (fun k => mass (q k)*v k d) cur := by
  simp_rw [shifted,WithBot.realMul_coe_coe,WithBot.unbotD_coe]
  rw [block,Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i hi
  split_ifs <;> simp [mul_assoc]

theorem step_good (S N D : Nat) (q : Nat → WithBot ℝ) (v : Nat → Fin D → ℝ)
    (cur : Nat) (a : Accumulator (WithBot ℝ) D) (hN : 0 < N) (reached : cur < S)
    (good : Good S D q v cur a) : Good S D q v (cur+N) (step S N D q v cur a) := by
  obtain ⟨r,hr⟩ := nextMax_finite S N q cur a.maximum hN reached
  have empty : a.maximum=⊥ → cur=0 := by
    intro e
    rcases good.2.2 with h | h
    · exact h
    · exact (h e).elim
  have scaled (f : Nat → ℝ) :
      κ a.maximum * prefixSum S f cur *
        (WithBot.realExp (WithBot.realSub a.maximum ↑r)).unbotD 0 =
      Real.exp (-r)*prefixSum S f cur :=
    rescale a.maximum r _ (fun h => by rw [empty h]; simp [prefixSum])
  refine ⟨?_,?_,Or.inr ?_⟩
  · change WithBot.realAdd (WithBot.realMul a.normalizer _) _ = _
    simp only [step,hr,good.1]
    rw [exp_sub_coe a.maximum r,WithBot.realMul_coe_coe,WithBot.realAdd_coe_coe]
    congr 1
    rw [scaled,shifted_sum,prefix_step]
    change Real.exp (-r)*_+Real.exp (-r)*_=Real.exp (-r)*(_+_)
    ring
  · intro d
    change WithBot.realAdd (WithBot.realMul (a.output d) _) _ = _
    simp only [step,hr,good.2.1 d]
    rw [exp_sub_coe a.maximum r,WithBot.realMul_coe_coe,WithBot.realAdd_coe_coe]
    congr 1
    rw [scaled,shifted_products,prefix_step]
    change Real.exp (-r)*_+Real.exp (-r)*_=Real.exp (-r)*(_+_)
    ring
  · change nextMax S N q cur a.maximum≠⊥
    rw [hr]; exact WithBot.coe_ne_bot

theorem loop_good (S N D : Nat) (q : Nat → WithBot ℝ) (v : Nat → Fin D → ℝ)
    (cur : Nat) (a : Accumulator (WithBot ℝ) D) (hN : 0 < N)
    (good : Good S D q v cur a) :
    ∃ done, S ≤ done ∧ Good S D q v done (foldRange (step S N D q v) cur S N a) := by
  by_cases reached : cur < S
  · rw [foldRange]
    simp only [dif_neg (Nat.ne_of_gt hN),dif_pos reached]
    exact loop_good S N D q v (cur+N) (step S N D q v cur a) hN
      (step_good S N D q v cur a hN reached good)
  · exact ⟨cur,Nat.le_of_not_gt reached,by simpa [foldRange,Nat.ne_of_gt hN,reached] using good⟩
termination_by S-cur
decreasing_by omega

/-- All blocks, including a final partial block, equal the independent
unshifted softmax-weighted sum. A finite final maximum is derived from S>0. -/
theorem mathematical (S N D : Nat) (q : Nat → WithBot ℝ) (v : Nat → Fin D → ℝ)
    (hN : 0 < N) (hS : 0 < S) (d : TileIndex [D]) :
    let a := foldRange (step S N D q v) 0 S N ⟨⊥,↑(0:ℝ),fun _ => ↑(0:ℝ)⟩
    WithBot.realDiv (a.output d) a.normalizer=↑(target S q (fun k => v k d.1)) := by
  obtain ⟨done,hd,good⟩ := loop_good S N D q v 0 ⟨⊥,↑(0:ℝ),fun _ => ↑(0:ℝ)⟩ hN
    (by simp [Good,κ,prefixSum])
  have ne : (foldRange (step S N D q v) 0 S N ⟨⊥,↑(0:ℝ),fun _ => ↑(0:ℝ)⟩).maximum≠⊥ := by
    rcases good.2.2 with e | e
    · omega
    · exact e
  obtain ⟨m,hm⟩ := WithBot.ne_bot_iff_exists.mp ne
  dsimp only
  rw [good.1,good.2.1 d,WithBot.realDiv_coe_coe,prefix_done S _ done hd,prefix_done S _ done hd]
  rw [←hm]
  apply congrArg (fun x : ℝ => (x : WithBot ℝ))
  change (Real.exp (-m)*_)/(Real.exp (-m)*_)=target S q (fun k => v k d.1)
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero (-m))]
  rfl

#axiomsClean total_mass_pos
#axiomsClean target_constant
#axiomsClean floor_unchanged
#axiomsClean step_good
#axiomsClean loop_good
#axiomsClean mathematical

#axiomsClean prefix_step
#axiomsClean prefix_done
#axiomsClean nextMax_finite
#axiomsClean rescale
#axiomsClean shifted
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.RealModel
