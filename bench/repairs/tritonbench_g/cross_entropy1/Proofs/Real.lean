import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Math
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Execution

namespace VeriTile.Bench.Repairs.CrossEntropy1.RealSource
open VeriTile.Triton FP.Structural Source
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Ideal-real interpretation of the same typed source used by the FP proof.
The bottom element is the source's masked negative-infinity sentinel. -/
def algebra : FP32 (WithBot ℝ) where
  literal := fun x => ↑x
  fromNat := fun n => ↑(n : ℝ)
  negInf := ⊥
  load := id
  add := Option.map₂ (·+·)
  sub := Option.map₂ (·-·)
  mul := Option.map₂ (·*·)
  div := Option.map₂ (·/·)
  exp := fun value => match value with
    | none => ↑(0 : ℝ)
    | some x => ↑(Real.exp x)
  log := Option.map Real.log
  sum := fun f => ∑ i, f i
  maximum := fun f => Finset.univ.sup f

@[simp] theorem fp_coe (x : ℝ) : fp algebra x=↑x := rfl
@[simp] theorem nat_coe (n : Nat) : nat algebra n=↑(n : ℝ) := rfl
@[simp] theorem add_coe (x y : ℝ) : binary algebra .add (↑x) (↑y)=↑(x+y) := rfl
@[simp] theorem sub_coe (x y : ℝ) : binary algebra .sub (↑x) (↑y)=↑(x-y) := rfl
@[simp] theorem mul_coe (x y : ℝ) : binary algebra .mul (↑x) (↑y)=↑(x*y) := rfl
@[simp] theorem div_coe (x y : ℝ) : binary algebra .div (↑x) (↑y)=↑(x/y) := rfl
@[simp] theorem exp_coe (x : ℝ) : unary algebra .exp (↑x)=↑(Real.exp x) := rfl
@[simp] theorem log_coe (x : ℝ) : unary algebra .log (↑x)=↑(Real.log x) := rfl

/-- Entry reads bind the real interpretation to the logical caller row. -/
def Input (p : Parameters) (memory : Memory (WithBot ℝ)) (row : Nat) (x : Nat → ℝ) : Prop :=
  ∀ col, col<p.cols → (memory p.logits (logitAddress p row col)).read .real=↑(x col)

theorem loaded (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (lane : Fin p.block) :
    loadedLogits algebra p memory row block lane=
      if column p block lane.val<p.cols then ↑(x (column p block lane.val)) else ⊥ := by
  unfold loadedLogits
  split
  · rename_i h
    simp [load,input _ h,algebra]
  · rfl

theorem maximum_finite (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (positive : 0<p.block)
    (start : block*p.block<p.cols) :
    ∃ m : ℝ, maximum algebra (loadedLogits algebra p memory row block)=↑m := by
  have lane : loadedLogits algebra p memory row block ⟨0,positive⟩=↑(x (block*p.block)) := by
    rw [loaded p memory row block x input]
    simp [column,start]
  have lower := Finset.le_sup (f:=loadedLogits algebra p memory row block)
    (Finset.mem_univ (⟨0,positive⟩ : Fin p.block))
  rw [lane] at lower
  have nonbot : maximum algebra (loadedLogits algebra p memory row block)≠⊥ := by
    intro h
    change _≤maximum algebra (loadedLogits algebra p memory row block) at lower
    rw [h] at lower
    exact (WithBot.not_coe_le_bot _) lower
  obtain ⟨m,hm⟩ := WithBot.ne_bot_iff_exists.mp nonbot
  exact ⟨m,hm.symm⟩

theorem masked_shift (classes width block : Nat) (positive : 0<width)
    (start : block*width<classes) (x : Nat → ℝ) (m : ℝ) :
    Real.log (∑ lane ∈ Finset.range width,
      if block*width+lane<classes then Real.exp (x (block*width+lane)-m) else 0)+m=
      Math.blockLSE classes width block x := by
  let shifted := ∑ lane ∈ Finset.range width,
      if block*width+lane<classes then Real.exp (x (block*width+lane)-m) else 0
  have pos : 0<shifted := by
    apply Finset.sum_pos'
    · intro i _
      split
      · exact le_of_lt (Real.exp_pos _)
      · exact le_refl _
    · refine ⟨0,by simp [positive],?_⟩
      simp only [Nat.add_zero,if_pos start]
      exact Real.exp_pos _
  have factor : Math.blockSum classes width block (fun j => Real.exp (x j))=Real.exp m*shifted := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    split
    · rw [← Real.exp_add]
      ring
    · rw [mul_zero]
  rw [Math.blockLSE,factor,Real.log_mul (ne_of_gt (Real.exp_pos m)) (ne_of_gt pos),Real.log_exp]
  exact add_comm _ _

theorem forward_lse (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (positive : 0<p.block)
    (start : block*p.block<p.cols) :
    (forwardValues algebra p memory row block).lse=↑(Math.blockLSE p.cols p.block block x) := by
  obtain ⟨m,hm⟩ := maximum_finite p memory row block x input positive start
  have each : ∀ lane : Fin p.block,
      unary algebra .exp (binary algebra .sub (loadedLogits algebra p memory row block lane) (↑m))=
        ↑(if column p block lane.val<p.cols then Real.exp (x (column p block lane.val)-m) else 0) := by
    intro lane
    rw [loaded p memory row block x input]
    split <;> rfl
  have sumEq : sum algebra (fun lane => unary algebra .exp
      (binary algebra .sub (loadedLogits algebra p memory row block lane) (↑m)))=
      ↑(∑ lane ∈ Finset.range p.block,
        if block*p.block+lane<p.cols then Real.exp (x (block*p.block+lane)-m) else 0) := by
    simp_rw [each]
    change (∑ i : Fin p.block, (↑(if column p block i.val<p.cols then Real.exp (x (column p block i.val)-m) else 0) : WithBot ℝ))=_
    rw [← WithBot.coe_sum]
    congr 1
    exact Fin.sum_univ_eq_sum_range (fun i => if block*p.block+i<p.cols then Real.exp (x (block*p.block+i)-m) else 0) p.block
  change binary algebra .add (unary algebra .log (sum algebra _))
    (maximum algebra (loadedLogits algebra p memory row block))=_
  rw [hm,sumEq]
  change (↑(Real.log (∑ lane ∈ Finset.range p.block, if block*p.block+lane<p.cols then Real.exp (x (block*p.block+lane)-m) else 0)+m) : WithBot ℝ)=_
  exact congrArg (fun y : ℝ => (↑y : WithBot ℝ))
    (masked_shift p.cols p.block block positive start x m)

theorem forward_sum (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (smooth : p.hasSmoothing=Bool.true) :
    (forwardValues algebra p memory row block).sumLogits=
      some (↑(Math.blockSum p.cols p.block block x)) := by
  have each : ∀ lane : Fin p.block,
      (if column p block lane.val<p.cols then loadedLogits algebra p memory row block lane else fp algebra 0)=
        (↑(if column p block lane.val<p.cols then x (column p block lane.val) else 0) : WithBot ℝ) := by
    intro lane
    rw [loaded p memory row block x input]
    split <;> simp_all [fp,algebra]
  simp only [forwardValues,smooth,↓reduceIte]
  congr 1
  change sum algebra _=_
  simp_rw [each]
  change (∑ i : Fin p.block, (↑(if column p block i.val<p.cols then x (column p block i.val) else 0) : WithBot ℝ))=_
  rw [← WithBot.coe_sum]
  congr 1
  exact Fin.sum_univ_eq_sum_range (fun i => if block*p.block+i<p.cols then x (block*p.block+i) else 0) p.block

/-- Contribution stored by one split block before host summation. -/
def partialLoss (p : Parameters) (x : Nat → ℝ) (target : Int) (block : Nat) : ℝ :=
  if target=p.ignored then 0 else
    -p.smoothing*Math.blockSum p.cols p.block block x/p.totalClasses-
      if (block*p.block : Int)≤target-p.classStart ∧
          target-p.classStart<(min p.cols ((block+1)*p.block) : Nat) then
        (1-p.smoothing)*x (target-p.classStart).toNat else 0

theorem split_loss (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (target : Int)
    (splitFlag : p.split=Bool.true) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    forwardLoss algebra p memory row block target (forwardValues algebra p memory row block)=
      ↑(partialLoss p x target block) := by
  by_cases ignored : target=p.ignored
  · simp [forwardLoss,partialLoss,ignored,fp,algebra]
  · by_cases inside : (block*p.block : Int)≤target-p.classStart ∧
        target-p.classStart<(min p.cols ((block+1)*p.block) : Nat)
    · have nonneg : 0≤target-p.classStart := le_trans (Int.ofNat_nonneg _) inside.1
      have active : (target-p.classStart).toNat<p.cols := by omega
      have gathered : load algebra memory p.logits (logitAddress p row (target-p.classStart).toNat)=
          ↑(x (target-p.classStart).toNat) := by simp [load,input _ active,algebra]
      cases hs : p.hasSmoothing with
      | «false» =>
        have zero := noSmooth hs
        simp only [forwardLoss,partialLoss,ignored,inside,splitFlag,hs,zero,gathered,
          Bool.false_eq_true,↓reduceIte,and_self,fp_coe,nat_coe,sub_coe]
        apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
        ring
      | «true» =>
        have sm := forward_sum p memory row block x input hs
        simp only [forwardLoss,ignored,↓reduceIte,splitFlag,inside,hs,sm,Option.getD_some,gathered,
          partialLoss,and_self,↓reduceIte]
        simp only [fp_coe,nat_coe,add_coe,sub_coe,mul_coe,div_coe,Nat.cast_one]
        apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
        ring
    · cases hs : p.hasSmoothing with
      | «false» =>
        have zero := noSmooth hs
        simp only [forwardLoss,partialLoss,ignored,inside,splitFlag,hs,zero,
          Bool.false_eq_true,↓reduceIte,fp_coe]
        apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
        ring
      | «true» =>
        have sm := forward_sum p memory row block x input hs
        simp only [forwardLoss,ignored,↓reduceIte,splitFlag,inside,hs,sm,Option.getD_some,partialLoss,and_self,↓reduceIte]
        simp only [fp_coe,nat_coe,add_coe,sub_coe,mul_coe,div_coe]
        apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
        ring

/-- Real arithmetic of one backward lane. Saved LSE is an explicit local read;
the host theorem must supply it from the actual preceding forward execution. -/
theorem backward_lane (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (target : Int) (L upstream : ℝ)
    (labelRead : Source.label p memory row=target)
    (lseRead : (memory p.lse (p.lseBase+row)).read .real=↑L)
    (upstreamRead : (memory p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (lane : Fin p.block) (active : column p block lane.val<p.cols) :
    backwardValues algebra p memory row block lane=
      ↑(if target=p.ignored then 0 else upstream*
        ((1+2*p.squareScale*L)*Real.exp (x (column p block lane.val)-L)-
          (if (column p block lane.val : Int)=target-p.classStart then 1-p.smoothing else 0)-
          p.smoothing/p.totalClasses)) := by
  by_cases ignored : target=p.ignored
  · simp [backwardValues,labelRead,ignored,fp,algebra]
  · have logit := loaded p memory row block x input lane
    rw [if_pos active] at logit
    have lseValue : load algebra memory p.lse (p.lseBase+row)=↑L := by simp [load,lseRead,algebra]
    have upstreamValue : load algebra memory p.upstream (p.upstreamBase+row*p.upstreamStride)=↑upstream := by
      simp [load,upstreamRead,algebra]
    cases hs : p.hasSmoothing with
    | «false» =>
      have zero := noSmooth hs
      by_cases hit : (column p block lane.val : Int)=target-p.classStart
      all_goals
        simp only [backwardValues,labelRead,ignored,↓reduceIte,hs,logit,lseValue,upstreamValue,hit,zero,Bool.false_eq_true,↓reduceIte]
        simp only [fp_coe,nat_coe,add_coe,sub_coe,mul_coe,div_coe,exp_coe]
        apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
        ring
    | «true» =>
      by_cases hit : (column p block lane.val : Int)=target-p.classStart
      all_goals
        simp only [backwardValues,labelRead,ignored,↓reduceIte,hs,logit,lseValue,upstreamValue,hit,Bool.true_eq_false,↓reduceIte]
        simp only [fp_coe,nat_coe,add_coe,sub_coe,mul_coe,div_coe,exp_coe]
        apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
        ring

theorem unsplit_loss (p : Parameters) (memory : Memory (WithBot ℝ)) (row : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (target : Int)
    (positive : 0<p.cols) (covered : p.cols≤p.block)
    (splitFlag : p.split=Bool.false) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : target=p.ignored ∨ (0≤target ∧ target<(p.cols : Int))) :
    forwardLoss algebra p memory row 0 target (forwardValues algebra p memory row 0)=
      ↑(Math.loss p.cols x target p.ignored p.smoothing p.squareScale) := by
  by_cases ignored : target=p.ignored
  · simp [forwardLoss,Math.loss,ignored]
  · have targetValid : 0≤target ∧ target<(p.cols : Int) := valid.resolve_left ignored
    have inside : (0*p.block : Int)≤target-p.classStart ∧
        target-p.classStart<(min p.cols ((0+1)*p.block) : Nat) := by
      simp only [startZero,sub_zero,zero_mul,Nat.zero_add,Nat.one_mul,Nat.min_eq_left covered]
      exact targetValid
    have lseEq := forward_lse p memory row 0 x input (by omega) (by simpa using positive)
    rw [Math.block_zero_lse p.cols p.block covered] at lseEq
    have gathered : load algebra memory p.logits (logitAddress p row (target-p.classStart).toNat)=
        ↑(x target.toNat) := by
      simp only [startZero,sub_zero]
      have hc : target.toNat<p.cols := by omega
      change Source.Cell.read .real (memory p.logits (logitAddress p row target.toNat))=_
      exact input _ hc
    have gathered2 : load algebra memory p.logits (logitAddress p row target.toNat)=↑(x target.toNat) := by
      simpa only [startZero,sub_zero] using gathered
    cases hs : p.hasSmoothing with
    | «false» =>
      have zero := noSmooth hs
      simp only [forwardLoss,Math.loss,ignored,inside,splitFlag,hs,zero,gathered,lseEq,
        Bool.false_eq_true,↓reduceIte,and_self,Nat.cast_zero,zero_mul,startZero,sub_zero,targetValid.1,
        Nat.zero_add,Nat.one_mul,Nat.min_eq_left covered,targetValid.2,gathered2]
      simp only [fp_coe,nat_coe,add_coe,sub_coe,mul_coe,div_coe]
      apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
      ring
    | «true» =>
      have sm := forward_sum p memory row 0 x input hs
      rw [Math.block_zero_sum p.cols p.block covered] at sm
      simp only [forwardLoss,Math.loss,ignored,inside,splitFlag,hs,gathered,lseEq,sm,
        Option.getD_some,Bool.false_eq_true,↓reduceIte,and_self,total,Nat.cast_zero,zero_mul,startZero,sub_zero,targetValid.1,
        Nat.zero_add,Nat.one_mul,Nat.min_eq_left covered,targetValid.2,gathered2]
      simp only [fp_coe,nat_coe,add_coe,sub_coe,mul_coe,div_coe,Nat.cast_one]
      apply congrArg (fun y : ℝ => (↑y : WithBot ℝ))
      ring

#axiomsClean forward_lse
#axiomsClean forward_sum
#axiomsClean split_loss
#axiomsClean unsplit_loss
#axiomsClean backward_lane
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.RealSource
