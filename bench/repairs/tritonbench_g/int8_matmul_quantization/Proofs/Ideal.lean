import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.Producer
import bench.repairs.tritonbench_g.int8_dequant_matmul.Proofs.Ideal

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 5000000
open Int8DequantMatmul (rsum rsum_congr rsum_scale rsum_add split_partition finRange_rsum fold_add realProduct)
def realInteger8 (x : ℝ) : Byte := BitVec.ofInt 8 (if x<0 then -Int.floor (-x) else Int.floor x)
def realOps : Ops ℝ ℝ ℝ where
  toOps := Int8DequantMatmul.realOps
  widen := id
  add := (·+·)
  absolute := abs
  maximum := max
  divide := (·/·)
  scale127 := fun x => x/127
  integer8 := realInteger8

/-- Independent mathematical inputs and target contain no implementation
schedule, arithmetic plan, compiler, memory result, or output reference. -/
structure RealInput (s : Shape) where
  A : Nat → Nat → ℝ
  B : Nat → Nat → ℝ
  aScale : Nat → ℝ
  bScale : Nat → ℝ
def target (s : Shape) (input : RealInput s) (r n : Nat) : ℝ :=
  rsum (fun k => input.A r k*input.B k n) s.K*input.aScale r*input.bScale n
def realInput (s : Shape) (b : Buffers) (m : Memory ℝ ℝ) : RealInput s where
  A := fun r k => ((read8 m (b.a,b.aOffset+r*s.K+k)).toInt : ℝ)
  B := fun k n => ((read8 m (b.b,b.bIndex s k n)).toInt : ℝ)
  aScale := fun r => loadScale realOps b.aScalePrecision m (b.aScale,b.aScaleOffset+r)
  bScale := fun n => loadScale realOps b.bScalePrecision m (b.bScale,b.bScaleOffset+n)
def realPartial (s : Shape) (c : Config) (input : RealInput s) (r n z : Nat) : ℝ :=
  rsum (fun t => rsum (fun l => if sourceK c t z l<s.K then
    input.A r (sourceK c t z l)*input.B (sourceK c t z l) n else 0) c.BK) (steps s c)*
      input.aScale r*input.bScale n
theorem operand_real (s : Shape) (c : Config) (b : Buffers) (m : Memory ℝ ℝ)
    (r n t z : Nat) (l : Fin c.BK) :
    realProduct (operandA s c b m r t z l) (operandB s c b m n t z l)=
      if sourceK c t z l.val<s.K then
        (realInput s b m).A r (sourceK c t z l.val)*(realInput s b m).B (sourceK c t z l.val) n else 0 := by
  by_cases h : sourceK c t z l.val<s.K <;> simp [operandA,operandB,h,realProduct,realInput]
theorem accumulation_real (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (m : Memory ℝ ℝ) (r n z t : Nat) :
    accumulation realOps s c b selected m r n z t=
      rsum (fun t => rsum (fun l => if sourceK c t z l<s.K then
        (realInput s b m).A r (sourceK c t z l)*(realInput s b m).B (sourceK c t z l) n else 0) c.BK) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    have sameOps : realOps.toOps=Int8DequantMatmul.realOps := rfl
    rw [accumulation,sameOps,Int8DequantMatmul.DotPlan.real,ih]
    simp_rw [operand_real]
    rw [finRange_rsum (fun l => if sourceK c t z l<s.K then
      (realInput s b m).A r (sourceK c t z l)*(realInput s b m).B (sourceK c t z l) n else 0) c.BK]
    rfl
theorem partial_real (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (m : Memory ℝ ℝ) (r n z : Nat) :
    partialValue realOps s c b selected m r n z=realPartial s c (realInput s b m) r n z := by
  rw [partialValue,accumulation_real]
  rfl
theorem partials_target (s : Shape) (c : Config) (input : RealInput s) (r n : Nat) :
    rsum (fun z => realPartial s c input r n z) c.S=target s input r n := by
  have h := split_partition s c.toConfig (fun k => input.A r k*input.B k n)
  have scale := rsum_scale (input.aScale r*input.bScale n)
    (fun z => rsum (fun t => rsum (fun l => if sourceK c t z l<s.K then
      input.A r (sourceK c t z l)*input.B (sourceK c t z l) n else 0) c.BK) (steps s c)) c.S
  have lhs : rsum (fun z => realPartial s c input r n z) c.S=
      rsum (fun z => (input.aScale r*input.bScale n)*
        rsum (fun t => rsum (fun l => if sourceK c t z l<s.K then
          input.A r (sourceK c t z l)*input.B (sourceK c t z l) n else 0) c.BK) (steps s c)) c.S := by
    apply rsum_congr
    intro z _
    unfold realPartial
    ring
  rw [lhs,scale]
  change (input.aScale r*input.bScale n)*_=_
  simp only [sourceK,steps]
  rw [h]
  unfold target
  ring
def embedReal : Precision → ℝ → Stored ℝ ℝ | .half,x => .inr x | .single,x => .inl x
theorem cellFold_real (c : Config) (precision : Precision) (values : Fin c.S → ℝ)
    (zs : List (Fin c.S)) (seed : ℝ) :
    cellFold realOps c precision values zs (embedReal precision seed)=
      embedReal precision (zs.foldl (fun old z => if c.S=1 then values z else old+values z) seed) := by
  induction zs generalizing seed with
  | nil => rfl
  | cons z zs ih =>
    unfold cellFold at *
    simp only [List.foldl_cons]
    by_cases single : c.S=1
    · simp only [if_pos single]
      cases precision <;> simpa [single,contribution,embedReal,realOps] using ih (values z)
    · simp only [if_neg single]
      cases precision <;> simpa [single,outputAdd,embedReal,realOps,Int8DequantMatmul.realOps] using ih (seed+values z)
theorem real_reference (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (q : Schedule s c) (m : Memory ℝ ℝ) (x : MatrixCell s) :
    wordReference realOps s c b selected q m x=embedReal b.outPrecision (target s (realInput s b m) x.1 x.2) := by
  let f : Fin c.S → ℝ := fun z => realPartial s c (realInput s b m) x.1 x.2 z.val
  have perm := ((q.complete x).map f).sum_eq
  unfold wordReference
  have zero : outputZero realOps b.outPrecision=embedReal b.outPrecision 0 := by cases b.outPrecision <;> rfl
  rw [zero,cellFold_real]
  simp_rw [partial_real]
  apply congrArg (embedReal b.outPrecision)
  change (projection x q.events).foldl (fun old z => if c.S=1 then f z else old+f z) 0=_
  by_cases single : c.S=1
  · have len : (projection x q.events).length=1 := by simpa [single] using (q.complete x).length_eq
    have hs : ∃ z,projection x q.events=[z] := by
      cases h : projection x q.events with
      | nil => simp [h] at len
      | cons z zs => cases zs with
        | nil => exact ⟨z,rfl⟩
        | cons z' zs => simp [h] at len
    obtain ⟨z,hz⟩ := hs
    rw [hz,List.foldl_cons,List.foldl_nil,if_pos single]
    have hz0 : z.val=0 := by have h:=z.isLt; omega
    simpa [single,rsum,f,hz0] using partials_target s c (realInput s b m) x.1 x.2
  · simp only [if_neg single]
    rw [fold_add,zero_add,perm]
    change ((List.finRange c.S).map (fun z => realPartial s c (realInput s b m) x.1 x.2 z.val)).sum=_
    rw [finRange_rsum]
    exact partials_target _ _ _ _ _
theorem ideal_complete (s : Shape) (c : Config) (b : Buffers) (sep : b.Separate)
    (selected : Selection s c) (q : Schedule s c) (m : Memory ℝ ℝ) (x : MatrixCell s) :
    launch realOps s c b selected q m (outputAddress s b x)=
      .inr (embedReal b.outPrecision (target s (realInput s b m) x.1 x.2)) := by
  rw [launch_word realOps s c b sep,real_reference]

#axiomsClean accumulation_real
#axiomsClean partials_target
#axiomsClean real_reference
#axiomsClean ideal_complete
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
