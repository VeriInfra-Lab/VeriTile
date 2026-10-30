import bench.tritonbench_g.int8_dequant_matmul.Proofs.Events

namespace VeriTile.Bench.Repairs.Int8DequantMatmul
noncomputable section
set_option maxHeartbeats 5000000

def realProduct (x y : BitVec 8) : ℝ := (x.toInt : ℝ)*(y.toInt : ℝ)
def realOps : Ops ℝ ℝ ℝ where
  izero := 0
  fzero := 0
  hzero := 0
  product8 := realProduct
  iadd := (·+·)
  mma32 := fun _ xs ys old => old+(List.ofFn (fun i => realProduct (xs i) (ys i))).sum
  toFloat := id
  mul := (·*·)
  narrow := id
  hadd := (·+·)

theorem option_sum (xs : List (Option α)) (f : α → ℝ) :
    ((xs.filterMap id).map f).sum=(xs.map (fun x => (x.map f).getD 0)).sum := by
  induction xs with
  | nil => rfl
  | cons x xs ih => cases x <;> simpa only [List.filterMap_cons,id_eq,Option.map_none,Option.map_some,Option.getD_none,Option.getD_some,List.map_cons,List.sum_cons,zero_add,add_right_inj] using ih
theorem Micro.real (op : Micro BK) (xs ys : Fin BK → BitVec 8) (old : ℝ) :
    op.eval realOps xs ys old=old+(op.terms.map (fun k => realProduct (xs k) (ys k))).sum := by
  cases op with
  | scalar k => simp [Micro.eval,Micro.terms,realOps]
  | instruction kind lanes =>
    simp only [Micro.eval,Micro.terms,realOps,option_sum,List.map_ofFn]
    congr 2
    congr 1
    funext i
    cases h : lanes i <;> simp [h,realProduct]
theorem operations_real (ops : List (Micro BK)) (xs ys : Fin BK → BitVec 8) (old : ℝ) :
    ops.foldl (fun v op => op.eval realOps xs ys v) old=
      old+((ops.flatMap Micro.terms).map (fun k => realProduct (xs k) (ys k))).sum := by
  induction ops generalizing old with
  | nil => simp
  | cons op ops ih =>
    rw [List.foldl_cons,ih,Micro.real]
    simp only [List.flatMap_cons,List.map_append,List.sum_append]
    ring
theorem DotPlan.real (p : DotPlan BK) (xs ys : Fin BK → BitVec 8) (old : ℝ) :
    p.eval realOps xs ys old=old+((List.finRange BK).map (fun k => realProduct (xs k) (ys k))).sum := by
  have h := (p.complete.map (fun k => realProduct (xs k) (ys k))).sum_eq
  unfold DotPlan.eval
  split_ifs
  · rw [operations_real,h]
  · change old+_=_
    rw [operations_real,h]
    simp [realOps]

theorem finRange_rsum (f : Nat → ℝ) (n : Nat) :
    ((List.finRange n).map (fun i => f i.val)).sum=rsum f n := by
  induction n with
  | zero => simp [rsum]
  | succ n ih => simpa [List.finRange_succ_last,List.map_append,List.map_map,Function.comp_def,rsum] using congrArg (fun v => v+f n) ih

def idealInput (s : Shape) (b : Buffers) (hasBias : Bool) (m : Memory ℝ ℝ) : RealInput s where
  A := fun r k => ((read8 m (b.a,b.aOffset+r*s.K+k)).toInt : ℝ)
  B := fun k n => ((read8 m (b.b,b.bOffset+k*s.N+n)).toInt : ℝ)
  x := fun r => read32 realOps m (b.x,b.xOffset+r)
  w := fun n => read32 realOps m (b.w,b.wOffset+n)
  bias := if hasBias then some (fun n => read32 realOps m (b.bias,b.biasOffset+n)) else none

theorem operand_real (s : Shape) (c : Config) (b : Buffers) (m : Memory ℝ ℝ)
    (r n t z : Nat) (hasBias : Bool) (l : Fin c.BK) :
    realProduct (operandA s c b m r t z l) (operandB s c b m n t z l)=
      if sourceK c t z l.val<s.K then
        (idealInput s b hasBias m).A r (sourceK c t z l.val)*
        (idealInput s b hasBias m).B (sourceK c t z l.val) n else 0 := by
  by_cases h : sourceK c t z l.val<s.K <;> simp [operandA,operandB,h,realProduct,idealInput]

theorem accumulation_real (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (m : Memory ℝ ℝ) (hasBias : Bool) (r n z t : Nat) :
    accumulation realOps s c b selected m r n z t=
      rsum (fun t => rsum (fun l => if sourceK c t z l<s.K then
        (idealInput s b hasBias m).A r (sourceK c t z l)*
        (idealInput s b hasBias m).B (sourceK c t z l) n else 0) c.BK) t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [accumulation,DotPlan.real,ih]
    simp_rw [operand_real s c b m r n t z hasBias]
    rw [finRange_rsum (fun l => if sourceK c t z l<s.K then (idealInput s b hasBias m).A r (sourceK c t z l)*(idealInput s b hasBias m).B (sourceK c t z l) n else 0) c.BK]
    rfl

theorem partial_real (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (m : Memory ℝ ℝ) (hasBias : Bool) (r n z : Nat) :
    partialValue realOps s c b selected (1/16129) hasBias m r n z=
      realPartial s c (idealInput s b hasBias m) r n z := by
  rw [partialValue,accumulation_real s c b selected m hasBias]
  cases hasBias <;> by_cases hz : z=0 <;>
    simp [realPartial,idealInput,biasAt,realOps,hz,div_eq_mul_inv]

theorem fold_add (xs : List α) (f : α → ℝ) (seed : ℝ) :
    xs.foldl (fun v x => v+f x) seed=seed+(xs.map f).sum := by
  induction xs generalizing seed with
  | nil => simp
  | cons x xs ih => simp only [List.foldl_cons,ih,List.map_cons,List.sum_cons]; ring
theorem real_reference (s : Shape) (c : Config) (b : Buffers) (selected : Selection s c)
    (q : Schedule s c) (m : Memory ℝ ℝ) (hasBias : Bool) (x : MatrixCell s) :
    wordReference realOps s c b selected (1/16129) hasBias q m x=
      target s (idealInput s b hasBias m) x.1 x.2 := by
  let f : Fin c.S → ℝ := fun z => realPartial s c (idealInput s b hasBias m) x.1 x.2 z.val
  have perm := ((q.complete x).map f).sum_eq
  unfold wordReference cellFold
  simp_rw [partial_real]
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
    have zero : z.val=0 := by have h := z.isLt; omega
    have pt := partials_target s c (idealInput s b hasBias m) x.1 x.2
    simpa [single,rsum,f,zero] using pt
  · simp only [if_neg single]
    rw [fold_add,zero_add,perm]
    change ((List.finRange c.S).map (fun z => realPartial s c (idealInput s b hasBias m) x.1 x.2 z.val)).sum=_
    rw [finRange_rsum]
    exact partials_target _ _ _ _ _

theorem ideal_complete (s : Shape) (c : Config) (b : Buffers) (fresh : b.Fresh)
    (selected : Selection s c) (q : Schedule s c) (m : Memory ℝ ℝ) (hasBias : Bool) (x : MatrixCell s) :
    read16 realOps (launch realOps s c b selected (1/16129) hasBias q m) (outputAddress s b x)=
      target s (idealInput s b hasBias m) x.1 x.2 := by
  rw [launch_output realOps s c b fresh,real_reference]

#axiomsClean DotPlan.real
#axiomsClean accumulation_real
#axiomsClean partial_real
#axiomsClean real_reference
#axiomsClean ideal_complete
end
end VeriTile.Bench.Repairs.Int8DequantMatmul
