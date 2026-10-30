import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteSemantics
import bench.tritonbench_g.int8_dequant_matmul.Proofs.Ideal

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
noncomputable section

def scalarPlan (k : Nat) : DotPlan k where
  operations := (List.finRange k).map Micro.scalar
  complete := by
    have all (xs : List (Fin k)) : (xs.map Micro.scalar).flatMap Micro.terms=xs := by
      induction xs with
      | nil=>rfl
      | cons x xs ih=>simp [Micro.terms,ih]
    simp only [all]
    exact List.Perm.refl _
  fused := Bool.false

def selectedPlan (s : Shape) (c : Config) (selected : Selection s c)
    (k row col split iteration : Nat) : DotPlan k :=
  if h : k=c.BK then h.symm ▸ selected.plan row col split iteration else scalarPlan k

@[simp] theorem selectedPlan_same (s : Shape) (c : Config) (selected : Selection s c)
    (row col split iteration : Nat) :
    selectedPlan s c selected c.BK row col split iteration=selected.plan row col split iteration := by
  simp [selectedPlan]

def nativeInteger (profile : FixedISAProfile) (s : Shape) (c : Config)
    (selected : Selection s c) : IntegerModel (BitVec 32) Word32 Word16 where
  ops := fpOps profile
  encode := BitVec.ofInt 32
  decode := BitVec.toInt
  roundtrip := by intro x;simp
  zero := rfl
  plans := selectedPlan s c selected

def integerRealOps : Ops Int ℝ ℝ where
  izero := 0
  fzero := 0
  hzero := 0
  product8 := fun x y=>x.toInt*y.toInt
  iadd := (·+·)
  mma32 := fun _ xs ys seed=>seed+(List.ofFn (fun i : Fin 32=>(xs i).toInt*(ys i).toInt)).sum
  toFloat := fun x=>(x:ℝ)
  mul := (·*·)
  narrow := id
  hadd := (·+·)

def idealInteger (s : Shape) (c : Config) (selected : Selection s c) : IntegerModel Int ℝ ℝ where
  ops := integerRealOps
  encode := id
  decode := id
  roundtrip := fun _=>rfl
  zero := rfl
  plans := selectedPlan s c selected

def asF (a : Ops I F H) : Sum F H → F
  | .inl x=>x
  | .inr _=>a.fzero
def asH (a : Ops I F H) : Sum F H → H
  | .inl _=>a.hzero
  | .inr x=>x

/-- The source's actual floating operations: int32 conversion, three FP32
multiplications, FP16 narrowing, and FP16 ordinary/atomic addition. All other
fields are unused by the checked quote. -/
def floating (q : IntegerModel I F H) : Algebra (Sum F H) where
  literal := fun _ dtype _=>if dtype=.fp16 then .inr q.ops.hzero else .inl q.ops.fzero
  negInf := .inl q.ops.fzero
  binary := fun _ dtype operation x y=>
    match dtype,operation with
    | .fp16,.add=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))
    | .real,.mul | .fp32,.mul=>.inl (q.ops.mul (asF q.ops x) (asF q.ops y))
    | _,_=>.inl q.ops.fzero
  atomicAdd := fun dtype=>if dtype=.fp16 then some (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) else none
  unary := fun _ _ _=>.inl q.ops.fzero
  cast := fun _ src dst x=>
    match src,dst with
    | .real,.fp16 | .fp32,.fp16=>.inr (q.ops.narrow (asF q.ops x))
    | .fp32,.real | .real,.fp32=>.inl (asF q.ops x)
    | _,_=>x
  fromNat := fun _ x=>.inl (q.ops.toFloat (q.encode (x:Int)))
  fromInt := fun _ x=>.inl (q.ops.toFloat (q.encode x))
  fp32Bits := fun _=>.inl q.ops.fzero
  fp32Load := id
  reduceMax := fun _ {_} _ _ _=>fun _=>.inl q.ops.fzero
  reduceSum := fun _ {_} _ _ _=>fun _=>.inl q.ops.fzero

theorem integer_micro_real (op : Micro k) (xs ys : Fin k → BitVec 8) (seed : Int) :
    ((op.eval integerRealOps xs ys seed : Int) : ℝ)=op.eval realOps xs ys (seed:ℝ) := by
  cases op with
  | scalar l=>simp [Micro.eval,integerRealOps,realOps,realProduct]
  | instruction kind lanes=>
    simp only [Micro.eval,integerRealOps,realOps,realProduct,Int.cast_add,Int.cast_list_sum,
      List.map_ofFn,Function.comp_apply,Int.cast_mul]
    apply congrArg (fun f : Fin 32→ℝ=>(seed:ℝ)+(List.ofFn f).sum)
    funext i
    simp only [Function.comp_apply,Int.cast_mul]

theorem integer_operations_real (operations : List (Micro k))
    (xs ys : Fin k → BitVec 8) (seed : Int) :
    ((operations.foldl (fun old op=>op.eval integerRealOps xs ys old) seed : Int) : ℝ)=
      operations.foldl (fun old op=>op.eval realOps xs ys old) (seed:ℝ) := by
  induction operations generalizing seed with
  | nil=>rfl
  | cons op rest ih=>simp only [List.foldl_cons,ih,integer_micro_real]

theorem integer_plan_real (plan : DotPlan k) (xs ys : Fin k → BitVec 8) (seed : Int) :
    ((plan.eval integerRealOps xs ys seed : Int) : ℝ)=plan.eval realOps xs ys (seed:ℝ) := by
  by_cases h : plan.fused
  · simp only [DotPlan.eval,h,↓reduceIte,integer_operations_real]
  · simp only [DotPlan.eval,h,↓reduceIte]
    change ((seed+plan.operations.foldl (fun old op=>op.eval integerRealOps xs ys old) 0 : Int) : ℝ)=
      (seed:ℝ)+plan.operations.foldl (fun old op=>op.eval realOps xs ys old) 0
    rw [Int.cast_add,integer_operations_real]
    simp only [Int.cast_zero]

#axiomsClean selectedPlan_same
#axiomsClean integer_micro_real
#axiomsClean integer_operations_real
#axiomsClean integer_plan_real
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
