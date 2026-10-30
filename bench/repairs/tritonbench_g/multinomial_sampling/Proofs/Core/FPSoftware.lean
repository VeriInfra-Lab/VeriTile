import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPStorage
namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
open VeriTile.Triton FP.Structural

/-- Nonempty binary trees have no artificial zero seed. Every input leaf is
retained, including source padding lanes. No association law is assumed. -/
inductive Tree (n : Nat) where
  | leaf : Fin n → Tree n
  | join : Tree n → Tree n → Tree n

def Tree.leaves : Tree n → List (Fin n)
  | .leaf i => [i] | .join a b => a.leaves++b.leaves
def Tree.map (f : Fin n → Fin m) : Tree n → Tree m
  | .leaf i => .leaf (f i) | .join a b => .join (a.map f) (b.map f)
theorem Tree.map_leaves (f : Fin n → Fin m) (t : Tree n) : (t.map f).leaves=t.leaves.map f := by
  induction t <;> simp [Tree.map,Tree.leaves, *]
def Tree.eval (op : Word → Word → Word) (x : Fin n → Word) : Tree n → Word
  | .leaf i => x i | .join a b => op (a.eval op x) (b.eval op x)

def ascending : (n : Nat) → Tree (n+1)
  | 0 => .leaf ⟨0,by decide⟩
  | n+1 => .join ((ascending n).map Fin.castSucc) (.leaf (Fin.last (n+1)))
theorem ascending_leaves (n : Nat) : (ascending n).leaves=List.finRange (n+1) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [ascending,Tree.leaves,Tree.map_leaves,ih]
    exact List.finRange_succ_last.symm

structure Plans where
  scanTree : (n : Nat) → Tree (n+1)
  scanValid : ∀ n, (scanTree n).leaves.Perm (List.finRange (n+1))
  maxTree : (n : Nat) → Tree (n+1)
  maxValid : ∀ n, (maxTree n).leaves.Perm (List.finRange (n+1))

def ascendingPlans : Plans :=
  ⟨ascending,fun n => by rw [ascending_leaves],ascending,fun n => by rw [ascending_leaves]⟩

def maxnum (x y : Word) : Word :=
  let a := f32 x
  let b := f32 y
  put32 <| if a.isNaN then (if b.isNaN then FP.Value.qNaN .fp32 else b)
  else if b.isNaN then a
  else if a.isZero && b.isZero then FP.Value.zero .fp32 (a.sign && b.sign)
  else if FP.lt .ieeeValue b a then a else b

def minnum (x y : Word) : Word :=
  let a := f32 x
  let b := f32 y
  put32 <| if a.isNaN then (if b.isNaN then FP.Value.qNaN .fp32 else b)
  else if b.isNaN then a
  else if a.isZero && b.isZero then FP.Value.zero .fp32 (a.sign || b.sign)
  else if FP.lt .ieeeValue a b then a else b

def scanOp : ScanOp → Word → Word → Word
  | .sum,x,y => put32 (FP.add .ieeeValue (f32 x) (f32 y))
  | .prod,x,y => put32 (FP.mul .ieeeValue (f32 x) (f32 y))
  | .max,x,y => maxnum x y
  | .min,x,y => minnum x y

def forwardScan (plans : Plans) (op : ScanOp) {shape : TileShape} (ax : Fin shape.length)
    (values : TileIndex shape → Word) (out : TileIndex shape) : Word :=
  let k := TileShape.axisCoord shape ax out
  (plans.scanTree k.val).eval (scanOp op) (fun i => values
    (TileShape.replaceAxisCoord shape ax out ⟨i.val,by have := i.isLt; have := k.isLt; omega⟩))

def scanValues (plans : Plans) (op : ScanOp) {shape : TileShape} (ax : Fin shape.length) :
    ScanDirection → (TileIndex shape → Word) → TileIndex shape → Word
  | .forward,values => forwardScan plans op ax values
  | .reverse,values => fun i => forwardScan plans op ax
      (fun j => values (TileShape.flipIndex shape ax j)) (TileShape.flipIndex shape ax i)

def maximum (plans : Plans) : {n : Nat} → (Fin n → Word) → Word
  | 0,_ => put32 (FP.Value.infinity .fp32 Bool.true)
  | n+1,values => (plans.maxTree n).eval maxnum values

def reduceMaximum (plans : Plans) {shape : TileShape} (ax : Fin shape.length) :
    (keepDims : Bool) → (TileIndex shape → Word) →
      TileIndex (TileShape.reduceShape shape ax keepDims) → Word
  | Bool.false,values => fun out => maximum plans (fun i => values (TileShape.insertAxisIndex shape ax out i))
  | Bool.true,values => fun out => maximum plans (fun i => values (TileShape.replaceAxisIndex shape ax out i))

/-- Closed four-storage bit profile. Scan/max schedules are explicit checked
configuration, not opaque numerical callbacks or assumed reassociation laws.
Triton3.8 tl.max uses default maxnum; exact native schedule/NaN payload/FTZ and
cast behavior still require binding. No probability-normalization claim. -/
def software (d : Storage) (plans : Plans) : Primitives Word :=
  { zero := fun _ _ => 0
    cast := storageCast d
    bits32 := fun b => b.bits.zeroExtend 64
    add := fun x y => put32 (FP.add .ieeeValue (f32 x) (f32 y))
    sub := fun x y => put32 (FP.sub .ieeeValue (f32 x) (f32 y))
    lt := fun x y => FP.lt .ieeeValue (f32 x) (f32 y)
    le := fun x y => FP.lt .ieeeValue (f32 x) (f32 y) || FP.eq .ieeeValue (f32 x) (f32 y)
    scan := fun _ {_} op ax dir values => scanValues plans op ax dir values
    maximum := fun _ {_} ax keep values => reduceMaximum plans ax keep values }

theorem maxnum_vectors :
    maxnum 0x3f800000 0x7fc00000 = 0x3f800000 ∧
    maxnum 0x7fc00000 0x3f800000 = 0x3f800000 ∧
    maxnum 0x80000000 0 = 0 ∧ maxnum 0x80000000 0x80000000 = 0x80000000 ∧
    maxnum 0x7fa12345 0x7fc43210 = 0x7fc00000 := by decide +kernel

#axiomsClean ascending_leaves
#axiomsClean software
#axiomsClean maxnum_vectors
end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
