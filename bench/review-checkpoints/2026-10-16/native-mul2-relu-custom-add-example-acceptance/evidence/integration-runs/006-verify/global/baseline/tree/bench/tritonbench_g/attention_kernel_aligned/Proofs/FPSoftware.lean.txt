import bench.tritonbench_g.attention_kernel_aligned.Proofs.FPReference
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Binary16
import VeriTile.Triton.Float.ScalarOps
namespace VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
abbrev Word := FP.Value .fp32
abbrev Half := MatmulDequantInt4.Binary16.Value

def widenHalf (x : Word) : Word :=
  match MatmulDequantInt4.Binary16.Value.decode ⟨x.bits.truncate 16⟩ with
  | .nan .. => FP.Value.qNaN .fp32
  | .infinity sign => FP.Value.infinity .fp32 sign
  | .finite sign magnitude => FP.round .fp32 .ieeeValue (if sign then -magnitude else magnitude) sign
def narrowHalf (x : Word) : Word :=
  let h := match x.decode with
    | .nan .. => MatmulDequantInt4.Binary16.Value.qNaN
    | .infinity sign => MatmulDequantInt4.Binary16.Value.infinity sign
    | .finite sign magnitude => MatmulDequantInt4.Binary16.round .ieeeValue
        (if sign then -magnitude else magnitude) sign
  ⟨h.bits.zeroExtend 32⟩
def halfAdd (x y : Word) : Word :=
  ⟨(MatmulDequantInt4.Binary16.add .ieeeValue ⟨x.bits.truncate 16⟩ ⟨y.bits.truncate 16⟩).bits.zeroExtend 32⟩
def maxnum (a b : Word) : Word :=
  if a.isNaN then (if b.isNaN then FP.Value.qNaN .fp32 else b)
  else if b.isNaN then a
  else if a.isZero && b.isZero then FP.Value.zero .fp32 (a.sign && b.sign)
  else if FP.lt .ieeeValue b a then a else b
def scalar (d : FloatDType) (op : Binary) (a b : Word) : Word :=
  match d,op with
  | .fp16,.add => halfAdd a b
  | .real,.add | .fp32,.add => FP.add .ieeeValue a b
  | .real,.sub | .fp32,.sub => FP.sub .ieeeValue a b
  | .real,.mul | .fp32,.mul => FP.mul .ieeeValue a b
  | .real,.div | .fp32,.div => FP.div .ieeeValue a b
  | .real,.max | .fp32,.max => maxnum a b
  | _,_ => FP.Value.qNaN .fp32
def convert (src dst : FloatDType) (a : Word) : Word :=
  match src,dst with
  | .fp16,.real | .fp16,.fp32 => widenHalf a
  | .real,.fp16 | .fp32,.fp16 => narrowHalf a
  | .real,.real | .real,.fp32 | .fp32,.real | .fp32,.fp32 | .fp16,.fp16 => a
  | _,_ => FP.Value.qNaN .fp32

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
  maximumTree : (n : Nat) → Tree (n+1)
  maximumValid : ∀ n, (maximumTree n).leaves.Perm (List.finRange (n+1))
  sumTree : (n : Nat) → Tree (n+1)
  sumValid : ∀ n, (sumTree n).leaves.Perm (List.finRange (n+1))
def ascendingPlans : Plans :=
  ⟨ascending,fun n => by rw [ascending_leaves],ascending,fun n => by rw [ascending_leaves]⟩
def reduceVector (plan : (n : Nat) → Tree (n+1)) (op : Word → Word → Word)
    (empty : Word) : {n : Nat} → (Fin n → Word) → Word
  | 0,_ => empty
  | n+1,v => (plan n).eval op v
def reduceValues (plan : (n : Nat) → Tree (n+1)) (op : Word → Word → Word)
    (empty : Word) {shape : TileShape} (ax : Fin shape.length) :
    (keep : Bool) → (TileIndex shape → Word) → TileIndex (TileShape.reduceShape shape ax keep) → Word
  | Bool.false,v => fun i => reduceVector plan op empty (fun j => v (TileShape.insertAxisIndex shape ax i j))
  | Bool.true,v => fun i => reduceVector plan op empty (fun j => v (TileShape.replaceAxisIndex shape ax i j))

/-- The remaining primitive bindings are named explicitly. No dot schedule,
exp2 approximation, or arbitrary-real host conversion is fabricated. -/
structure External where
  hostLiteral : ℝ → Word
  exp2 : Word → Word
  dot : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values Word left.toTileDType (batch++[m,k]) → Values Word right.toTileDType (batch++[k,n]) →
    Values Word out.toTileDType (batch++[m,n])
def log2e : Word := FP.round .fp32 .ieeeValue (144269504/100000000 : Rat)
noncomputable def software (external : External) (plans : Plans) : Primitives Word :=
  { literal := fun _ _ x => if x=0 then FP.Value.zero .fp32 else
      if x=1.44269504 then log2e else external.hostLiteral x
    negInf := FP.Value.infinity .fp32 Bool.true
    binary := fun _ => scalar
    unary := fun _ op x => if op=.exp2 then external.exp2 x else FP.Value.qNaN .fp32
    cast := fun _ => convert
    max := fun _ {_} ax keep v => reduceValues plans.maximumTree maxnum (FP.Value.infinity .fp32 Bool.true) ax keep v
    sum := fun _ {_} ax keep v => reduceValues plans.sumTree (FP.add .ieeeValue) (FP.Value.zero .fp32) ax keep v
    dot := external.dot }

theorem half_is_not_identity :
    narrowHalf ⟨0x3f800001⟩ = ⟨0x3c00⟩ ∧
    widenHalf (narrowHalf ⟨0x3f800001⟩) = ⟨0x3f800000⟩ ∧
    widenHalf (narrowHalf ⟨0x3f800001⟩) ≠ (⟨0x3f800001⟩ : Word) := by decide +kernel
theorem scalar_boundary_vectors :
    narrowHalf ⟨0x80000000⟩ = ⟨0x8000⟩ ∧
    widenHalf ⟨1⟩ = ⟨0x33800000⟩ ∧
    halfAdd ⟨0x3c00⟩ ⟨0x1000⟩ = ⟨0x3c00⟩ ∧
    maxnum ⟨0x7fc00000⟩ ⟨0x3f800000⟩ = ⟨0x3f800000⟩ ∧
    log2e = ⟨0x3fb8aa3b⟩ := by decide +kernel

#axiomsClean ascending_leaves
#axiomsClean half_is_not_identity
#axiomsClean scalar_boundary_vectors
end VeriTile.Bench.TritonBenchG.AttentionKernelAligned.FPValue
