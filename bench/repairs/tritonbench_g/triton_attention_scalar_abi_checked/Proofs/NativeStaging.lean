import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.NativeGraph

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

inductive Kind where | forward | preprocess | backward
  deriving DecidableEq

/-- Local source-expression staging retains the dot until its actual seeded
MMA use and retains scalar products until their compiler contraction site. -/
inductive Stage where
  | ready (g : Graph)
  | zero | one | negInf
  | literal (x : ℝ)
  | add (a b : Stage)
  | sub (a b : Stage)
  | mul (a b : Stage)
  | dot (w : Width) (k : Nat) (a b : Nat → Graph)

def lower (kind : Kind) : Stage → Graph
  | .ready g => g
  | .zero => .word 0
  | .one => .word 0x3f800000
  | .negInf => .word 0xff800000
  | .literal x => .literal x
  | .add .zero .negInf => .word 0xff800000
  | .add .zero (.dot w k a b) => dot w k a b (kind != .forward) (.word 0)
  | .add a (.dot w k q v) => dot w k q v Bool.true (lower kind a)
  | .add a (.mul b c) =>
      if kind = .forward then .fma (lower kind b) (lower kind c) (lower kind a)
      else .add (lower kind a) (lower kind (.mul b c))
  | .add a b => .add (lower kind a) (lower kind b)
  | .sub .zero a => .neg (lower kind a)
  | .sub (.mul a b) c =>
      if kind = .backward then .fma (lower kind b) (lower kind a) (.neg (lower kind c))
      else .sub (lower kind (.mul a b)) (lower kind c)
  | .sub a b => .sub (lower kind a) (lower kind b)
  | .mul a (.literal x) => .mul (.literal x) (lower kind a)
  | .mul a .one => .mul (.word 0x3f800000) (lower kind a)
  | .mul a b => .mul (lower kind a) (lower kind b)
  | .dot w k a b => dot w k a b Bool.false (.word 0)

abbrev Expr := Nat → Stage
instance : Inhabited Expr := ⟨fun _ => .zero⟩
def literal (x : ℝ) : Stage := if x = 0 then .zero else if x = 1 then .one else .literal x

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (j : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false => xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true => xs (TileShape.replaceAxisIndex shape ax i j)

def forwardReduction (combine : Graph → Graph → Graph) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner => .ready ((Reduction.plan ⟨owner%4,Nat.mod_lt _ (by decide)⟩).eval combine
    (fun j => if h : j < TileShape.axisDim shape ax then
      lower .forward (reductionInputs ax keep xs i ⟨j,h⟩ (Reduction.owner j).val) else .word 0))

def productValue : Stage → Graph
  | .mul y g => .mul (lower .preprocess g) (lower .preprocess y)
  | a => lower .preprocess a
def productFma (a : Stage) (seed : Graph) : Graph :=
  match a with
  | .mul y g => .fma (lower .preprocess g) (lower .preprocess y) seed
  | _ => .add (lower .preprocess a) seed

def preLocal (xs : Nat → Stage) (owner : Nat) : Graph :=
  productFma (xs (owner*4+3))
    (productFma (xs (owner*4+2))
      (productFma (xs (owner*4)) (productValue (xs (owner*4+1)))))

def preprocessReduction (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  let dim := TileShape.axisDim shape ax
  let values := fun j => if h : j < dim then reductionInputs ax keep xs i ⟨j,h⟩ 0 else .zero
  fun owner => .ready (TritonAttention.Whole001.Native.shuffleAdd ops
    (TritonAttention.Whole001.Native.preMasks dim) (preLocal values) (owner % (dim/4)))

def algebra (kind : Kind) : Algebra Expr where
  literal := fun _ _ x _ => literal x
  negInf := fun _ => .negInf
  binary := fun _ _ op a b u => match op with
    | .add => .add (a u) (b u)
    | .sub => .sub (a u) (b u)
    | .mul => .mul (a u) (b u)
    | .max => .ready (.max (lower kind (a u)) (lower kind (b u)))
    | .div => .ready (.div (lower kind (a u)) (lower kind (b u)))
    | _ => .zero
  unary := fun _ op a u => match op with
    | .exp => .ready (.exp2 (.mul (lower kind (a u)) (.word 0x3fb8aa3b)))
    | _ => .zero
  cast := fun _ src dst a u => match src,dst with
    | .real,.fp16 | .fp32,.fp16 => .ready (.narrow (lower kind (a u)))
    | .fp16,.real | .fp16,.fp32 => .ready (.widen (lower kind (a u)))
    | _,_ => a u
  fromNat := fun _ n _ => literal n
  fromInt := fun _ n _ => literal n
  fp32Bits := fun b _ => .ready (.word b.bits)
  fp32Load := id
  reduceMax := forwardReduction Graph.max
  reduceSum := if kind = .preprocess then preprocessReduction else forwardReduction Graph.add

def dotValues (kind : Kind) (_ : Option ComputeDType) (left right out : FloatDType)
    (_ : DotOptions) : (batch : TileShape) → {m k n : Nat} →
    Values Expr left.toTileDType (batch++[m,k]) →
    Values Expr right.toTileDType (batch++[k,n]) →
    Values Expr out.toTileDType (batch++[m,n])
  | [],_,D,N,a,b,i => ofFloat out (fun _ => .dot (TritonAttention.Whole001.Native.width N) D
      (fun d => if h : d < D then lower kind
        (toFloat left (a (i.1,⟨d,h⟩,())) (Reduction.owner d).val) else .word 0)
      (fun d => if h : d < D then lower kind
        (toFloat right (b (⟨d,h⟩,i.2.1,())) (Reduction.owner i.2.1.val).val) else .word 0))
  | _::ds,_,_,_,a,b,i => dotValues kind none left right out {} ds
      (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2

def model (kind : Kind) : Primitives Expr where
  literal := (algebra kind).literal
  negInf := (algebra kind).negInf
  binary := (algebra kind).binary
  unary := (algebra kind).unary
  cast := (algebra kind).cast
  max := (algebra kind).reduceMax
  sum := (algebra kind).reduceSum
  dot := fun p l r out opts {batch} {_} {_} {_} a b => dotValues kind p l r out opts batch a b

end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
