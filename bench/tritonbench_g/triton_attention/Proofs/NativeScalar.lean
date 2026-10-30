import bench.tritonbench_g.triton_attention.Proofs.NativeMemory
import bench.tritonbench_g.triton_attention.Contract
namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
open VeriTile Triton FP.Structural Source
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

/-- The specialization bit comes from the raw Python scalar class, never from
its FP32 value. Runtime float 1.0 is deliberately Bool.false. -/
def constexprScale (s : Checked.ScalarABI) : Bool :=
  Checked.scalarClass s.raw == .constexprOne

/-- Source-expression lowering with the observed constexpr-one optimization.
Literal provenance is retained in Stage.one; a loaded word equal to one is a
Stage.ready node and cannot trigger this rule. -/
def lowerScalar (one : Bool) (kind : Kind) : Stage → Graph
  | .ready g => g
  | .zero => .word 0
  | .one => .word 0x3f800000
  | .negInf => .word 0xff800000
  | .literal x => .literal x
  | .add .zero .negInf => .word 0xff800000
  | .add .zero (.dot w k a b) => dot w k a b (kind != .forward) (.word 0)
  | .add a (.dot w k q v) => dot w k q v Bool.true (lowerScalar one kind a)
  | .add a (.mul b c) =>
      if kind = .forward then .fma (lowerScalar one kind b) (lowerScalar one kind c) (lowerScalar one kind a)
      else .add (lowerScalar one kind a) (lowerScalar one kind (.mul b c))
  | .add a b => .add (lowerScalar one kind a) (lowerScalar one kind b)
  | .sub .zero a => .neg (lowerScalar one kind a)
  | .sub (.mul a .one) c =>
      if one then .sub (lowerScalar one kind a) (lowerScalar one kind c)
      else if kind = .backward then .fma (.word 0x3f800000) (lowerScalar one kind a) (.neg (lowerScalar one kind c))
      else .sub (.mul (.word 0x3f800000) (lowerScalar one kind a)) (lowerScalar one kind c)
  | .sub (.mul a b) c =>
      if kind = .backward then .fma (lowerScalar one kind b) (lowerScalar one kind a) (.neg (lowerScalar one kind c))
      else .sub (lowerScalar one kind (.mul a b)) (lowerScalar one kind c)
  | .sub a b => .sub (lowerScalar one kind a) (lowerScalar one kind b)
  | .mul a (.literal x) => .mul (.literal x) (lowerScalar one kind a)
  | .mul a .one => if one then lowerScalar one kind a else .mul (.word 0x3f800000) (lowerScalar one kind a)
  | .mul a b => .mul (lowerScalar one kind a) (lowerScalar one kind b)
  | .dot w k a b => dot w k a b Bool.false (.word 0)

def forwardReductionScalar (one : Bool) (combine : Graph → Graph → Graph) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>.ready ((Reduction.plan ⟨owner%4,Nat.mod_lt _ (by decide)⟩).eval combine
    (fun j=>if h : j<TileShape.axisDim shape ax then
      lowerScalar one .forward (reductionInputs ax keep xs i ⟨j,h⟩ (Reduction.owner j).val) else .word 0))

def algebraScalar (one : Bool) (kind : Kind) : Algebra Expr :=
  { algebra kind with
    binary := fun _ _ op a b u=>match op with
      | .add=>.add (a u) (b u)
      | .sub=>.sub (a u) (b u)
      | .mul=>.mul (a u) (b u)
      | .max=>.ready (.max (lowerScalar one kind (a u)) (lowerScalar one kind (b u)))
      | .div=>.ready (.div (lowerScalar one kind (a u)) (lowerScalar one kind (b u)))
      | _=>.zero
    unary := fun _ op a u=>match op with
      | .exp=>.ready (.exp2 (.mul (lowerScalar one kind (a u)) (.word 0x3fb8aa3b)))
      | _=>.zero
    cast := fun _ src dst a u=>match src,dst with
      | .real,.fp16 | .fp32,.fp16=>.ready (.narrow (lowerScalar one kind (a u)))
      | .fp16,.real | .fp16,.fp32=>.ready (.widen (lowerScalar one kind (a u)))
      | _,_=>a u
    reduceMax := forwardReductionScalar one Graph.max
    reduceSum := if kind=.preprocess then preprocessReduction else forwardReductionScalar one Graph.add }

def dotValuesScalar (one : Bool) (kind : Kind) (_ : Option ComputeDType) (left right out : FloatDType)
    (_ : DotOptions) : (batch : TileShape) → {m k n : Nat} →
    Values Expr left.toTileDType (batch++[m,k]) → Values Expr right.toTileDType (batch++[k,n]) →
    Values Expr out.toTileDType (batch++[m,n])
  | [],_,D,N,a,b,i=>ofFloat out (fun _=>.dot (TritonAttention.Whole001.Native.width N) D
      (fun d=>if h : d<D then lowerScalar one kind (toFloat left (a (i.1,⟨d,h⟩,())) (Reduction.owner d).val) else .word 0)
      (fun d=>if h : d<D then lowerScalar one kind (toFloat right (b (⟨d,h⟩,i.2.1,())) (Reduction.owner i.2.1.val).val) else .word 0))
  | _::ds,_,_,_,a,b,i=>dotValuesScalar one kind none left right out {} ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def modelScalar (one : Bool) (kind : Kind) : Primitives Expr where
  literal := (algebraScalar one kind).literal
  negInf := (algebraScalar one kind).negInf
  binary := (algebraScalar one kind).binary
  unary := (algebraScalar one kind).unary
  cast := (algebraScalar one kind).cast
  max := (algebraScalar one kind).reduceMax
  sum := (algebraScalar one kind).reduceSum
  dot := fun p l r out opts {batch} {_} {_} {_} a b=>dotValuesScalar one kind p l r out opts batch a b

theorem raw_integer_one (s : Checked.ScalarABI) (h : s.raw=.integer 1) : constexprScale s=Bool.true := by
  simp [constexprScale,Checked.scalarClass,h]

theorem raw_float_runtime (s : Checked.ScalarABI) (bits : BitVec 64) (h : s.raw=.float64 bits) : constexprScale s=Bool.false := by
  simp [constexprScale,Checked.scalarClass,h]

theorem constexpr_backward_sub (a m : Stage) :
    lowerScalar Bool.true .backward (.sub (.mul a .one) m)=.sub (lowerScalar Bool.true .backward a) (lowerScalar Bool.true .backward m) := rfl

theorem runtime_backward_fma (a m : Stage) :
    lowerScalar Bool.false .backward (.sub (.mul a .one) m)=
      .fma (.word 0x3f800000) (lowerScalar Bool.false .backward a) (.neg (lowerScalar Bool.false .backward m)) := rfl

#axiomsClean raw_integer_one
#axiomsClean raw_float_runtime
#axiomsClean constexpr_backward_sub
#axiomsClean runtime_backward_fma
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.Native
