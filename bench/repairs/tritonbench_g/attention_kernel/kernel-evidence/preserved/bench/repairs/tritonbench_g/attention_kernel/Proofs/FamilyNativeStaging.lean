import bench.repairs.tritonbench_g.attention_kernel.Proofs.FamilyNativePlan
import bench.repairs.tritonbench_g.attention_kernel.Proofs.NativeStaging
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole002.FamilyNative
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernel.Whole001
open Native (Graph Stage lower literal convert binary zero)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

structure Layout where
  width : NativePlan.Width
  columns : NativePlan.WarpColumns
  scaleWord : BitVec 32

def layout (s : NativePlan.Selection p abi) (scaleWord : BitVec 32) : Layout := ⟨s.width,s.columns,scaleWord⟩

abbrev Expr := Fin 32→Stage
instance : Inhabited Expr := ⟨fun _=>.zero⟩

def physicalOwner (plan : Layout) (column : Nat) : Fin 32 :=
  NativePlan.physicalOwner plan.width plan.columns column

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (row : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i row)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i row)

def reduction (plan : Layout) (combine : Native.BinaryOp) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>.ready ((NativePlan.tree plan.width plan.columns owner).value combine (fun j=>
    if h : j<TileShape.axisDim shape ax then
      lower (reductionInputs ax keep xs i ⟨j,h⟩ (physicalOwner plan j)) else zero))

/-- The sole product of two source literals with right operand log2(e)
is the dynamic launch scale. Bind its observed FP32 word at that syntactic
site, retaining signed zero. Bias conversion and all +0 seeds are separate
storage-typed expressions and do not use this branch. -/
def binaryWithScale (word : BitVec 32) (dtype : FloatDType)
    (op : Binary) (a b : Stage) : Stage :=
  match dtype,op,a,b with
  | .real,.mul,.zero,.literal c =>
      if c=1.44269504 then .mul (.ready (.bits word)) b else binary dtype op a b
  | .real,.mul,.literal _,.literal c =>
      if c=1.44269504 then .mul (.ready (.bits word)) b else binary dtype op a b
  | _,_,_,_ => binary dtype op a b

theorem scale_word (word : BitVec 32) (scale : ℝ) :
    lower (binaryWithScale word .real .mul (literal scale) (literal 1.44269504))=
      .binary .mul32 (.bits word) (.literal32 1.44269504) := by
  by_cases hz : scale=0 <;> norm_num [literal,hz,binaryWithScale,lower]

/-- The inherited scalar staging is unchanged: actual source casts, bias
storage arithmetic, scale-first Q product, exp2, normalizer FMA, full division,
and seeded fixed-MMA lowering. Only the finite row-reduction layout varies. -/
def algebra (plan : Layout) : Algebra Expr where
  literal:=fun _ _ x _=>literal x
  negInf:=fun _=>.negInf
  binary:=fun _ dtype op a b u=>binaryWithScale plan.scaleWord dtype op (a u) (b u)
  unary:=fun _ op a u=>match op with
    | .exp2=>.ready (.unary .exp2ApproxFTZ (lower (a u)))
    | _=>.zero
  cast:=fun _ src dst a u=>convert src dst (a u)
  fromNat:=fun _ n _=>literal n
  fromInt:=fun _ n _=>literal n
  fp32Bits:=fun b _=>.ready (.bits b.bits)
  fp32Load:=id
  reduceMax:=reduction plan .maxNum32
  reduceSum:=reduction plan .add32

def dotValues (plan : Layout) (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape)→{m k n : Nat}→
      Values Expr left.toTileDType (batch++[m,k])→
      Values Expr right.toTileDType (batch++[k,n])→
      Values Expr out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>ofFloat out (fun _=>.dot (if left=.bf16 then .bf16 else .fp16) D i.1.val i.2.1.val
      (fun d=>toFloat left (a (i.1,d,())) (physicalOwner plan d.val))
      (fun d=>toFloat right (b (d,i.2.1,())) (physicalOwner plan i.2.1.val)))
  | _::ds,_,_,_,a,b,i=>dotValues plan none left right out {} ds
      (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dotModel (plan : Layout) : Source.Dot Expr where
  apply:=fun p l r out options {batch} {_} {_} {_} a b=>dotValues plan p l r out options batch a b

def reference (plan : Layout) (p : Source.Parameters) (s : State Expr) (i : TileIndex [p.bm,p.dim]) : Graph :=
  lower (Whole.answer (algebra plan) (dotModel plan) p s i (physicalOwner plan i.2.1.val))

theorem source_native (plan : Layout) (p : Source.Parameters) (s : State Expr) (hbn : 0<p.bn) :
    ∃ t,FP.Structural.exec (FP.ProfiledRewrite.engine (Source.withDot (algebra plan) (dotModel plan)))
        (Source.kernel p) s=some t ∧
      t.mem=(Source.written (algebra plan) p s (Source.accumulated (algebra plan) (dotModel plan) p s)).mem :=
  Source.source_memory (algebra plan) (dotModel plan) p s hbn

theorem local_transport {α β : Type} (f : α→β) (A : Native.ISA α) (B : Native.ISA β)
    (h : Native.LocalTransport f A B) (plan : Layout) (p : Source.Parameters) (s : State Expr)
    (i : TileIndex [p.bm,p.dim]) :
    f ((reference plan p s i).eval A)=(reference plan p s i).eval B := Native.transport f A B h _

#specNonCircular reference avoiding [FP.Structural.exec,Source.bodyExec,Whole.runPrograms]
#axiomsClean scale_word
#axiomsClean source_native
#axiomsClean local_transport
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole002.FamilyNative
