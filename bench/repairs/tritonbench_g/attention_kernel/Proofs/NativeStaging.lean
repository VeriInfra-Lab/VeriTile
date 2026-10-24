import bench.repairs.tritonbench_g.attention_kernel.Proofs.Native
import bench.repairs.tritonbench_g.attention_kernel.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Native
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

inductive Stage where
  | ready (g : Graph)
  | zero | negInf
  | literal (x : ℝ)
  | add (a b : Stage)
  | sub (a b : Stage)
  | mul (a b : Stage)
  | dot (dtype : TypedSource.Storage) (D row col : Nat) (a b : Fin D→ Stage)

/-- Dots remain pending until their observed accumulator seed is available.
This constructs the actual seeded instruction chain; it does not assert that
a separately rounded zero-seeded dot plus an add equals that chain. -/
def lower : Stage→ Graph
  | .ready g=>g
  | .zero=>zero
  | .negInf=>.bits 0xff800000
  | .literal x=>.literal32 x
  | .add .zero .negInf=>.bits 0xff800000
  | .add (.add .zero (.dot dtype D r c a b)) bias=>
      dot16 dtype D r c (fun d=>lower (a d)) (fun d=>lower (b d)) (lower bias)
  | .add seed (.dot dtype D r c a b)=>
      dot16 dtype D r c (fun d=>lower (a d)) (fun d=>lower (b d)) (lower seed)
  | .add (.mul a b) c=>.ternary .fma32 (lower a) (lower b) (lower c)
  | .add a b=>.binary .add32 (lower a) (lower b)
  | .sub a b=>.binary .sub32 (lower a) (lower b)
  | .mul a (.mul b c)=>.binary .mul32 (.binary .mul32 (lower b) (lower c)) (lower a)
  | .mul a b=>.binary .mul32 (lower a) (lower b)
  | .dot dtype D r c a b=>dot16 dtype D r c (fun d=>lower (a d)) (fun d=>lower (b d)) zero

abbrev Expr := Fin 16→ Stage
instance : Inhabited Expr := ⟨fun _=>.zero⟩

def literal (x : ℝ) : Stage := if x=0 then .zero else .literal x

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (row : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i row)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i row)

def reduction (combine : BinaryOp) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>.ready ((plan (TileShape.axisDim shape ax) owner).value combine (fun j=>
    if h : j<TileShape.axisDim shape ax then
      lower (reductionInputs ax keep xs i ⟨j,h⟩ (physicalOwner j)) else zero))

/-- Constant conversion uses the actual source literals. All other casts are
explicit scalar graph nodes. FP32 dot→ FP32 compute is only a carrier identity. -/
def convert (src dst : FloatDType) (a : Stage) : Stage :=
  match a with
  | .zero=>.zero
  | _=>match src,dst with
    | .fp32,.real=>a
    | .real,.fp16=>match a with
      | .literal x=>if x=1.44269504 then .ready (.bits 0x3dc5) else .ready (.unary .narrowHalf (lower a))
      | _=>.ready (.unary .narrowHalf (lower a))
    | .fp16,.real=>.ready (.unary .widenHalf (lower a))
    | .real,.bf16=>match a with
      | .literal x=>if x=1.44269504 then .ready (.bits 0x3fb9) else .ready (.unary .narrowBF16 (lower a))
      | _=>.ready (.unary .narrowBF16 (lower a))
    | .bf16,.real=>.ready (.unary .widenBF16 (lower a))
    | _,_=>a

def binary (dtype : FloatDType) (op : Binary) (a b : Stage) : Stage :=
  match dtype,op with
  | .fp16,.add=>.ready (.binary .addHalf (lower b) (lower a))
  | .fp16,.mul=>.ready (.binary .mulHalf (lower a) (lower b))
  | .bf16,.add=>.ready (.ternary .fmaBF16 (lower b) (.bits 0x3f80) (lower a))
  | .bf16,.mul=>.ready (.ternary .fmaBF16 (lower a) (lower b) (.bits 0x8000))
  | _,.add=>.add a b
  | _,.sub=>.sub a b
  | _,.mul=>.mul a b
  | _,.max=>.ready (.binary .maxNum32 (lower a) (lower b))
  | _,.div=>.ready (.binary .divFull32 (lower a) (lower b))
  | _,_=>.zero

def algebra : Algebra Expr where
  literal:=fun _ _ x _=>literal x
  negInf:=fun _=>.negInf
  binary:=fun _ dtype op a b u=>binary dtype op (a u) (b u)
  unary:=fun _ op a u=>match op with
    | .exp2=>.ready (.unary .exp2ApproxFTZ (lower (a u)))
    | _=>.zero
  cast:=fun _ src dst a u=>convert src dst (a u)
  fromNat:=fun _ n _=>literal n
  fromInt:=fun _ n _=>literal n
  fp32Bits:=fun b _=>.ready (.bits b.bits)
  fp32Load:=id
  reduceMax:=reduction .maxNum32
  reduceSum:=reduction .add32

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape)→{m k n : Nat}→
      Values Expr left.toTileDType (batch++[m,k])→
      Values Expr right.toTileDType (batch++[k,n])→
      Values Expr out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>ofFloat out (fun _=>.dot (if left=.bf16 then .bf16 else .fp16) D i.1.val i.2.1.val
      (fun d=>toFloat left (a (i.1,d,())) (physicalOwner d.val))
      (fun d=>toFloat right (b (d,i.2.1,())) (physicalOwner i.2.1.val)))
  | _::ds,_,_,_,a,b,i=>dotValues none left right out {} ds
      (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dotModel : Source.Dot Expr where
  apply:=fun p l r out options {batch} {_} {_} {_} a b=>dotValues p l r out options batch a b

def reference (p : Source.Parameters) (s : State Expr) (i : TileIndex [p.bm,p.dim]) : Graph :=
  lower (Whole.answer algebra dotModel p s i (physicalOwner i.2.1.val))

theorem source_native (p : Source.Parameters) (s : State Expr) (hbn : 0<p.bn) :
    ∃ t,FP.Structural.exec (FP.ProfiledRewrite.engine (Source.withDot algebra dotModel)) (Source.kernel p) s=some t ∧
      t.mem=(Source.written algebra p s (Source.accumulated algebra dotModel p s)).mem :=
  Source.source_memory algebra dotModel p s hbn

theorem native_local_transport {α β : Type} (f : α→β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (p : Source.Parameters) (s : State Expr) (i : TileIndex [p.bm,p.dim]) :
    f ((reference p s i).eval A)=(reference p s i).eval B := transport f A B h _

#specNonCircular reference avoiding [FP.Structural.exec,Source.bodyExec,Whole.runPrograms]
#axiomsClean source_native
#axiomsClean native_local_transport
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Native
