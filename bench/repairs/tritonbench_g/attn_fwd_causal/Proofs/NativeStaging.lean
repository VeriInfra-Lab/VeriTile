import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Native
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Host
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Native
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

inductive Stage where
  | ready (g : Graph)
  | zero | one | negInf
  | literal (x : ℝ)
  | add (a b : Stage)
  | sub (a b : Stage)
  | mul (a b : Stage)

/-- Materialization follows the observed PTX: two ordered score multiplies,
standalone score/alpha subtraction, and default fusion of l/acc updates. -/
def lower : Stage → Graph
  | .ready g=>g
  | .zero=>zero
  | .one=>.bits 0x3f800000
  | .negInf=>.bits 0xff800000
  | .literal x=>.literal32 x
  | .add .zero .one=>.bits 0x3f800000
  | .add .zero .negInf=>.bits 0xff800000
  | .add (.mul a b) c=>.fma32 (lower a) (lower b) (lower c)
  | .add a b=>.add32 (lower a) (lower b)
  | .sub a b=>.sub32 (lower a) (lower b)
  | .mul (.mul a b) c=>.mul32 (lower c) (.mul32 (lower a) (lower b))
  | .mul a b=>.mul32 (lower a) (lower b)

/-- Each source value retains the four physical lane replicas. Reductions
consume the replica physically owning each key. The returned element later
selects the replica owning its output feature. -/
abbrev Expr := Fin 4 → Stage
instance : Inhabited Expr := ⟨fun _=>.zero⟩

def literal (x : ℝ) : Stage := if x=0 then .zero else if x=1 then .one else .literal x

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (row : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i row)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i row)

def reduction (combine : Graph → Graph → Graph) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>.ready ((plan owner).value combine (fun j=>
    if h : j.val<TileShape.axisDim shape ax then
      lower (reductionInputs ax keep xs i ⟨j.val,h⟩ (physicalOwner j.val)) else zero))

def algebra : Algebra Expr where
  literal:=fun _ _ x _=>literal x
  negInf:=fun _=>.negInf
  binary:=fun _ _ op a b u=>match op with
    | .add=>.add (a u) (b u)
    | .sub=>.sub (a u) (b u)
    | .mul=>.mul (a u) (b u)
    | .max=>.ready (.maxNum32 (lower (a u)) (lower (b u)))
    | .div=>.ready (.divFull32 (lower (a u)) (lower (b u)))
    | _=>.zero
  unary:=fun _ op a u=>match op with
    | .exp2=>.ready (.exp2ApproxFTZ (lower (a u)))
    | _=>.zero
  cast:=fun _ src dst a u=>match src,dst with
    | .real,.fp16=>.ready (.narrow16 (lower (a u)))
    | .fp16,.real=>.ready (.widen16 (lower (a u)))
    | _,_=>.ready (lower (a u))
  fromNat:=fun _ n _=>literal n
  fromInt:=fun _ n _=>literal n
  fp32Bits:=fun b _=>.ready (.bits b.bits)
  fp32Load:=fun a u=>.ready (lower (a u))
  reduceMax:=reduction Graph.maxNum32
  reduceSum:=reduction Graph.add32

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Expr left.toTileDType (batch++[m,k]) →
      Values Expr right.toTileDType (batch++[k,n]) →
      Values Expr out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>ofFloat out (fun _=>.ready (
      if left=.fp16 ∧ right=.fp16 ∧ out=.fp16 then
        dotHalf D i.1.val i.2.1.val
          (fun d=>lower (toFloat left (a (i.1,d,())) (physicalOwner d.val)))
          (fun d=>lower (toFloat right (b (d,i.2.1,())) (physicalOwner i.2.1.val)))
      else dotTF32 D i.1.val i.2.1.val
          (fun d=>lower (toFloat left (a (i.1,d,())) ⟨0,by decide⟩))
          (fun d=>lower (toFloat right (b (d,i.2.1,())) ⟨0,by decide⟩))))
  | _::ds,_,_,_,a,b,i=>dotValues none left right out {} ds
      (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dotModel : Source.Dot Expr where
  apply:=fun p l r out options {batch} {_} {_} {_} a b=>dotValues p l r out options batch a b

/-- The finite native graph is determined by the independently stated source
recurrence with concrete ISA chains, casts and four fixed reduction plans. -/
def reference (p : Source.Parameters) (s : State Expr) (i : TileIndex [128,128]) : Graph :=
  lower (Whole.answer algebra dotModel p s i (physicalOwner i.2.1.val))

theorem source_native (p : Source.Parameters) (s : State Expr) :
    ∃ t,FP.Structural.exec (FP.ProfiledRewrite.engine (Source.withDot algebra dotModel)) (Source.kernel p) s=some t ∧
      t.mem=(Source.written algebra p s (Source.accumulated algebra dotModel p s)).mem :=
  Source.source_memory algebra dotModel p s

theorem native_local_transport {α β : Type} (f : α→β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (p : Source.Parameters) (s : State Expr) (i : TileIndex [128,128]) :
    f ((reference p s i).eval A)=(reference p s i).eval B := transport f A B h _

#specNonCircular reference avoiding [FP.Structural.exec,Source.bodyExec,Whole.runPrograms]
#axiomsClean source_native
#axiomsClean native_local_transport
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Native
