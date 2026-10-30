import bench.tritonbench_g.attention_kernel_aligned.Proofs.Native
import bench.tritonbench_g.attention_kernel_aligned.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Native
open VeriTile Triton FP.Structural Full FPValue
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

inductive Stage where
  | ready (g : Graph)
  | zero | one | negInf | negInf16
  | literal (x : ℝ)
  | add (a b : Stage)
  | halfAdd (a b : Stage)
  | sub (a b : Stage)
  | mul (a b : Stage)
  | dot16 (D row col : Nat) (a b : Fin D → Graph)
  | dot32 (D row col : Nat) (a b : Fin D → Graph)

/-- The source AST retains dots until the observed compiler fusion point.
QK begins with the half bias sum; PV begins with acc*alpha. No equation between
an unseeded dot plus add and a seeded MMA is assumed. -/
def lower : Stage → Graph
  | .ready g=>g
  | .zero=>zero
  | .one=>.bits 0x3f800000
  | .negInf=>.bits 0xff800000
  | .negInf16=>.bits 0xfc00
  | .literal x=>.literal32 x
  | .halfAdd (.halfAdd .zero (.dot16 D r c a b)) bias=>dotHalf16 D r c a b (lower bias)
  | .halfAdd a b=>.add16 (lower b) (lower a)
  | .add .zero .negInf=>.bits 0xff800000
  | .add (.mul a b) (.dot32 D r c q k)=>dotHalf32 D r c q k (.mul32 (lower a) (lower b))
  | .add (.mul a b) c=>.fma32 (lower a) (lower b) (lower c)
  | .add a b=>.add32 (lower a) (lower b)
  | .sub a b=>.sub32 (lower a) (lower b)
  | .mul a (.mul scale constant)=>.mul32 (.mul32 (lower scale) (lower constant)) (lower a)
  | .mul a b=>.mul32 (lower a) (lower b)
  | .dot16 D r c a b=>dotHalf16 D r c a b zero
  | .dot32 D r c a b=>dotHalf32 D r c a b zero

abbrev Expr := Fin 4 → Stage
instance : Inhabited Expr := ⟨fun _=>.zero⟩
def literal (x : ℝ) : Stage := if x=0 then .zero else if x=1 then .one else .literal x

def reductionInputs {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep))
    (j : Fin (TileShape.axisDim shape ax)) : Expr :=
  match keep with
  | Bool.false=>xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true=>xs (TileShape.replaceAxisIndex shape ax i j)

def reduction (combine : Graph → Graph → Graph) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>.ready ((plan owner).value combine (fun j=>
    if h : j.val<TileShape.axisDim shape ax then
      lower (reductionInputs ax keep xs i ⟨j.val,h⟩ (physicalOwner j.val)) else zero))

def narrow : Stage → Stage
  | .zero=>.zero | .negInf=>.negInf16 | a=>.ready (.narrow16 (lower a))
def widen : Stage → Stage
  | .zero=>.zero | .negInf16=>.negInf | a=>.ready (.widen16 (lower a))

def algebra : Algebra Expr where
  literal:=fun _ _ x _=>literal x
  negInf:=fun _=>.negInf
  binary:=fun _ dtype op a b u=>match op with
    | .add=>if dtype=.fp16 then .halfAdd (a u) (b u) else .add (a u) (b u)
    | .sub=>.sub (a u) (b u)
    | .mul=>.mul (a u) (b u)
    | .max=>.ready (.maxNum32 (lower (a u)) (lower (b u)))
    | .div=>.ready (.divFull32 (lower (a u)) (lower (b u)))
    | _=>.zero
  unary:=fun _ op a u=>match op with
    | .exp2=>.ready (.exp2ApproxFTZ (lower (a u)))
    | _=>.zero
  cast:=fun _ src dst a u=>match src,dst with
    | .real,.fp16=>narrow (a u)
    | .fp16,.real=>widen (a u)
    | _,_=>a u
  fromNat:=fun _ n _=>literal n
  fromInt:=fun _ n _=>literal n
  fp32Bits:=fun b _=>.ready (.bits b.bits)
  fp32Load:=fun a=>a
  reduceMax:=reduction Graph.maxNum32
  reduceSum:=reduction Graph.add32

def dotValues (_ : Option ComputeDType) (left right out : FloatDType) (_ : DotOptions) :
    (batch : TileShape) → {m k n : Nat} →
      Values Expr left.toTileDType (batch++[m,k]) →
      Values Expr right.toTileDType (batch++[k,n]) →
      Values Expr out.toTileDType (batch++[m,n])
  | [],_,D,_,a,b,i=>ofFloat out (fun _=>
      if out=.fp16 then .dot16 D i.1.val i.2.1.val
        (fun d=>lower (toFloat left (a (i.1,d,())) (physicalOwner d.val)))
        (fun d=>lower (toFloat right (b (d,i.2.1,())) (physicalOwner i.2.1.val)))
      else .dot32 D i.1.val i.2.1.val
        (fun d=>lower (toFloat left (a (i.1,d,())) (physicalOwner d.val)))
        (fun d=>lower (toFloat right (b (d,i.2.1,())) (physicalOwner i.2.1.val))))
  | _::ds,_,_,_,a,b,i=>dotValues none left right out {} ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def model : Primitives Expr where
  literal:=algebra.literal
  negInf:=algebra.negInf
  binary:=algebra.binary
  unary:=algebra.unary
  cast:=algebra.cast
  max:=algebra.reduceMax
  sum:=algebra.reduceSum
  dot:=fun p l r out opts {batch} {_} {_} {_} a b=>dotValues p l r out opts batch a b

def reductionFor (p : Parameters) (combine : Graph → Graph → Graph) (_ : Option ComputeDType)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Expr .real shape)
    (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>.ready (familyValue combine (fun j=>
    if h : j<TileShape.axisDim shape ax then
      lower (reductionInputs ax keep xs i ⟨j,h⟩ (physicalOwner j)) else zero)
      (VeriTile.Bench.Repairs.AttentionKernelAligned.NativePlan.selected p owner))

/-- The only no-precision Real multiply in this source is qk_scale.
Bind that scalar input from its observed launch bits, preserving -0.0.
All other literals (especially +0 seeds) remain unchanged. -/
def binaryFor (p : Parameters) (prec : Option ComputeDType) (dtype : FloatDType)
    (op : Binary) (a b : Expr) : Expr :=
  if prec=none ∧ dtype=.real ∧ op=.mul then
    fun u=>.mul (.ready (.bits p.scaleWord)) (b u)
  else algebra.binary prec dtype op a b

def algebraFor (p : Parameters) : Algebra Expr :=
  { algebra with
    binary := binaryFor p
    reduceMax := reductionFor p Graph.maxNum32
    reduceSum := reductionFor p Graph.add32 }

def modelFor (p : Parameters) : Primitives Expr :=
  { model with
    binary := binaryFor p
    max := reductionFor p Graph.maxNum32
    sum := reductionFor p Graph.add32 }

def reference (p : Parameters) (s : State Expr) (i : Source.QLane p) : Graph :=
  lower (Whole.answer (modelFor p) p s i (physicalOwner i.2.1.val))

theorem source_native_selected (p : Parameters) (s : State Expr) (hBN : 0<p.blockN) :
    ∃ t,FP.Structural.exec (backend (algebraFor p) (modelFor p)) (Source.source p) s=some t ∧
      t.mem=(Source.written (modelFor p) p s).mem :=
  Whole.source_memory (algebraFor p) (modelFor p) p s hBN

theorem source_native (p : Parameters) (s : State Expr) (hBN : 0<p.blockN) :
    ∃ t,FP.Structural.exec (backend algebra model) (Source.source p) s=some t ∧
      t.mem=(Source.written model p s).mem := Whole.source_memory algebra model p s hBN

theorem native_local_transport {α β : Type} (f : α→β) (A : ISA α) (B : ISA β)
    (h : LocalTransport f A B) (p : Parameters) (s : State Expr) (i : Source.QLane p) :
    f ((reference p s i).eval A)=(reference p s i).eval B := transport f A B h _

#specNonCircular reference avoiding [FP.Structural.exec,Whole.runPrograms]
#axiomsClean source_native
#axiomsClean source_native_selected
#axiomsClean native_local_transport
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Native
