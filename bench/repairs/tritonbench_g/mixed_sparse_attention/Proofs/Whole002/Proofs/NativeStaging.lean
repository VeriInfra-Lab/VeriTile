import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Native
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Output
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
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
  | .add seed (.dot dtype D r c a b)=>
      dot16 dtype D r c (fun d=>lower (a d)) (fun d=>lower (b d)) (lower seed)
  | .add (.mul a b) c=>.ternary .fma32 (lower a) (lower b) (lower c)
  | .add a b=>.binary .add32 (lower a) (lower b)
  | .sub a b=>.binary .sub32 (lower a) (lower b)
  | .mul a (.mul b c)=>.binary .mul32 (.binary .mul32 (lower b) (lower c)) (lower a)
  | .mul a b=>.binary .mul32 (lower a) (lower b)
  | .dot dtype D r c a b=>dot16 dtype D r c (fun d=>lower (a d)) (fun d=>lower (b d)) zero

abbrev Expr := Fin 4→ Stage
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
    | .real,.fp16=>.ready (.unary .narrowHalf (lower a))
    | .fp16,.real=>.ready (.unary .widenHalf (lower a))
    | .real,.bf16=>.ready (.unary .narrowBF16 (lower a))
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

/-- Runtime ABI scalar bits are independent of the ideal Real scale. -/
structure Runtime where
  scaleBits : BitVec 32

/-- The native algebra accepts every runtime FP32 word. This binding and the
public kernel admission require a finite scale decoded for the ideal Real target. -/
def ScaleBinding (runtime : Runtime) (c : Source.Config) : Prop :=
  (Float32Bits.mk runtime.scaleBits).decodeReal=some c.scale

/-- The scale coefficient is a unique literal right operand in this source.
Input storage words enter as ready bit graphs and cannot imitate this marker. -/
def scaleRhs : Stage → Bool
  | .literal x => decide (x=1.44269504)
  | _ => Bool.false

def binaryFor (runtime : Runtime) (dtype : FloatDType) (op : Binary) (a b : Stage) : Stage :=
  if dtype=.real ∧ op=.mul ∧ scaleRhs b=Bool.true then
    .mul (.ready (.bits runtime.scaleBits)) b
  else binary dtype op a b

def algebra (runtime : Runtime) : Algebra Expr where
  literal:=fun _ _ x _=>literal x
  negInf:=fun _=>.negInf
  binary:=fun _ dtype op a b u=>binaryFor runtime dtype op (a u) (b u)
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

/-- Comparison is the ordinary scalar ISA predicate after evaluating the
actual operand graphs. It is not a whole-kernel or whole-output oracle. -/
structure CompareISA (α : Type) extends ISA α where
  equal : α→α→Bool

def equal {α : Type} (I : CompareISA α) (a b : Expr) : Bool :=
  I.equal ((lower (a ⟨0,by decide⟩)).eval I.toISA) ((lower (b ⟨0,by decide⟩)).eval I.toISA)

def dotValues (dtype : TypedSource.Storage) : (batch : TileShape)→{m k n : Nat}→
    (TileIndex (batch++[m,k])→Expr)→(TileIndex (batch++[k,n])→Expr)→(TileIndex (batch++[m,n])→Expr)
  | [],_,D,_,a,b,i=>fun _=>.dot dtype D i.1.val i.2.1.val
      (fun d=>a (i.1,d,()) (physicalOwner d.val))
      (fun d=>b (d,i.2.1,()) (physicalOwner i.2.1.val))
  | _::ds,_,_,_,a,b,i=>dotValues dtype ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2

def dotModel {α : Type} (dtype : TypedSource.Storage) (I : CompareISA α) : Source.Dot Expr where
  equal:=equal I
  apply:=fun _ _ {batch} {_} {_} {_} a b=>dotValues dtype batch a b

variable (runtime : Runtime)

def reference {α : Type} (I : CompareISA α) (c : Source.Config) (s : State Expr)
    (i : TileIndex [c.BM,c.BD]) : Graph :=
  lower (Source.outputValue (algebra runtime) (dotModel c.storage I) c
    (Source.finalValues (algebra runtime) (dotModel c.storage I) c s) i (physicalOwner i.2.1.val))

theorem source_native_graph {α : Type} (I : CompareISA α) (c : Source.Config) (s : State Expr) (hbn : 0<c.BN) :
    ∃ t,FP.Structural.exec (Source.backend (algebra runtime) (dotModel c.storage I)) (Source.kernel c) s=some t ∧
      t.mem=(Source.outputMemory (algebra runtime) (dotModel c.storage I) c s
        (Source.finalValues (algebra runtime) (dotModel c.storage I) c s)).mem :=
  Source.source_memory (algebra runtime) (dotModel c.storage I) c s hbn

structure ComparisonTransport {α β : Type} (f : α→β) (A : CompareISA α) (B : CompareISA β) : Prop where
  primitive : LocalTransport f A.toISA B.toISA
  equality : ∀ a b,A.equal a b=B.equal (f a) (f b)

theorem equality_transport {α β : Type} (f : α→β) (A : CompareISA α) (B : CompareISA β)
    (h : ComparisonTransport f A B) (a b : Expr) : equal A a b=equal B a b := by
  simp only [equal,h.equality,transport f A.toISA B.toISA h.primitive]

theorem dotModel_transport {α β : Type} (f : α→β) (A : CompareISA α) (B : CompareISA β)
    (h : ComparisonTransport f A B) (dtype : TypedSource.Storage) : dotModel dtype A=dotModel dtype B := by
  have he : equal A=equal B := funext (fun a=>funext (fun b=>equality_transport f A B h a b))
  simp only [dotModel,he]

/-- Both comparisons and the returned graph transport from the same local
scalar/MMA law family. No output agreement is assumed. -/
theorem native_local_transport {α β : Type} (f : α→β) (A : CompareISA α) (B : CompareISA β)
    (h : ComparisonTransport f A B) (c : Source.Config) (s : State Expr) (i : TileIndex [c.BM,c.BD]) :
    f ((reference runtime A c s i).eval A.toISA)=(reference runtime B c s i).eval B.toISA := by
  rw [transport f A.toISA B.toISA h.primitive]
  have he : reference runtime A c s i=reference runtime B c s i := by
    simp only [reference,dotModel_transport f A B h]
  rw [he]

#axiomsClean source_native_graph
#axiomsClean equality_transport
#axiomsClean native_local_transport
#specNonCircular reference avoiding [FP.Structural.exec,Source.whole_typed]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
