import bench.tritonbench_g.flash_attn.Proofs.Grid
import bench.tritonbench_g.flash_attn.Proofs.Kernels

namespace VeriTile.Bench.Repairs.FlashAttn.KernelNative
open VeriTile Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 5000000

inductive MMAWidth where
  | n16 | n32 | n64 | n128
  deriving DecidableEq

def MMAWidth.columns : MMAWidth→Nat
  | .n16=>16 | .n32=>32 | .n64=>64 | .n128=>128

theorem MMAWidth.positive (w : MMAWidth) : 0<w.columns := by cases w <;> decide

def width (n : Nat) : MMAWidth :=
  if n=16 then .n16 else if n=32 then .n32 else if n=64 then .n64 else .n128

/-- Numerical leaves and fixed instruction structure. Matrix nodes retain
source coordinates, all K8 inputs, and a separate live/clear control bit. -/
inductive Node where
  | word (w : BitVec 32)
  | zero | one | negInf
  | literal (x : ℝ)
  | unary (op : Unary) (x : Node)
  | binary (op : Binary) (x y : Node)
  | product (x y : Node)
  | fma (x y z : Node)
  | narrow (x : Node)
  | widen (x : Node)
  | dot (width : MMAWidth) (row column k : Nat) (a b : Fin k→Node) (live : Bool) (seed : Node)

abbrev Expr := Fin 4→Node
instance : Inhabited Node := ⟨.zero⟩
instance : Inhabited Expr := ⟨fun _=>.zero⟩

def literal (x : ℝ) : Node := if x=0 then .zero else .literal x

/-- Dot/add consumption precedes scalar FMA contraction. The source's masked
QK seed and rescaled PV seed remain live even when their bits are zero. -/
def consume : Binary→Node→Node→Node
  | .mul,.one,.literal x=>.literal x
  | .mul,a,b=>.product a b
  | .add,a,.dot w row col k left right Bool.false .zero=>.dot w row col k left right Bool.true a
  | .add,.zero,.negInf=>.negInf
  | .add,.product a b,c=>.fma a b c
  | op,a,b=>.binary op a b

def cast (source target : FloatDType) (x : Node) : Node :=
  if source=.real ∧ target=.fp16 then .narrow x else
  if source=.fp16 ∧ target=.real then .widen x else x

def tree (op : Binary) (input : Fin 64→Node) : Reduction.Tree→Node
  | .leaf i=>input i
  | .merge a b=>.binary op (tree op input a) (tree op input b)
  | .shuffle _ a=>tree op input a

def reductionInputs (shape : TileShape) (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) :
    TileIndex (TileShape.reduceShape shape ax keep)→Fin (TileShape.axisDim shape ax)→Expr :=
  match keep with
  | Bool.false=>fun i j=>xs (TileShape.insertAxisIndex shape ax i j)
  | Bool.true=>fun i j=>xs (TileShape.replaceAxisIndex shape ax i j)

def reduction (op : Binary) {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Expr .real shape) (i : TileIndex (TileShape.reduceShape shape ax keep)) : Expr :=
  fun owner=>tree op (fun j=>
    if h : j.val<TileShape.axisDim shape ax then
      reductionInputs shape ax keep xs i ⟨j.val,h⟩ (Reduction.owner j.val)
    else .zero) (Reduction.plan owner)

def algebra : Algebra Expr where
  literal:=fun _ _ x _=>literal x
  negInf:=fun _=>.negInf
  binary:=fun _ _ op a b owner=>consume op (a owner) (b owner)
  unary:=fun _ op a owner=>.unary op (a owner)
  cast:=fun _ source target a owner=>cast source target (a owner)
  fromNat:=fun _ n _=>.literal n
  fromInt:=fun _ n _=>.literal n
  fp32Bits:=fun w _=>.word w.bits
  fp32Load:=id
  reduceMax:=fun _ {_} ax keep xs=>reduction .max ax keep xs
  reduceSum:=fun _ {_} ax keep xs=>reduction .add ax keep xs

/-- The A operand is consumed from its physical reduction replica. Q's
pre-scaled values have no reduction-replica dependence. -/
def dotAt : (batch : TileShape)→{m k n : Nat}→
    Values Expr .real (batch++[m,k])→Values Expr .real (batch++[k,n])→
    Values Expr .real (batch++[m,n])
  | [],_,k,n,a,b,i,owner=>.dot (width n) i.1.val i.2.1.val k
      (fun j=>a (i.1,j,()) (Reduction.owner j.val))
      (fun j=>b (j,i.2.1,()) owner) Bool.false .zero
  | _::ds,_,_,_,a,b,i,owner=>dotAt ds (fun j=>a (i.1,j)) (fun j=>b (i.1,j)) i.2 owner

def dot : ActualSource.Dot Expr where
  apply:=fun _ left right out _ {batch} {_} {_} {_} a b i=>
    ofFloat out (dotAt batch (fun j=>toFloat left (a j)) (fun j=>toFloat right (b j)) i)

/-- Only single scalar/conversion instructions and one coordinate of a fixed
m64n{16,32,64,128}k8 TF32 WGMMA are ports. No whole-dot/reduction/output port. -/
structure Instructions where
  add32 : BitVec 32→BitVec 32→BitVec 32
  sub32 : BitVec 32→BitVec 32→BitVec 32
  mul32 : BitVec 32→BitVec 32→BitVec 32
  divFull32 : BitVec 32→BitVec 32→BitVec 32
  max32 : BitVec 32→BitVec 32→BitVec 32
  exp2ApproxFTZ : BitVec 32→BitVec 32
  log2Libdevice : BitVec 32→BitVec 32
  fmaRN32 : BitVec 32→BitVec 32→BitVec 32→BitVec 32
  narrowRNE16 : BitVec 32→BitVec 16
  widen16 : BitVec 16→BitVec 32
  wgmmaTF32 : (width : MMAWidth)→Fin 64→Fin width.columns→
    (Fin 8→BitVec 32)→(Fin 8→BitVec 32)→Bool→BitVec 32→BitVec 32

def binary (I : Instructions) : Binary→BitVec 32→BitVec 32→BitVec 32
  | .add=>I.add32 | .sub=>I.sub32 | .mul=>I.mul32 | .div=>I.divFull32
  | .max=>I.max32 | _=>fun _ _=>0

/-- Profile operand order is determined by graph constructors alone. -/
def reverseMul : Node→Node→Bool
  | _,.product (.word _) (.literal _)=>Bool.true
  | _,_=>Bool.false

def eval (I : Instructions) : Node→BitVec 32
  | .word w=>w | .zero=>0 | .one=>0x3f800000 | .negInf=>0xff800000
  | .literal x=>if x=1.44269504 then 0x3fb8aa3b else 0
  | .unary op x=>if op=.exp2 then I.exp2ApproxFTZ (eval I x)
      else if op=.log2 then I.log2Libdevice (eval I x) else 0
  | .binary op x y=>binary I op (eval I x) (eval I y)
  | .product x y=>if reverseMul x y then I.mul32 (eval I y) (eval I x) else I.mul32 (eval I x) (eval I y)
  | .fma x y z=>I.fmaRN32 (eval I x) (eval I y) (eval I z)
  | .narrow x=>(I.narrowRNE16 (eval I x)).zeroExtend 32
  | .widen x=>I.widen16 ((eval I x).setWidth 16)
  | .dot w row col k a b live seed=>
    (List.range ((k+7)/8)).foldl (fun previous chunk=>
      I.wgmmaTF32 w ⟨row%64,Nat.mod_lt _ (by decide)⟩
        ⟨col%w.columns,Nat.mod_lt _ w.positive⟩
        (fun j=>if h : chunk*8+j.val<k then eval I (a ⟨chunk*8+j.val,h⟩) else 0)
        (fun j=>if h : chunk*8+j.val<k then eval I (b ⟨chunk*8+j.val,h⟩) else 0)
        (if chunk=0 then live else Bool.true) previous) (eval I seed)

theorem consumed_dot_live (w : MMAWidth) (row col k : Nat) (a b : Fin k→Node) (seed : Node) :
    consume .add seed (.dot w row col k a b Bool.false .zero)=
      .dot w row col k a b Bool.true seed := rfl

theorem k8_coverage (k d : Nat) (hd : d<k) :
    d/8∈List.range ((k+7)/8) ∧ d%8<8 ∧ d/8*8+d%8=d := by
  refine ⟨List.mem_range.mpr (by omega),Nat.mod_lt _ (by decide),?_⟩
  simpa [Nat.mul_comm] using Nat.div_add_mod d 8

theorem query_operand_order (q scale : BitVec 32) (coefficient : ℝ) :
    reverseMul (.word q) (.product (.word scale) (.literal coefficient))=Bool.true := rfl

theorem live_zero_seed (w : MMAWidth) (row col k : Nat) (a b : Fin k→Node) :
    consume .add (.word 0) (.dot w row col k a b Bool.false .zero)=
      .dot w row col k a b Bool.true (.word 0) := rfl

theorem constexpr_one_fold (coefficient : ℝ) :
    consume .mul .one (.literal coefficient)=.literal coefficient := rfl

theorem constexpr_query_order (q : BitVec 32) (coefficient : ℝ) :
    reverseMul (.word q) (.literal coefficient)=Bool.false := rfl

#axiomsClean constexpr_one_fold
#axiomsClean constexpr_query_order
#axiomsClean query_operand_order
#axiomsClean live_zero_seed
#axiomsClean consumed_dot_live
#axiomsClean k8_coverage
end
end VeriTile.Bench.Repairs.FlashAttn.KernelNative
