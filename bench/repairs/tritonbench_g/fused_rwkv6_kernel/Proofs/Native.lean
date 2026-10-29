import bench.repairs.tritonbench_g.fused_rwkv6_kernel.Proofs.Host
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel.Native
noncomputable section
set_option maxHeartbeats 8000000

/-- Source expression graph; cast nodes remain explicit. -/
inductive Term where
  | zero
  | word (region : Region) (address : Nat) (value : BitVec 32)
  | load32 (x : Term)
  | store32 (x : Term)
  | add (x y : Term)
  | mul (x y : Term)
  | exp (x : Term)
  deriving Repr

/-- Policies inspect only topology and locations, never numerical values. -/
inductive Topology where
  | zero
  | word (region : Region) (address : Nat)
  | load32 (x : Topology)
  | store32 (x : Topology)
  | add (x y : Topology)
  | mul (x y : Topology)
  | exp (x : Topology)
  deriving DecidableEq, Repr

def topology : Term → Topology
  | .zero => .zero
  | .word r a _ => .word r a
  | .load32 x => .load32 (topology x)
  | .store32 x => .store32 (topology x)
  | .add x y => .add (topology x) (topology y)
  | .mul x y => .mul (topology x) (topology y)
  | .exp x => .exp (topology x)

def sourceOps : Ops Term := ⟨.zero,.load32,.store32,.add,.mul,.exp,id⟩
inductive Contraction where
  | separate | left | right
  deriving DecidableEq, Repr

structure Compiler where
  contraction : Topology → Topology → Contraction
  swapAdd : Topology → Topology → Bool
  swapMul : Topology → Topology → Bool
  swapFma : Topology → Topology → Topology → Bool

/-- Finite plans have only literal/move and fixed-arity scalar instructions. -/
inductive Plan where
  | word (bits : BitVec 32)
  | move32 (x : Plan)
  | add (x y : Plan)
  | mul (x y : Plan)
  | fma (x y z : Plan)
  | exp2Approx (x : Plan)
  deriving Repr

structure Instructions where
  addRN : BitVec 32 → BitVec 32 → BitVec 32
  mulRN : BitVec 32 → BitVec 32 → BitVec 32
  fmaRN : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32
  exp2Approx : BitVec 32 → BitVec 32

def eval (a : Instructions) : Plan → BitVec 32
  | .word bits => bits
  | .move32 x => eval a x
  | .add x y => a.addRN (eval a x) (eval a y)
  | .mul x y => a.mulRN (eval a x) (eval a y)
  | .fma x y z => a.fmaRN (eval a x) (eval a y) (eval a z)
  | .exp2Approx x => a.exp2Approx (eval a x)

def addPlan (c : Compiler) (x y : Term) (a b : Plan) : Plan :=
  if c.swapAdd (topology x) (topology y) then .add b a else .add a b

def fmaPlan (c : Compiler) (x y z : Term) (a b d : Plan) : Plan :=
  if c.swapFma (topology x) (topology y) (topology z) then .fma b a d else .fma a b d

/-- Retains default fusion with local left/right contractions. FP32 casts are
identity moves and barriers to contraction across partial output memory.
Pinned exp lowering is mul by 0x3fb8aa3b followed by ex2.approx.f32. -/
def compile (c : Compiler) : Term → Plan
  | .zero => .word 0
  | .word _ _ bits => .word bits
  | .load32 x => .move32 (compile c x)
  | .store32 x => .move32 (compile c x)
  | .mul x y => if c.swapMul (topology x) (topology y)
      then .mul (compile c y) (compile c x) else .mul (compile c x) (compile c y)
  | .exp x => .exp2Approx (.mul (compile c x) (.word 0x3fb8aa3b))
  | .add x y => match c.contraction (topology x) (topology y),x,y with
    | .left,.mul u v,y => fmaPlan c u v y (compile c u) (compile c v) (compile c y)
    | .right,x,.mul u v => fmaPlan c u v x (compile c u) (compile c v) (compile c x)
    | _,x,y => addPlan c x y (compile c x) (compile c y)

def observe (a : Instructions) (compiler : Compiler) (t : Term) := eval a (compile compiler t)
def entry (m : Memory (BitVec 32)) : Memory Term := fun r b => .word r b (m r b)

def nativeForward (a : Instructions) (compiler : Compiler) (c : Shape) (plan : Plans c)
    (scale : BitVec 32) (m : Memory (BitVec 32)) : Memory (BitVec 32) :=
  fun r b => observe a compiler (forward sourceOps c plan (.word .outside 0 scale) (entry m) r b)

/-- Exact returned words from the executed memory graph and independent arrays. -/
theorem returned (a : Instructions) (compiler : Compiler) (c : Shape) (hc : Sizes c)
    (plan : Plans c) (scale : BitVec 32) (m : Memory (BitVec 32))
    (z t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    nativeForward a compiler c plan scale m .result (resultAddress c z t f)=
    observe a compiler (scalarOutput sourceOps c plan (inputs c (entry m))
      (.word .outside 0 scale) z t f) := by
  unfold nativeForward
  rw [returned_value sourceOps c hc plan _ _ z t f hz ht hf]

theorem terminal (a : Instructions) (compiler : Compiler) (c : Shape) (hc : Sizes c)
    (plan : Plans c) (scale : BitVec 32) (m : Memory (BitVec 32)) (hs : c.final=true)
    (z e f : Nat) (hz : z<c.Z) (he : e<c.K) (hf : f<c.V) :
    nativeForward a compiler c plan scale m .ht (stateAddress c z e f)=
    observe a compiler (.store32 (scalarState sourceOps c (inputs c (entry m)) z e f c.T)) := by
  unfold nativeForward
  rw [terminal_value sourceOps c hc plan _ _ hs z e f hz he hf]
  rfl

theorem frame (a : Instructions) (compiler : Compiler) (c : Shape) (hc : Sizes c)
    (plan : Plans c) (scale : BitVec 32) (m : Memory (BitVec 32)) (r : Region) (b : Nat)
    (h : ¬WholeOutput c r b) : nativeForward a compiler c plan scale m r b=m r b := by
  unfold nativeForward
  rw [complete_frame sourceOps c hc plan _ _ r b h]
  rfl

/-- Only fixed-arity local instruction conformance remains numerical. -/
structure InstructionAgreement (hardware reference : Instructions) : Prop where
  addRN : ∀ x y, hardware.addRN x y=reference.addRN x y
  mulRN : ∀ x y, hardware.mulRN x y=reference.mulRN x y
  fmaRN : ∀ x y z, hardware.fmaRN x y z=reference.fmaRN x y z
  exp2Approx : ∀ x, hardware.exp2Approx x=reference.exp2Approx x

theorem eval_agreement (a b : Instructions) (h : InstructionAgreement a b) (p : Plan) :
    eval a p=eval b p := by
  induction p with
  | word w => rfl
  | move32 p ih => exact ih
  | add x y hx hy => simp only [eval,h.addRN,hx,hy]
  | mul x y hx hy => simp only [eval,h.mulRN,hx,hy]
  | fma x y z hx hy hz => simp only [eval,h.fmaRN,hx,hy,hz]
  | exp2Approx x hx => simp only [eval,h.exp2Approx,hx]

theorem fpequiv (hardware reference : Instructions) (h : InstructionAgreement hardware reference)
    (compiler : Compiler) (c : Shape) (plan : Plans c) (scale : BitVec 32)
    (m : Memory (BitVec 32)) :
    nativeForward hardware compiler c plan scale m=nativeForward reference compiler c plan scale m := by
  funext r b
  exact eval_agreement hardware reference h _

#axiomsClean returned
#axiomsClean terminal
#axiomsClean frame
#axiomsClean fpequiv
#specNonCircular scalarOutput avoiding [runLoop,runProgram,runGrid,forward,nativeForward]
end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel.Native
