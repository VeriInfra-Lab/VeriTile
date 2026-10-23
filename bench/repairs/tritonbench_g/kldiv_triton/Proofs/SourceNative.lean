import bench.repairs.tritonbench_g.kldiv_triton.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.SourceNative
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 5000000

abbrev Word := BitVec 32
abbrev Byte := BitVec 8
def zero : Word := BitVec.ofNat 32 0
def minusOne : Word := BitVec.ofNat 32 0xbf800000

/-- Ordinary selected FP32 source primitive meanings, with operands retained
in source order. These computational parameters are not numerical laws. -/
structure Primitives where
  add : Word → Word → Word
  subtract : Word → Word → Word
  multiply : Word → Word → Word
  logarithm : Word → Word
  exponential : Word → Word
  negate : Word → Word
  equal : Word → Word → Bool

/-- The compiler supplies each concrete ordered binary addition tree.
ReductionPlan records coverage; no numerical regrouping law is assumed. -/
abbrev Plans := (width : Nat) → ReductionPlan width

/-- Independent source-ordered point graph, including the raw zero extension. -/
def loss (p : Primitives) (flag : Bool) (prediction target : Word) : Word :=
  if flag then p.multiply (p.exponential target) (p.subtract target prediction)
  else if p.equal target zero then zero
  else p.multiply target (p.subtract (p.logarithm target) prediction)

def chunk (p : Primitives) (plans : Plans) (flag : Bool) (cols width k : Nat)
    (prediction target : Nat → Word) : Word :=
  HostReduction.eval p.add zero (fun lane : Fin width =>
    if k*width+lane.val<cols then loss p flag (prediction (k*width+lane.val)) (target (k*width+lane.val)) else zero)
    (plans width).tree

def derivative (p : Primitives) (flag : Bool) (target : Word) : Word :=
  if flag then p.negate (p.exponential target) else p.multiply target minusOne

def forwardTarget (p : Primitives) (plans : Plans) (flag : Bool) (mode : Math.Mode)
    (cols : Nat) (prediction target : Nat → Word) (c : Nat) : Word :=
  if mode=.none then loss p flag (prediction c) (target c)
  else chunk p plans flag cols (Host.blockSize cols) c prediction target

def literal (x : ℝ) : Word :=
  if x = -1 then minusOne else if x=1 then BitVec.ofNat 32 0x3f800000 else zero

def sumValues (p : Primitives) (plans : Plans) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (values : Values Word .real shape) :
    Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => HostReduction.eval p.add zero
      (fun lane => values (TileShape.insertAxisIndex shape ax out lane))
      (plans (TileShape.axisDim shape ax)).tree
  · exact fun out => HostReduction.eval p.add zero
      (fun lane => values (TileShape.replaceAxisIndex shape ax out lane))
      (plans (TileShape.axisDim shape ax)).tree

def algebra (p : Primitives) (plans : Plans) : Algebra Word where
  literal := fun _ _ => literal
  negInf := BitVec.ofNat 32 0xff800000
  binary := fun _ _ op a b => match op with
    | .add => p.add a b
    | .sub => p.subtract a b
    | .mul => p.multiply a b
    | _ => zero
  unary := fun _ op a => match op with
    | .log => p.logarithm a
    | .exp => p.exponential a
    | .neg _ => p.negate a
    | _ => zero
  compareEq := fun _ _ => some p.equal
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => zero
  fromInt := fun _ _ => zero
  fp32Bits := fun bits => bits.bits
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => zero
  reduceSum := fun _ => sumValues p plans

def model (p : Primitives) (plans : Plans) : Tile.Model Word :=
  ⟨algebra p plans,p.equal,rfl⟩
def adapter (p : Primitives) (plans : Plans) : Storage.Adapter Word Word :=
  ⟨model p plans,id,id⟩

theorem loss_lowering (p : Primitives) (plans : Plans) (flag : Bool) (y t : Word) :
    Tile.loss (model p plans) flag y t=loss p flag y t := by
  cases flag <;> norm_num [Tile.loss,Tile.rawLoss,Tile.zero,model,algebra,literal,loss]

theorem point_lowering (p : Primitives) (plans : Plans) (flag : Bool) (y t : Word) :
    Host.point (adapter p plans) flag y t=loss p flag y t := loss_lowering p plans flag y t

theorem derivative_lowering (p : Primitives) (plans : Plans) (flag : Bool) (t : Word) :
    Host.derivative (adapter p plans) flag t=derivative p flag t := by
  cases flag <;> norm_num [Host.derivative,adapter,Tile.gradient,model,algebra,literal,derivative]

theorem chunk_lowering (p : Primitives) (plans : Plans) (flag : Bool) (cols width k : Nat)
    (prediction target : Nat → Word) :
    Host.chunk (adapter p plans) flag cols width k prediction target=chunk p plans flag cols width k prediction target := by
  change HostReduction.eval p.add zero (fun lane : Fin width =>
    Tile.laneLoss (model p plans) flag cols (k*width+lane.val) prediction target) (plans width).tree=_
  unfold chunk
  apply HostReduction.eval_congr
  intro lane
  by_cases active : k*width+lane.val<cols
  · simp only [Tile.laneLoss,Tile.loaded,active,if_true]
    exact loss_lowering p plans flag _ _
  · simp only [Tile.laneLoss,active,if_false]
    norm_num [Tile.zero,model,algebra,literal]

theorem forward_reference {M β : Type} (ops : Ops M Word β) (neg : Negation Word)
    (p : Primitives) (plans : Plans) (prediction target : Matrix) (flag : Bool) (mode : Math.Mode)
    (memory : M) (r c : Nat) :
    forwardReference ops neg (adapter p plans) prediction target flag mode memory r c=
      forwardTarget p plans flag mode prediction.cols (row ops neg memory prediction r) (row ops neg memory target r) c := by
  unfold forwardReference forwardTarget
  split
  · exact point_lowering p plans flag _ _
  · exact chunk_lowering p plans flag _ _ _ _ _

#specNonCircular Primitives avoiding [forwardSource,backwardSource,loss,derivative]
#specNonCircular forwardTarget avoiding [forwardSource,Tile.loss,Tile.chunkSum,Storage.forwardValue,forwardReference,algebra]
#specNonCircular derivative avoiding [backwardSource,Tile.gradient,Storage.backwardValue,algebra]
#axiomsClean point_lowering
#axiomsClean derivative_lowering
#axiomsClean chunk_lowering
#axiomsClean forward_reference
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.SourceNative
