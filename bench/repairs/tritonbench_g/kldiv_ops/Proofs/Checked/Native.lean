import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Checked.Host
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.NativeBridge

namespace VeriTile.Bench.Repairs.KldivOps.Contiguous.Native
open VeriTile.Triton FP.Structural
open KldivTriton.Native (Word Node atom literal word)
open FP.Equational (ReductionPlan)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 6000000

/-- Fixed-arity scalar source-lowering ports. Operand roles remain in source
order; any installed compiler operand selection is part of that local port.
No reduction, output, memory, address or execution predicate is a port. -/
structure Ports where
  add : Word → Word → Word
  sub : Word → Word → Word
  multiply : Word → Word → Word
  maximum : Word → Word → Word
  logarithm : Word → Word
  exponential : Word → Word
  negate : Word → Word
  multiplyMinusOne : Word → Word
  bindEpsilon : BitVec 64 → Word

/-- Actual compiler-selected ordered binary addition trees. Coverage belongs
to ReductionPlan, and each scalar addition is interpreted by Ports.add. -/
abbrev Plans := (width : Nat) → ReductionPlan width

def sumValues (p : Ports) (plans : Plans) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (values : Values Node .real shape) :
    Values Node .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => atom (KldivTriton.HostReduction.eval p.add (word 0)
      (fun j => (values (TileShape.insertAxisIndex shape ax out j)).value)
      (plans (TileShape.axisDim shape ax)).tree)
  · exact fun out => atom (KldivTriton.HostReduction.eval p.add (word 0)
      (fun j => (values (TileShape.replaceAxisIndex shape ax out j)).value)
      (plans (TileShape.axisDim shape ax)).tree)

def algebra (p : Ports) (plans : Plans) : Algebra Node where
  literal := fun _ _ => literal
  negInf := atom (word 0xff800000)
  binary := fun _ _ op a b => match op with
    | .add => atom (p.add a.value b.value)
    | .sub => atom (p.sub a.value b.value)
    | .mul => atom (if b.literalMinusOne then p.multiplyMinusOne a.value else p.multiply a.value b.value)
    | .max => atom (p.maximum a.value b.value)
    | _ => atom (word 0)
  unary := fun _ op a => match op with
    | .log => atom (p.logarithm a.value)
    | .exp => atom (p.exponential a.value)
    | .neg _ => atom (p.negate a.value)
    | _ => atom (word 0)
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => atom (word 0)
  fromInt := fun _ _ => atom (word 0)
  fp32Bits := fun _ => atom (word 0)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => atom (word 0)
  reduceSum := fun _ => sumValues p plans

def adapter (p : Ports) (plans : Plans) : Adapter Word Node :=
  ⟨algebra p plans,atom,Node.value⟩

def loss (p : Ports) (flag : Bool) (eps y t : Word) : Word :=
  if flag then p.multiply (p.exponential t) (p.sub t y)
  else p.multiply t (p.sub (p.logarithm (p.maximum t eps)) y)

def block (p : Ports) (plans : Plans) (cols width : Nat) (flag : Bool)
    (eps : Word) (ys ts : Nat → Word) (start : Nat) : Word :=
  KldivTriton.HostReduction.eval p.add (word 0)
    (fun j => if start+j.val < cols then loss p flag eps (ys (start+j.val)) (ts (start+j.val)) else word 0)
    (plans width).tree

def accumulate (p : Ports) (plans : Plans) (cols width : Nat) (flag : Bool)
    (eps : Word) (ys ts : Nat → Word) (start : Nat) (carry : Word) : Word :=
  if h : 0 < width ∧ start < cols then
    accumulate p plans cols width flag eps ys ts (start+width)
      (p.add carry (block p plans cols width flag eps ys ts start))
  else carry
termination_by cols-start
decreasing_by omega

def row (p : Ports) (plans : Plans) (cols : Nat) (flag : Bool) (eps : Word)
    (ys ts : Nat → Word) : Word :=
  accumulate p plans cols (Host.blockSize cols) flag eps ys ts 0 (word 0)

def derivative (p : Ports) (flag : Bool) (t : Word) : Word :=
  if flag then p.negate (p.exponential t) else p.multiplyMinusOne t

theorem scalar_lowering (p : Ports) (plans : Plans) (flag : Bool) (eps y t : Word) :
    Tile.scalar (algebra p plans) flag (atom eps) (atom y) (atom t) = atom (loss p flag eps y t) := by
  cases flag <;> rfl

theorem backward_lowering (p : Ports) (plans : Plans) (flag : Bool) (t : Word) :
    Backward.scalar (algebra p plans) flag (atom t) = atom (derivative p flag t) := by
  cases flag <;> simp [Backward.scalar,algebra,literal,derivative,atom]

theorem block_lowering (p : Ports) (plans : Plans) (cols width : Nat) (flag : Bool)
    (eps : Word) (ys ts : Nat → Word) (start : Nat) :
    FPGraph.block (algebra p plans) cols width flag (atom eps)
      (fun i => atom (ys i)) (fun i => atom (ts i)) start = atom (block p plans cols width flag eps ys ts start) := by
  unfold FPGraph.block block
  simp only [algebra,sumValues,TileShape.axisDim,TileShape.insertAxisIndex]
  congr 1
  apply KldivTriton.HostReduction.eval_congr
  intro j
  split
  next active => exact congrArg Node.value (scalar_lowering p plans flag eps (ys (start+j.val)) (ts (start+j.val)))
  next inactive => simp [Tile.zero,algebra,literal,atom]

theorem accumulate_lowering (p : Ports) (plans : Plans) (cols width : Nat) (flag : Bool)
    (eps : Word) (ys ts : Nat → Word) (start : Nat) (carry : Word) :
    FPGraph.accumulate (algebra p plans) cols width flag (atom eps)
      (fun i => atom (ys i)) (fun i => atom (ts i)) start (atom carry) =
      atom (accumulate p plans cols width flag eps ys ts start carry) := by
  rw [FPGraph.accumulate,accumulate]
  split
  next active =>
    rw [block_lowering]
    change FPGraph.accumulate (algebra p plans) cols width flag (atom eps)
      (fun i => atom (ys i)) (fun i => atom (ts i)) (start+width)
      (atom (p.add carry (block p plans cols width flag eps ys ts start))) = _
    exact accumulate_lowering p plans cols width flag eps ys ts (start+width) _
  next inactive => rfl
termination_by cols-start
decreasing_by omega

theorem row_lowering (p : Ports) (plans : Plans) (cols : Nat) (flag : Bool) (eps : Word)
    (ys ts : Nat → Word) :
    Contiguous.row (adapter p plans) cols flag eps ys ts = row p plans cols flag eps ys ts := by
  unfold Contiguous.row adapter FPGraph.row
  have zero : Tile.zero (algebra p plans) = atom (word 0) := by simp [Tile.zero,algebra,literal]
  rw [zero,accumulate_lowering]
  rfl

structure LocalPorts (actual target : Ports) : Prop where
  add : ∀ a b, actual.add a b = target.add a b
  sub : ∀ a b, actual.sub a b = target.sub a b
  multiply : ∀ a b, actual.multiply a b = target.multiply a b
  maximum : ∀ a b, actual.maximum a b = target.maximum a b
  logarithm : ∀ a, actual.logarithm a = target.logarithm a
  exponential : ∀ a, actual.exponential a = target.exponential a
  negate : ∀ a, actual.negate a = target.negate a
  multiplyMinusOne : ∀ a, actual.multiplyMinusOne a = target.multiplyMinusOne a
  bindEpsilon : ∀ bits, actual.bindEpsilon bits = target.bindEpsilon bits

theorem ports_eq (actual target : Ports) (h : LocalPorts actual target) : actual = target := by
  cases actual;cases target
  simp only [Ports.mk.injEq]
  exact ⟨funext fun a => funext (h.add a),funext fun a => funext (h.sub a),
    funext fun a => funext (h.multiply a),funext fun a => funext (h.maximum a),
    funext h.logarithm,funext h.exponential,funext h.negate,funext h.multiplyMinusOne,
    funext h.bindEpsilon⟩

#specNonCircular loss avoiding [Tile.scalar,Program.grid,forwardKernel]
#specNonCircular row avoiding [FPGraph.row,Program.grid,forwardKernel]
#axiomsClean scalar_lowering
#axiomsClean backward_lowering
#axiomsClean row_lowering
#axiomsClean ports_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Contiguous.Native
