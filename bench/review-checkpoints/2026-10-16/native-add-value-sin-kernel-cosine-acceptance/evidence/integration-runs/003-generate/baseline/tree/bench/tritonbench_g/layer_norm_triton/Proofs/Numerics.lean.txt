import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.Float.TermModel
import VeriTile.Meta.StatementAudit

/-! Task-local ordered FP graph for all three kernels. No Real arithmetic or
algebraic sum law is assumed. Every reduction uses a valid explicit tree.
The source's unmasked arithmetic on padded lanes and undefined masked W/B
loads are retained. Sqrt is a named primitive callback, not Real.sqrt. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormLock
open VeriTile.Triton
open FP.Equational (ReductionTree ReductionPlan)
abbrev I32 := BitVec 32

/-- Explicit fp32-compute source profiles. Float64 affine operands would promote
those operations and are NOT silently interpreted as fp32 by this model. -/
inductive Storage where | fp16 | bf16 | fp32
  deriving DecidableEq, Repr
inductive Region where | x | w | bias | dy | mean | rstd | y | dx | dw | db | finalDW | finalDB
  deriving DecidableEq, Repr

structure DTypes where
  x : Storage := .fp32
  w : Storage := .fp32
  bias : Storage := .fp32
  dy : Storage := .fp32
  y : Storage := .fp32
  dx : Storage := .fp32
  partialType : Storage := .fp32
  finalType : Storage := .fp32
  deriving DecidableEq, Repr

structure Backend (V : Type) where
  zero : V
  zeroStored : Storage → V
  one : V
  add : V → V → V
  sub : V → V → V
  mul : V → V → V
  div : V → V → V
  sqrt : V → V
  fromI32 : I32 → V
  cast : Storage → Storage → V → V

structure Config where
  grid : Nat
  block : Nat
  n : I32
  stride : I32
  group : I32
  dtype : DTypes := {}
  deriving DecidableEq, Repr

abbrev Memory (V : Type) := Region → Int → V
abbrev Undef (V : Type) := Region → Int → V

def put {α β : Type} [DecidableEq α] (f : α → β) (k : α) (v : β) : α → β :=
  fun j => if j = k then v else f j
@[simp] theorem put_same {α β : Type} [DecidableEq α] (f : α → β) (k : α) (v : β) :
    put f k v k = v := by simp [put]
@[simp] theorem put_other {α β : Type} [DecidableEq α] (f : α → β) (k j : α) (v : β)
    (h : j ≠ k) : put f k v j = f j := by simp [put, h]

def rowAddr (c : Config) (p : Nat) (col : I32) : Int :=
  (BitVec.ofNat 32 p * c.stride).toInt + col.toInt
def colAt (off : I32) (lane : Nat) : I32 := off + BitVec.ofNat 32 lane
def live (c : Config) (off : I32) (lane : Nat) : Bool := (colAt off lane).slt c.n

def reduce (b : Backend V) (xs : Fin n → V) : ReductionTree n → V
  | .input i => xs i
  | .zero => b.zero
  | .add a d => b.add (reduce b xs a) (reduce b xs d)

def sum (b : Backend V) (plan : ReductionPlan n) (xs : Nat → V) : V :=
  reduce b (fun i => xs i.val) plan.tree

/-- Imperative vector accumulator across source loop offsets. -/
def streamRun (b : Backend V) (read : I32 → Nat → V) :
    List I32 → (Nat → V) → (Nat → V)
  | [], state => state
  | off :: rest, state => streamRun b read rest (fun lane => b.add (state lane) (read off lane))

/-- Independent per-lane chronological fold; padding reads are not discarded. -/
def streamReference (b : Backend V) (read : I32 → Nat → V)
    (offsets : List I32) (initial : Nat → V) : Nat → V :=
  fun lane => (offsets.map (fun off => read off lane)).foldl b.add (initial lane)

theorem stream_value (b : Backend V) (read : I32 → Nat → V)
    (offsets : List I32) (initial : Nat → V) :
    streamRun b read offsets initial = streamReference b read offsets initial := by
  induction offsets generalizing initial with
  | nil => rfl
  | cons off rest ih =>
    simp only [streamRun, ih]
    rfl

def input32 (b : Backend V) (c : Config) (mem : Memory V) (region : Region)
    (dtype : Storage) (p : Nat) (off : I32) (lane : Nat) : V :=
  b.cast dtype .fp32 (if live c off lane then mem region (rowAddr c p (colAt off lane)) else b.zeroStored dtype)

def xRead (b : Backend V) (c : Config) (mem : Memory V) (p : Nat) : I32 → Nat → V :=
  input32 b c mem .x c.dtype.x p

def varianceRead (b : Backend V) (c : Config) (mem : Memory V) (p : Nat)
    (mean : V) (off : I32) (lane : Nat) : V :=
  let x := input32 b c mem .x c.dtype.x p off lane
  let centered := if live c off lane then b.sub x mean else b.zero
  b.mul centered centered

structure ForwardValues (V : Type) where
  mean : V
  rstd : V
  y : I32 → Nat → V

/-- Source loops maintain vector _mean/_var registers, then reduce each vector. -/
def forwardSource (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p : Nat) (offsets : List I32) : ForwardValues V :=
  let meanReg := streamRun b (xRead b c mem p) offsets (fun _ => b.zero)
  let mean := b.div (sum b plan meanReg) (b.fromI32 c.n)
  let varReg := streamRun b (varianceRead b c mem p mean) offsets (fun _ => b.zero)
  let variance := b.div (sum b plan varReg) (b.fromI32 c.n)
  let rstd := b.div b.one (b.sqrt (b.add variance eps))
  let y := fun off lane =>
    let col := colAt off lane
    let w := if live c off lane then mem .w col.toInt else undef .w col.toInt
    let bias := if live c off lane then mem .bias col.toInt else undef .bias col.toInt
    let x := input32 b c mem .x c.dtype.x p off lane
    let xhat := b.mul (b.sub x mean) rstd
    b.cast .fp32 c.dtype.y (b.add (b.mul xhat (b.cast c.dtype.w .fp32 w)) (b.cast c.dtype.bias .fp32 bias))
  ⟨mean, rstd, y⟩

/-- Closed source-order recurrence written independently of vector state. -/
def forwardReference (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p : Nat) (offsets : List I32) : ForwardValues V :=
  let mean := b.div (sum b plan (fun lane =>
    (offsets.map (fun off => xRead b c mem p off lane)).foldl b.add b.zero)) (b.fromI32 c.n)
  let variance := b.div (sum b plan (fun lane =>
    (offsets.map (fun off => varianceRead b c mem p mean off lane)).foldl b.add b.zero)) (b.fromI32 c.n)
  let inv := b.div b.one (b.sqrt (b.add variance eps))
  ⟨mean, inv, fun off lane =>
    b.cast .fp32 c.dtype.y
      (b.add
        (b.mul (b.mul (b.sub (xRead b c mem p off lane) mean) inv)
          (b.cast c.dtype.w .fp32 (if live c off lane then mem .w (colAt off lane).toInt else undef .w (colAt off lane).toInt)))
        (b.cast c.dtype.bias .fp32 (if live c off lane then mem .bias (colAt off lane).toInt else undef .bias (colAt off lane).toInt)))⟩

theorem forward_value (b : Backend V) (c : Config) (plan : ReductionPlan c.block)
    (mem : Memory V) (undef : Undef V) (eps : V) (p : Nat) (offsets : List I32) :
    forwardSource b c plan mem undef eps p offsets = forwardReference b c plan mem undef eps p offsets := by
  simp only [forwardSource, stream_value]
  rfl

/-- The final reduction's loop carries all BM×BN accumulator lanes. -/
def partialRead (b : Backend V) (c : Config) (mem : Memory V) (region : Region)
    (m : I32) (bm bn pid : Nat) (off : I32) (lane : Nat) : V :=
  let row := off + BitVec.ofNat 32 (lane / bn)
  let col := BitVec.ofNat 32 pid * BitVec.ofNat 32 bn + BitVec.ofNat 32 (lane % bn)
  let raw := if row.slt m && col.slt c.n then mem region (row * c.n + col).toInt else b.zeroStored c.dtype.partialType
  b.cast c.dtype.partialType .fp32 raw

def finalSource (b : Backend V) (c : Config) (mem : Memory V) (m : I32)
    (bm bn pid : Nat) (plan : ReductionPlan bm) (offsets : List I32) (region : Region) : Nat → V :=
  let regs := streamRun b (partialRead b c mem region m bm bn pid) offsets (fun _ => b.zero)
  fun col => b.cast .fp32 c.dtype.finalType (sum b plan (fun row => regs (row * bn + col)))

def finalReference (b : Backend V) (c : Config) (mem : Memory V) (m : I32)
    (bm bn pid : Nat) (plan : ReductionPlan bm) (offsets : List I32) (region : Region) : Nat → V :=
  fun col => b.cast .fp32 c.dtype.finalType (sum b plan (fun row =>
    (offsets.map (fun off => partialRead b c mem region m bm bn pid off (row * bn + col))).foldl b.add b.zero))

theorem final_value (b : Backend V) (c : Config) (mem : Memory V) (m : I32)
    (bm bn pid : Nat) (plan : ReductionPlan bm) (offsets : List I32) (region : Region) :
    finalSource b c mem m bm bn pid plan offsets region = finalReference b c mem m bm bn pid plan offsets region := by
  simp only [finalSource, stream_value]
  rfl

#axiomsClean stream_value
#axiomsClean forward_value
#axiomsClean final_value
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock
