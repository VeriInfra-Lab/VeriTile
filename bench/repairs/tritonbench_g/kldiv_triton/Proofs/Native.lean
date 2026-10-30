import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Tile

namespace VeriTile.Bench.Repairs.KldivTriton.Native
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 3000000

abbrev Word := BitVec 32
def word (n : Nat) : Word := BitVec.ofNat 32 n

/-- Pinned local FP32 primitive ports. `logTL` is the actual installed
polynomial/FMA/FTZ intrinsic, not Real.log or a guessed lg2 instruction.
The exp path exposes its real multiply-by-log2(e) and ex2.approx separately. -/
structure Instructions where
  addRN : Word → Word → Word
  subRN : Word → Word → Word
  mulRN : Word → Word → Word
  div : Word → Word → Word
  logTL : Word → Word
  ex2Approx : Word → Word
  equal : Word → Word → Bool

def numWarps (B : Nat) : Nat := if 32768 ≤ B then 32 else if 8192 ≤ B then 16
  else if 2048 ≤ B then 8 else 4

def butterfly (add : Word → Word → Word) : List Nat → (Nat → Word) → Nat → Word
  | [], values => values
  | distance :: rest, values => butterfly add rest
      (fun lane => add (values lane) (values (Nat.xor lane distance)))

/-- The inspected PTX starts each thread with slot zero, then adds slots
1,2,... in order. There is no extra zero seed and no fused loss/add operation. -/
def threadSum (p : Instructions) (B : Nat) (values : Nat → Word) (thread : Nat) : Word :=
  let threads := 32 * numWarps B
  (List.range (B / threads - 1)).foldl
    (fun accum slot => p.addRN accum (values (thread + threads * (slot + 1))))
    (values thread)

def warpDistances (B : Nat) := [16,8,4,2,1].filter (fun d => d < min B 32)
def groupDistances (B : Nat) := [16,8,4,2,1].filter (fun d => d < min (B / 32) (numWarps B))

/-- Pinned block reduction graph: local slot fold, descending warp butterfly,
then descending butterfly over shared warp leaders. The 15 source profiles
and both flags/all three reductions are audited in native-reductions001. -/
def reduction (p : Instructions) (B : Nat) (values : Nat → Word) : Word :=
  let warps := butterfly p.addRN (warpDistances B) (threadSum p B values)
  if B ≤ 32 then warps 0
  else butterfly p.addRN (groupDistances B) (fun warp => warps (warp * 32)) 0

def exp (p : Instructions) (target : Word) : Word :=
  p.ex2Approx (p.mulRN target (word 0x3fb8aa3b))

/-- Exact loss operand order in the pinned FP32 Triton 3.7.1 SM90 profiles.
Raw-target loss is difference-first at every block. Log-target loss changes
from factor-first to difference-first at block 512. This is local compiler
lowering of the source multiply, not an assumed commutativity law. -/
inductive LossMulOrder where
  | factorFirst
  | differenceFirst
  deriving DecidableEq

def lossOrder (B : Nat) (flag : Bool) : LossMulOrder :=
  if flag then if B < 512 then .factorFirst else .differenceFirst
  else .differenceFirst

def lossMultiply (p : Instructions) (order : LossMulOrder) (factor difference : Word) : Word :=
  match order with
  | .factorFirst => p.mulRN factor difference
  | .differenceFirst => p.mulRN difference factor

theorem loss_order_profiles : ∀ k : Fin 15,
    lossOrder (2^k.val) Bool.false = .differenceFirst ∧
    lossOrder (2^k.val) Bool.true = if k.val < 9 then .factorFirst else .differenceFirst := by
  decide

/-- Abstract noncommutative-port control: swapping operands is observable.
This is not a claim about a numerical GPU counterexample. -/
theorem loss_order_observable (p : Instructions) (a b : Word) (different : a ≠ b) :
    lossMultiply {p with mulRN := fun x _ => x} .factorFirst a b ≠
      lossMultiply {p with mulRN := fun x _ => x} .differenceFirst a b := different

def forwardPoint (p : Instructions) (B : Nat) (flag : Bool) (prediction target : Word) : Word :=
  if flag then lossMultiply p (lossOrder B flag) (exp p target) (p.subRN target prediction)
  else if p.equal target (word 0) then word 0
  else lossMultiply p (lossOrder B flag) target (p.subRN (p.logTL target) prediction)

/-- Default target*(-1) lowers to XOR; the log-target unary minus lowers
to subtraction from positive zero in this exact pinned native profile. -/
def backwardPoint (p : Instructions) (flag : Bool) (target : Word) : Word :=
  if flag then p.subRN (word 0) (exp p target)
  else target ^^^ word 0x80000000

def chunk (p : Instructions) (flag : Bool) (N B k : Nat) (ys ts : Nat → Word) : Word :=
  reduction p B (fun lane =>
    if lane < B ∧ k * B + lane < N then forwardPoint p B flag (ys (k * B + lane)) (ts (k * B + lane))
    else word 0)

/-- Provenance for the source's literal -1 only. A data word equal to -1
does not acquire this tag, so a general multiplication cannot become XOR. -/
structure Node where
  value : Word
  literalMinusOne : Bool := Bool.false
  deriving Inhabited

def atom (value : Word) : Node := ⟨value, Bool.false⟩
def literal (x : ℝ) : Node :=
  if x = -1 then ⟨word 0xbf800000, Bool.true⟩
  else atom (if x = 1 then word 0x3f800000 else word 0)

def multiply (p : Instructions) (a b : Node) : Node :=
  atom (if b.literalMinusOne then a.value ^^^ word 0x80000000 else p.mulRN a.value b.value)

def sumValues (p : Instructions) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (values : Values Node .real shape) :
    Values Node .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => atom (reduction p (TileShape.axisDim shape ax) (fun j =>
      if h : j < TileShape.axisDim shape ax then
        (values (TileShape.insertAxisIndex shape ax out ⟨j,h⟩)).value else word 0))
  · exact fun out => atom (reduction p (TileShape.axisDim shape ax) (fun j =>
      if h : j < TileShape.axisDim shape ax then
        (values (TileShape.replaceAxisIndex shape ax out ⟨j,h⟩)).value else word 0))

def algebra (p : Instructions) : Algebra Node where
  literal := fun _ _ => literal
  negInf := atom (word 0xff800000)
  binary := fun _ _ op a b => match op with
    | .add => atom (p.addRN a.value b.value)
    | .sub => atom (p.subRN a.value b.value)
    | .mul => multiply p a b
    | .div => atom (p.div a.value b.value)
    | _ => atom (word 0)
  unary := fun _ op a => match op with
    | .log => atom (p.logTL a.value)
    | .exp => atom (exp p a.value)
    | .neg _ => atom (p.subRN (word 0) a.value)
    | _ => atom (word 0)
  compareEq := fun _ _ => some (fun a b => p.equal a.value b.value)
  cast := fun _ _ _ a => a
  fromNat := fun _ _ => atom (word 0)
  fromInt := fun _ _ => atom (word 0)
  fp32Bits := fun _ => atom (word 0)
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => atom (word 0)
  reduceSum := fun _ => sumValues p

def model (p : Instructions) : Tile.Model Node :=
  ⟨algebra p, (fun a b => p.equal a.value b.value), rfl⟩

/-- Only the source loss multiply uses the selected compiler operand order.
The exp scaling multiply is inside `exp`; reduction additions and backward
literal-minus-one lowering retain their own explicit graphs. -/
def forwardAlgebra (p : Instructions) (B : Nat) (flag : Bool) : Algebra Node :=
  { algebra p with
    binary := fun precision dtype op a b => match op with
      | .mul => atom (lossMultiply p (lossOrder B flag) a.value b.value)
      | _ => (algebra p).binary precision dtype op a b }

def forwardModel (p : Instructions) (B : Nat) (flag : Bool) : Tile.Model Node :=
  ⟨forwardAlgebra p B flag, (fun a b => p.equal a.value b.value), rfl⟩

theorem forward_lowering (p : Instructions) (B : Nat) (flag : Bool) (y t : Word) :
    Tile.loss (forwardModel p B flag) flag (atom y) (atom t) = atom (forwardPoint p B flag y t) := by
  cases flag <;> norm_num [Tile.loss, Tile.rawLoss, Tile.zero, forwardModel, forwardAlgebra, algebra,
    literal, atom, forwardPoint]
  split <;> rfl

theorem backward_lowering (p : Instructions) (flag : Bool) (t : Word) :
    Tile.gradient (model p) flag (atom t) = atom (backwardPoint p flag t) := by
  cases flag <;> simp [Tile.gradient, model, algebra, multiply, literal, atom, backwardPoint]

theorem chunk_lowering (p : Instructions) (flag : Bool) (N B k : Nat) (ys ts : Nat → Word) :
    Tile.chunkSum (forwardModel p B flag) flag N B k (fun i => atom (ys i)) (fun i => atom (ts i)) =
      atom (chunk p flag N B k ys ts) := by
  change atom (reduction p B (fun j => if h : j < B then
    (Tile.laneLoss (forwardModel p B flag) flag N (k * B + j) (fun i => atom (ys i)) (fun i => atom (ts i))).value
    else word 0)) = atom (chunk p flag N B k ys ts)
  unfold chunk
  congr 2
  funext j
  by_cases hb : j < B <;> by_cases active : k * B + j < N
  · simp only [hb, active, and_self, if_true, Tile.laneLoss, Tile.loaded]
    change (Tile.loss (forwardModel p B flag) flag (atom (ys (k*B+j))) (atom (ts (k*B+j)))).value = _
    rw [forward_lowering]
    rfl
  · norm_num [hb, active, Tile.laneLoss, Tile.zero, forwardModel, forwardAlgebra, algebra, literal, atom]
  · simp [hb]
  · simp [hb]

/-- Numerical deferrals are local raw-word primitive equalities only.
They cannot assert execution success, output mapping, a reduction equation,
or a memory/protocol property. All payloads and signed zeros remain in scope. -/
structure Obligations (actual target : Instructions) : Prop where
  add : ∀ a b, actual.addRN a b = target.addRN a b
  sub : ∀ a b, actual.subRN a b = target.subRN a b
  mul : ∀ a b, actual.mulRN a b = target.mulRN a b
  div : ∀ a b, actual.div a b = target.div a b
  log : ∀ a, actual.logTL a = target.logTL a
  ex2 : ∀ a, actual.ex2Approx a = target.ex2Approx a
  eq : ∀ a b, actual.equal a b = target.equal a b

theorem instructions_eq (actual target : Instructions) (h : Obligations actual target) :
    actual = target := by
  cases actual; cases target
  simp only [Instructions.mk.injEq]
  exact ⟨funext fun a => funext (h.add a), funext fun a => funext (h.sub a),
    funext fun a => funext (h.mul a), funext fun a => funext (h.div a),
    funext h.log, funext h.ex2, funext fun a => funext (h.eq a)⟩

theorem chunk_equiv (actual target : Instructions) (h : Obligations actual target)
    (flag : Bool) (N B k : Nat) (ys ts : Nat → Word) :
    chunk actual flag N B k ys ts = chunk target flag N B k ys ts := by
  rw [instructions_eq actual target h]

#specNonCircular forwardPoint avoiding [algebra, forwardAlgebra, Tile.loss]
#specNonCircular backwardPoint avoiding [algebra, Tile.gradient]
#specNonCircular chunk avoiding [algebra, forwardAlgebra, Tile.chunkSum]
#axiomsClean loss_order_profiles
#axiomsClean loss_order_observable
#axiomsClean forward_lowering
#axiomsClean backward_lowering
#axiomsClean chunk_lowering
#axiomsClean chunk_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Native
