import bench.tritonbench_g.rms_norm_triton.Proofs.KernelBoundary

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
open VeriTile.Triton FP.Structural
open Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

/-- Source operations, including the entire unreassociated reduction input.
The interpreter builds these nodes before any native contraction. -/
inductive Term (α : Type) where
  | input (value : α)
  /-- A scalar supplied by the call ABI, distinct from a source literal. -/
  | parameter (name : String)
  /-- Integer call value converted at its actual floating use. -/
  | fromNat (precision : Option ComputeDType) (value : Nat)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Term α)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Term α)
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary) (left right : Term α)
  | sum (precision : Option ComputeDType) (width : Nat) (values : Fin width → Term α)
  | unused

instance : Inhabited (Term α) := ⟨.unused⟩

def graphSum (p : Option ComputeDType) {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values (Term α) .real shape) : Values (Term α) .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => .sum p (TileShape.axisDim shape ax)
      (fun j => xs (TileShape.insertAxisIndex shape ax out j))
  · exact fun out => .sum p (TileShape.axisDim shape ax)
      (fun j => xs (TileShape.replaceAxisIndex shape ax out j))

def graphAlgebra : Algebra (Term α) where
  literal := Term.literal
  negInf := .unused
  binary := Term.binary
  unary := Term.unary
  cast := Term.cast
  fromNat := Term.fromNat
  fromInt := fun _ _ => .unused
  fp32Bits := fun _ => .unused
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => .unused
  reduceSum := fun p {_} ax keep xs => graphSum p ax keep xs

/-- Actual ordered reduction graph. `fma i rest` retains a square's operands
until the native fused instruction. Each padded input occurs exactly once.
Different output lanes may have different roots after butterfly shuffles. -/
inductive SumTree (B : Nat) where
  | zero
  | leaf (i : Fin B)
  | add (left right : SumTree B)
  | fma (i : Fin B) (rest : SumTree B)

def SumTree.leaves : SumTree B → List (Fin B)
  | .zero => []
  | .leaf i => [i]
  | .add left right => left.leaves++right.leaves
  | .fma i rest => i::rest.leaves

structure ReductionFamily where
  tree : (B lane : Nat) → SumTree B
  complete : ∀ B lane, (tree B lane).leaves.Perm (List.finRange B)

/-- Only local native instructions and the actual runtime epsilon word.
`epsilon` is a call argument, not an output callback or literal-equality law. -/
structure NativeModel (α : Type) where
  literal : ℝ → α
  /-- Native integer-to-FP32 conversion for non-specialized call integers. -/
  fromNat : Nat → α
  epsilon : α
  mul : α → α → α
  add : α → α → α
  fma : α → α → α → α
  div : α → α → α
  sqrt : α → α
  unused : α

/-- A local lowering result retains product operands so subsequent native FMA
contractions do not consume a separately rounded product. -/
structure Lowered (α : Type) where
  value : α
  product : Option (α × α) := none
  constant : Option ℝ := none
  reciprocal : Bool := Bool.false

def SumTree.evaluate (M : NativeModel α) (xs : Fin B → Lowered α) : SumTree B → α
  | .zero => M.literal 0
  | .leaf i => (xs i).value
  | .add left right => M.add (left.evaluate M xs) (right.evaluate M xs)
  | .fma i rest => match (xs i).product with
    | some (x,y) => M.fma x y (rest.evaluate M xs)
    | none => M.unused

/-- Singleton reduction is eliminated by the compiler, preserving its product
for the N=1 x*x+epsilon FMA. Other reductions use the supplied actual tree. -/
def reduceInfo (M : NativeModel α) (profile : ReductionFamily) (B lane : Nat)
    (xs : Fin B → Lowered α) : Lowered α :=
  if h : B = 1 then xs ⟨0,by omega⟩
  else ⟨(profile.tree B lane).evaluate M xs,none,none,Bool.false⟩

def literalInfo (M : NativeModel α) (r : ℝ) : Lowered α :=
  ⟨M.literal r,none,some r,Bool.false⟩

/-- Triton specializes an integer call value of one. Other integer arguments
retain the native conversion and are not treated as floating constants. -/
def naturalInfo (M : NativeModel α) (n : Nat) : Lowered α :=
  if n = 1 then literalInfo M 1 else ⟨M.fromNat n,none,none,Bool.false⟩

/-- The binder's one-specialization and an ordinary runtime integer remain
separate even when a native model gives literals and conversions different values. -/
theorem naturalInfo_one (M : NativeModel α) : naturalInfo M 1 = literalInfo M 1 := by
  simp [naturalInfo]
theorem naturalInfo_runtime (M : NativeModel α) (n : Nat) (h : n ≠ 1) :
    naturalInfo M n = ⟨M.fromNat n,none,none,Bool.false⟩ := by
  simp [naturalInfo,h]

def multiplyInfo (M : NativeModel α) (left right : Lowered α) : Lowered α :=
  ⟨if right.reciprocal then M.mul right.value left.value else M.mul left.value right.value,
    some (left.value,right.value),none,Bool.false⟩

def divideInfo (M : NativeModel α) (left right : Lowered α) : Lowered α :=
  if right.constant = some 1 then left
  else ⟨M.div left.value right.value,none,none,decide (left.constant = some 1)⟩

/-- Native source add is variance+epsilon. Runtime epsilon keeps its exact word,
including signed zero; the Real AST literal is not mistaken for that word.
The observed instruction orders epsilon before variance unless fused. -/
def addEpsilonInfo (M : NativeModel α) (left : Lowered α) : Lowered α :=
  ⟨match left.product with
    | some (x,y) => M.fma x y M.epsilon
    | none => M.add M.epsilon left.value,none,none,Bool.false⟩

def lower (M : NativeModel α) (profile : ReductionFamily) (lane : Nat) : Term α → Lowered α
  | .input value => ⟨value,none,none,Bool.false⟩
  | .fromNat (some .fp32) n => naturalInfo M n
  | .parameter "eps" => ⟨M.epsilon,none,none,Bool.false⟩
  | .literal (some .fp32) .real r => literalInfo M r
  | .cast (some .fp32) .real .real value => lower M profile lane value
  | .unary (some .fp32) .sqrt value => ⟨M.sqrt (lower M profile lane value).value,none,none,Bool.false⟩
  | .binary (some .fp32) .real .mul left right =>
      multiplyInfo M (lower M profile lane left) (lower M profile lane right)
  | .binary (some .fp32) .real .div left right =>
      divideInfo M (lower M profile lane left) (lower M profile lane right)
  | .binary (some .fp32) .real .add left (.cast (some .fp32) .real .real (.literal (some .fp32) .real _)) =>
      addEpsilonInfo M (lower M profile lane left)
  | .binary (some .fp32) .real .add left (.parameter "eps") =>
      addEpsilonInfo M (lower M profile lane left)
  | .sum (some .fp32) B values => reduceInfo M profile B lane (fun i => lower M profile lane (values i))
  | _ => ⟨M.unused,none,none,Bool.false⟩

def interpret (M : NativeModel α) (profile : ReductionFamily) (lane : Nat) (t : Term α) : α :=
  (lower M profile lane t).value

/-- Independent native recurrence with actual scalar instructions. The tree
and lane root are structural data, not a numerical callback for a full row. -/
def reference (M : NativeModel α) (profile : ReductionFamily) (N B : Nat)
    (xs ws : Nat → α) (i : Nat) : α :=
  let inputs : Fin B → Lowered α := fun j =>
    let x := if j.val < N then xs j.val else M.literal 0
    ⟨x,none,none,Bool.false⟩
  let squares := fun j => multiplyInfo M (inputs j) (inputs j)
  let total := reduceInfo M profile B i squares
  let variance := divideInfo M total (literalInfo M N)
  let radicand := addEpsilonInfo M variance
  let root : Lowered α := ⟨M.sqrt radicand.value,none,none,Bool.false⟩
  let rrms := divideInfo M (literalInfo M 1) root
  (multiplyInfo M (multiplyInfo M ⟨xs i,none,none,Bool.false⟩ rrms)
    ⟨ws i,none,none,Bool.false⟩).value

/-- Singleton reduction/division are eliminated before contraction: the
source's x*x+eps is one FMA, not a rounded multiplication followed by add. -/
theorem reference_singleton (M : NativeModel α) (profile : ReductionFamily) (xs ws : Nat → α) :
    reference M profile 1 1 xs ws 0 =
      M.mul (M.mul (M.div (M.literal 1) (M.sqrt (M.fma (xs 0) (xs 0) M.epsilon))) (xs 0)) (ws 0) := by
  simp [reference,reduceInfo,divideInfo,literalInfo,multiplyInfo,addEpsilonInfo]

theorem source_graph_lowering (M : NativeModel α) (profile : ReductionFamily)
    (c : Layout) (eps : ℝ) (xs ws : Nat → α) (i : Nat) :
    interpret M profile i (rowValue graphAlgebra c eps (fun j => .input (xs j)) (fun j => .input (ws j)) i) =
      reference M profile c.N c.B xs ws i := by
  simp only [interpret,rowValue,inputTile,graphAlgebra,graphSum,lower,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,reference,id_eq]
  have squares : (fun j : Fin c.B => multiplyInfo M
      (lower M profile i (if j.val < c.N then .input (xs j.val) else .literal (some .fp32) .real 0))
      (lower M profile i (if j.val < c.N then .input (xs j.val) else .literal (some .fp32) .real 0))) =
      (fun j : Fin c.B => multiplyInfo M
        ⟨if j.val < c.N then xs j.val else M.literal 0,none,none,Bool.false⟩
        ⟨if j.val < c.N then xs j.val else M.literal 0,none,none,Bool.false⟩) := by
    funext j
    split_ifs <;> rfl
  norm_num only
  rw [squares]

def entryGraph (entry : Memory α) : Memory (Term α) := fun allocation offset => .input (entry allocation offset)

/-- The physical output address determines the logical lane/root. This matters
for native butterfly FMA contractions; no uniform rounded row sum is assumed. -/
def projectMemory (M : NativeModel α) (profile : ReductionFamily) (N base : Nat)
    (memory : Memory (Term α)) : Memory α := fun allocation offset =>
  interpret M profile ((offset-base)%N) (memory allocation offset)


end
#guard_msgs (drop info) in
#specNonCircular reference avoiding [lower,interpret,rowValue,sourceRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton.Observed
