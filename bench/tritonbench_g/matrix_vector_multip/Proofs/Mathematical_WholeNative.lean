import bench.tritonbench_g.matrix_vector_multip.Proofs.Mathematical_WholeSource

namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNative
open VeriTile.Triton FP.Structural WholeSource
open Typed (Parameters)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000

/-- Native primitive semantics supplied at the execution-model boundary. FMA
is one ternary instruction; no split multiplication/addition identity is used. -/
structure Scalars where
  fma : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 32
  add : BitVec 32 → BitVec 32 → BitVec 32

/-- Ordered native add graph. Leaves name physical accumulator coordinates;
there is no reduction callback, value premise, or tree-reassociation rule. -/
inductive AddTree (columns : Nat) where
  | leaf (column : Fin columns)
  | add (left right : AddTree columns)
  deriving DecidableEq

def AddTree.leaves : AddTree columns → List (Fin columns)
  | .leaf j => [j]
  | .add left right => left.leaves++right.leaves

def AddTree.eval (add : α → α → α) (values : Fin columns → α) : AddTree columns → α
  | .leaf j => values j
  | .add left right => add (left.eval add values) (right.eval add values)

/-- An exact-specialization certificate contains only finite graph structure
and FMA operand orientation. The external compiler/model bridge checks source
locations, memory/mask/carry correspondence and concrete shuffle/shared-memory
routes. Different last-writer choices may supply different ordered trees. -/
structure Graph (rows columns : Nat) where
  tree : Fin rows → AddTree columns
  reverseFMA : Fin rows → Fin columns → Bool
  complete : ∀ i,(tree i).leaves.Perm (List.finRange columns)

theorem Graph.exactly_once (g : Graph rows columns) (i : Fin rows) :
    (g.tree i).leaves.Nodup ∧ ∀ j : Fin columns,j∈(g.tree i).leaves := by
  constructor
  · exact (g.complete i).nodup_iff.mpr (List.nodup_finRange columns)
  · intro j
    exact (g.complete i).mem_iff.mpr (List.mem_finRange j)

/-- A lossless staging carrier for the original source expression. Products
are not rounded separately: the following source add is lowered to a carry
node, which is interpreted by the actual ternary FMA instruction. -/
inductive Payload where
  | word (bits : BitVec 32)
  | product (left right : Payload)
  | carry (previous left right : Payload)
  deriving Inhabited

def Payload.bits : Payload → BitVec 32
  | .word bits => bits
  | _ => 0

def evaluateCarry (B : Scalars) (reverse : Bool) : Payload → BitVec 32
  | .word bits => bits
  | .carry previous left right =>
      if reverse then B.fma right.bits left.bits (evaluateCarry B reverse previous)
      else B.fma left.bits right.bits (evaluateCarry B reverse previous)
  | .product _ _ => 0

def sumTile (B : Scalars) (g : Graph rows columns)
    (acc : Values Payload .real [rows,columns]) : Values Payload .real [rows] :=
  fun i => .word ((g.tree i.1).eval B.add
    (fun j => evaluateCarry B (g.reverseFMA i.1 j) (acc (i.1,j,PUnit.unit))))

def sum (B : Scalars) (g : Graph rows columns) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (acc : Values Payload .real shape) :
    Values Payload .real (TileShape.reduceShape shape ax keep) := by
  classical
  by_cases hs : shape=[rows,columns]
  · subst shape
    by_cases ha : ax=⟨1,by simp⟩
    · subst ax
      cases keep
      · exact sumTile B g acc
      · exact fun _ => .word 0
    · exact fun _ => .word 0
  · exact fun _ => .word 0

def algebra (B : Scalars) (g : Graph rows columns) : Algebra Payload where
  literal:=fun _ _ _ => .word 0
  negInf:=.word 0
  binary:=fun _ _ op left right => match op with
    | .mul => .product left right
    | .add => match right with
      | .product a b => .carry left a b
      | _ => .word 0
    | _ => .word 0
  unary:=fun _ _ _ => .word 0
  cast:=fun _ _ _ x => x
  fromNat:=fun _ _ => .word 0
  fromInt:=fun _ _ => .word 0
  fp32Bits:=fun _ => .word 0
  fp32Load:=id
  reduceMax:=fun _ {_} _ _ _ _ => .word 0
  reduceSum:=fun _ {_} => sum B g

def liftMemory (memory : RegionName → Nat → Cell (BitVec 32)) : RegionName → Nat → Cell Payload :=
  fun r i => .mk .fp32 (.word ((memory r i).read .fp32))

def evaluateAcc (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (acc : Acc Payload p) : Acc (BitVec 32) p :=
  fun i => evaluateCarry B (g.reverseFMA i.1 i.2.1) (acc i)

def matrixLoad (p : Parameters) (pid off : Nat) (memory : RegionName → Nat → Cell (BitVec 32))
    (i : Fin p.blockRows) (j : Fin p.blockColumns) : BitVec 32 :=
  if row p pid i.val<p.rows ∧ off+j.val<p.columns then
    (memory p.a (aOffset p pid off i.val j.val)).read .fp32 else 0

def vectorLoad (p : Parameters) (off : Nat) (memory : RegionName → Nat → Cell (BitVec 32))
    (j : Fin p.blockColumns) : BitVec 32 :=
  if off+j.val<p.columns then (memory p.b (bOffset p off j.val)).read .fp32 else 0

/-- Actual native per-slot instruction, including zero-filled masked lanes and
the exact operand order recorded by the specialization certificate. -/
def next (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (pid off : Nat) (memory : RegionName → Nat → Cell (BitVec 32))
    (acc : Acc (BitVec 32) p) : Acc (BitVec 32) p := fun i =>
  if g.reverseFMA i.1 i.2.1 then
    B.fma (vectorLoad p off memory i.2.1) (matrixLoad p pid off memory i.1 i.2.1) (acc i)
  else B.fma (matrixLoad p pid off memory i.1 i.2.1) (vectorLoad p off memory i.2.1) (acc i)

def accumulate (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (pid : Nat) (memory : RegionName → Nat → Cell (BitVec 32)) (off : Nat)
    (acc : Acc (BitVec 32) p) : Acc (BitVec 32) p :=
  if h : off<p.columns ∧ 0<p.blockColumns then
    accumulate B p g pid memory (off+p.blockColumns) (next B p g pid off memory acc)
  else acc
termination_by p.columns-off

theorem next_eq (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (pid off : Nat) (memory : RegionName → Nat → Cell (BitVec 32)) (acc : Acc Payload p) :
    evaluateAcc B p g (nextAcc (algebra B g) p pid off (liftMemory memory) acc)=
      next B p g pid off memory (evaluateAcc B p g acc) := by
  funext i
  simp only [evaluateAcc,nextAcc,algebra,aChunk,bChunk,load,liftMemory]
  simp only [evaluateCarry,next,matrixLoad,vectorLoad]
  split <;> congr 2 <;> split <;> simp [Payload.bits,Cell.read]

theorem accumulated_eq (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (pid : Nat) (memory : RegionName → Nat → Cell (BitVec 32)) (off : Nat) (acc : Acc Payload p) :
    evaluateAcc B p g (accumulated (algebra B g) p pid (liftMemory memory) off acc)=
      accumulate B p g pid memory off (evaluateAcc B p g acc) := by
  conv_lhs => rw [accumulated]
  conv_rhs => rw [accumulate]
  split
  · rw [accumulated_eq,next_eq]
  · rfl
termination_by p.columns-off

def tileFormula (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (pid : Nat) (memory : RegionName → Nat → Cell (BitVec 32)) (i : Fin p.blockRows) : BitVec 32 :=
  (g.tree i).eval B.add (fun j => accumulate B p g pid memory 0 (fun _ => 0) (i,j,PUnit.unit))

/-- Complete original source lowering: all masked loads and casts, every loop
carry, the ordered native reduction graph and FP32 destination store. This
proof is definitional and uses no floating-point arithmetic identity. -/
theorem tileFormula_eq (B : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (pid : Nat) (memory : RegionName → Nat → Cell (BitVec 32)) (i : Fin p.blockRows) :
    tileValue (algebra B g) p pid (liftMemory memory) i=.word (tileFormula B p g pid memory i) := by
  unfold tileValue storedValue outputValues
  change sum B g ⟨1,by simp⟩ Bool.false (accumulated (algebra B g) p pid (liftMemory memory) 0 (fun _ => .word 0))
    (i,PUnit.unit)=_
  simp only [sum,dif_pos rfl]
  unfold sumTile
  change Payload.word ((g.tree i).eval B.add (fun j =>
    evaluateAcc B p g (accumulated (algebra B g) p pid (liftMemory memory) 0 (fun _ => .word 0))
      (i,j,PUnit.unit)))=_
  rw [accumulated_eq]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.WholeNative

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
/-- Local compiler carry primitive, not an IEEE split multiply/add contraction equation. -/
def algebra (native : Scalars) (g : Graph BN BM) : Algebra Payload :=
  { WholeNative.algebra native g with fma := fun _ => some (fun a b c => .carry c a b) }
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
