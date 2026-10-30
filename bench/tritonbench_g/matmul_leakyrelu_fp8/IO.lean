import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.KernelBuffers
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO


namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.IO
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- The quoted FP32 input precision is carried by erased Real cells in the
existing interpreter. The final stored channel has its actual FP16 cell tag. -/
abbrev inputs (g : KernelBuffers.Geometry) : List IOChannel :=
  [⟨.real,[g.M,g.K],"A"⟩,⟨.real,[g.K,g.N],"B"⟩]
abbrev outputs (g : KernelBuffers.Geometry) : List IOChannel :=
  [⟨.fp16,[g.M,g.N],"C"⟩]

def invocation (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (pid : Nat) : Invocation α where
  kernel := Execution.source g.problem (b.views g)
  backend := Execution.backend M (Execution.selectedDot ops)
  pids := fun _ => pid
  setPids := fun ax => decide (ax=0)

/-- Same grouped complete source grid and arbitrary nonnegative input strides.
A/B regions denote their first-element pointers; physical offsets and optional
input storage sharing remain in the unchanged physical bridge. -/
abbrev matmulIO (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) : ProgramIO α α where
  inputs := inputs g
  outputs := outputs g
  scratch := []
  buffers := ["A","B","C"]
  read := fun i _ => match i with
    | ⟨0,_⟩ => fun j => j.1.val*b.A.rowStride+j.2.1.val*b.A.columnStride
    | ⟨1,_⟩ => fun j => j.1.val*b.B.rowStride+j.2.1.val*b.B.columnStride
    | ⟨_+2,h⟩ => False.elim (by simp at h)
  inputMask := fun _ _ _ => True
  write := fun o _ => match o with
    | ⟨0,_⟩ => fun j => j.1.val*b.C.rowStride+j.2.1.val*b.C.columnStride
    | ⟨_+1,h⟩ => False.elim (by simp at h)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => (Source.tiles g.problem).map (invocation M ops g b)
  encode := id
  observe := fun _ _ => id

def matrixA [Inhabited α] (g : KernelBuffers.Geometry) (vals : ChannelValues α (inputs g))
    (i k : Nat) : α :=
  if hi : i<g.M then if hk : k<g.K then vals 0 (⟨i,hi⟩,⟨k,hk⟩,()) else default else default

def matrixB [Inhabited α] (g : KernelBuffers.Geometry) (vals : ChannelValues α (inputs g))
    (k j : Nat) : α :=
  if hk : k<g.K then if hj : j<g.N then vals 1 (⟨k,hk⟩,⟨j,hj⟩,()) else default else default

def result (g : KernelBuffers.Geometry) (f : Nat → Nat → α) : ChannelValues α (outputs g) :=
  fun o => match o with
    | ⟨0,_⟩ => fun j => f j.1.val j.2.1.val
    | ⟨_+1,h⟩ => False.elim (by simp at h)

/-- Independent K8-block recurrence over logical matrix entries. -/
def accumulated (ops : Operations α) (K : Nat) (A B : Nat → Nat → α)
    (i j : Nat) : Nat → α
  | 0 => ops.zero
  | count+1 => block ops
      (fun k => if count*64+k.val<K then A i (count*64+k.val) else ops.zero)
      (fun k => if count*64+k.val<K then B (count*64+k.val) j else ops.zero)
      (accumulated ops K A B i j count)

def native (ops : Operations α) (g : KernelBuffers.Geometry) (A B : Nat → Nat → α)
    (i j : Nat) : α :=
  if g.K=0 then ops.zero else ops.castHalf
    (activate ops g.enabled (accumulated ops g.K A B i j g.problem.kTiles))

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.IO

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Pair
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
variable {α : Type} [Inhabited α]
def candidate (p : Problem) (b : Buffers) := TypedSource.rightSlope "A" "B" "C"
  p.M p.N p.K b.A.rowStride b.A.columnStride b.B.rowStride b.B.columnStride
  b.C.rowStride b.C.columnStride p.enabled

def invocation (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) (pid : Nat) : Invocation α :=
  { IO.invocation M ops g b pid with kernel := candidate g.problem (b.views g) }

def candidateIO (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) : ProgramIO α α :=
  { IO.matmulIO M ops g b with
    schedule := fun _ => (Source.tiles g.problem).map (invocation M ops g b) }

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.Pair

namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.PairIO
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked Execution
noncomputable section
variable {α : Type} [Inhabited α]
def variant (M : Algebra α) (ops : Operations α) (g : KernelBuffers.Geometry)
    (b : KernelBuffers.Buffers) : ProgramIO.Variant (IO.matmulIO M ops g b) where
  schedule := (Pair.candidateIO M ops g b).schedule
  observe := (IO.matmulIO M ops g b).observe

end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.PairIO



namespace VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReaderIO
open VeriTile.Triton FP.Structural
open MatmulLeakyreluFp8 MatmulLeakyreluFp8.Checked
noncomputable section

/-- Both complete finite input matrices; no channel or unused value is discarded. -/
structure Inputs (g : KernelBuffers.Geometry) where
  A : Fin g.M → Fin g.K → ℝ
  B : Fin g.K → Fin g.N → ℝ

def inputs (g : KernelBuffers.Geometry) : ChannelValues ℝ (IO.inputs g) ≃ Inputs g where
  toFun := fun xs => ⟨(fun r t => xs (0 : Fin 2) (r,t,())), (fun t c => xs (1 : Fin 2) (t,c,()))⟩
  invFun := fun xs => Fin.cases (fun j => xs.A j.1 j.2.1)
    (Fin.cases (fun j => xs.B j.1 j.2.1) (fun i => i.elim0))
  left_inv := by
    intro xs; funext i j; fin_cases i <;> rcases j with ⟨a,b,⟨⟩⟩ <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputs (g : KernelBuffers.Geometry) : ChannelValues ℝ (IO.outputs g) ≃
    (Fin g.M → Fin g.N → ℝ) where
  toFun := fun ys r c => ys (0 : Fin 1) (r,c,())
  invFun := fun ys => Fin.cases (fun j => ys j.1 j.2.1) (fun i => i.elim0)
  left_inv := by
    intro ys; funext i j; fin_cases i; rcases j with ⟨a,b,⟨⟩⟩; rfl
  right_inv := fun _ => rfl

/-- Only coordinate equivalences; the original typed IO, complete schedule,
input binding, successful execution, safety and exterior frame are reused. -/
def view (M : Algebra ℝ) (g : KernelBuffers.Geometry) (b : KernelBuffers.Buffers) :
    LogicalProgramIO ℝ ℝ (Inputs g) (Fin g.M → Fin g.N → ℝ) where
  core := IO.matmulIO M realOps g b
  inputs := inputs g
  outputs := outputs g
end
end VeriTile.Bench.Repairs.MatmulLeakyreluFp8ScalarCast.ReaderIO
