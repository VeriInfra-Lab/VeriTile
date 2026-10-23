import VeriTile.Triton.Memory.RegionalGroupedKernelIO
import VeriTile.Triton.Memory.LogicalIO
import bench.repairs.tritonbench_g.fused_activation.Proofs.Projection
import VeriTile.Triton
import bench.repairs.tritonbench_g.fused_activation.Proofs.Graph
import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.GuardedIO

/-! Public typed IO data for the existing native activation pair. The source
reads x, cyclic bias and input before updating x; both variants retain the
selected FP32 source tags and the same native FMA interpretation. -/
namespace VeriTile.Bench.Repairs.FusedActivation.NativeIO
open VeriTile.Triton FP.Structural
noncomputable section

def inputs (p : Parameters) : List IOChannel :=
  [⟨.real, [p.block], p.output⟩, ⟨.real, [p.block], p.y⟩, ⟨.real, [p.block], p.x⟩]
def outputs (p : Parameters) : List IOChannel := [⟨.real, [p.block], p.output⟩]
abbrev Inputs (p : Parameters) := ChannelValues Graph.Word (inputs p)

def inputLane (p : Parameters) : (i : Fin (inputs p).length) →
    TileIndex (inputs p)[i].shape → Fin p.block
  | ⟨0, _⟩ => fun j => j.1
  | ⟨1, _⟩ => fun j => j.1
  | ⟨2, _⟩ => fun j => j.1
  | ⟨i + 3, h⟩ => False.elim (by simp [inputs] at h)
def outputLane (p : Parameters) : (o : Fin (outputs p).length) →
    TileIndex (outputs p)[o].shape → Fin p.block
  | ⟨0, _⟩ => fun j => j.1
  | ⟨i + 1, h⟩ => False.elim (by simp [outputs] at h)

def memory (p : Parameters) (W : Nat) (launch : Nat → Nat) : MemoryIO Graph.Word where
  inputs := inputs p
  outputs := outputs p
  scratch := []
  buffers := [p.output, p.y, p.x]
  read := fun channel _ j =>
    let offset := launch 0 * p.block + (inputLane p channel j).val
    if channel.val = 1 then offset % W else offset
  inputMask := fun channel _ j => launch 0 * p.block + (inputLane p channel j).val < p.n
  write := fun channel _ j => launch 0 * p.block + (outputLane p channel j).val
  outputMask := fun channel _ j => launch 0 * p.block + (outputLane p channel j).val < p.n
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def invocation (rest : Algebra Graph.Expr) (p : Parameters) (W : Nat)
    (a : Bool) (launch : Nat → Nat) : Invocation Graph.Expr where
  kernel := candidateKernel p W (1/2) a
  backend := Graph.algebra rest
  pids := launch

def original (rest : Algebra Graph.Expr) (ops : Graph.Operations)
    (p : Parameters) (W : Nat) (a : Bool) (launch : Nat → Nat) : ProgramIO Graph.Word Graph.Expr :=
  { memory p W launch with
    schedule := fun _ => [invocation rest p W a launch]
    encode := .leaf
    observe := fun _ _ => Graph.eval ops }

def commuted (rest : Algebra Graph.Expr) (ops : Graph.Operations)
    (p : Parameters) (W : Nat) (a : Bool) (launch : Nat → Nat) :
    ProgramIO.Variant (original rest ops p W a launch) where
  schedule := fun _ => [{ invocation rest p W a launch with
    kernel := commutedKernel p.output p.y p.x W p.n p.block (1/2) a }]
  observe := fun _ _ => Graph.eval ops

def x {p : Parameters} (values : Inputs p) (j : Fin p.block) : Graph.Word :=
  values ⟨0, by simp [inputs]⟩ (j, PUnit.unit)
def bias {p : Parameters} (values : Inputs p) (j : Fin p.block) : Graph.Word :=
  values ⟨1, by simp [inputs]⟩ (j, PUnit.unit)
def input {p : Parameters} (values : Inputs p) (j : Fin p.block) : Graph.Word :=
  values ⟨2, by simp [inputs]⟩ (j, PUnit.unit)

/-- Exactly the existing outer-add rewrite domain, after the common native
FMA. It is independent of execution and is not a Python input guard. -/
def domain (ops : Graph.Operations) (D : FP.Guarded.Domain Graph.Word)
    (p : Parameters) (launch : Nat → Nat) (values : Inputs p) : Prop :=
  ∀ j : Fin p.block, launch 0 * p.block + j.val < p.n →
    D .finite (ops.fmaRN32 (input values j) Graph.half (x values j)) ∧
    D .finite (bias values j)

end
end VeriTile.Bench.Repairs.FusedActivation.NativeIO

/-! Public grouped mathematical IO data, shared by the primary regional
specification and its optional flat corollary. -/
namespace VeriTile.Bench.Repairs.FusedActivation.RealIO
open VeriTile.Triton
noncomputable section

/-- Independent scalar formula, with the wrapper's fixed multiplier. Both
sigmoid and the benchmark's direct ReLU branch are covered. -/
def formula (activation : Bool) (x bias input : ℝ) : ℝ :=
  let z := (1/2) * input + x + bias
  if activation then Real.sigmoid z else max 0 z

/-- Three entry-state reads, cyclic bias indexing, and an in-place x write. -/
def activationIO (x b input : RegionName) (W N B : Nat) (a : Bool) : GroupedMasked2DKernelIO where
  kernel := kernel x b input W N B (1/2) a
  projection := Projection.original_projection x b input W N B (1/2) a
  nIn := 3
  nOut := 1
  bufs := [x, b, input]
  inp := fun i => match i with
    | ⟨0, _⟩ => x
    | ⟨1, _⟩ => b
    | ⟨_ + 2, _⟩ => input
  out := fun _ => x
  B := B
  read := fun i pid _ j => match i with
    | ⟨0, _⟩ => pid * B + j.val
    | ⟨1, _⟩ => (pid * B + j.val) % W
    | ⟨_ + 2, _⟩ => pid * B + j.val
  readMask := fun _ pid _ j => pid * B + j.val < N
  write := fun _ pid _ j => pid * B + j.val
  writeMask := fun _ pid _ j => pid * B + j.val < N

end
end VeriTile.Bench.Repairs.FusedActivation.RealIO

namespace VeriTile.Bench.Repairs.FusedActivation.RealIO
open VeriTile.Triton
noncomputable section

/-- The two program IDs and three independent entry-state windows. -/
structure ActivationValues (B : Nat) where
  pid0 : Nat
  pid1 : Nat
  x : Fin B → ℝ
  bias : Fin B → ℝ
  input : Fin B → ℝ

def inputCoordinates (B : Nat) :
    (Nat × Nat × (Fin 3 → Fin B → ℝ)) ≃ ActivationValues B where
  toFun := fun xs => ⟨xs.1, xs.2.1, xs.2.2 0, xs.2.2 1, xs.2.2 2⟩
  invFun := fun xs => (xs.pid0, xs.pid1,
    Fin.cases xs.x (Fin.cases xs.bias (Fin.cases xs.input (fun i => Fin.elim0 i))))
  left_inv := by
    rintro ⟨p0, p1, xs⟩
    apply congrArg (fun ys : Fin 3 → Fin B → ℝ => (p0, p1, ys))
    funext i
    fin_cases i <;> rfl
  right_inv := by intro xs; cases xs; rfl

def outputCoordinates (B : Nat) : (Fin 1 → Fin B → ℝ) ≃ (Fin B → ℝ) where
  toFun := fun xs => xs 0
  invFun := fun xs _ => xs
  left_inv := by intro xs; funext i; fin_cases i; rfl
  right_inv := fun _ => rfl

/-- Coordinate transport of the full regional judgment, including both PIDs. -/
def logicalActivationIO (x b input : RegionName) (W N B : Nat) (a : Bool) :
    LogicalIO (Nat × Nat × (Fin 3 → Fin B → ℝ)) (Fin 1 → Fin B → ℝ)
      (ActivationValues B) (Fin B → ℝ)
      (fun f => (activationIO x b input W N B a).RegionalImplements
        (fun p0 p1 xs => f (p0, p1, xs))) where
  inputs := inputCoordinates B
  outputs := outputCoordinates B
end
end VeriTile.Bench.Repairs.FusedActivation.RealIO
