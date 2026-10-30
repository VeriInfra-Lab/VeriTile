import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelObserved
import VeriTile.Triton.Float.ProgramIO
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.KernelRealTarget
import VeriTile.Triton.Float.LogicalProgramIO

/-! Complete logical tensor windows and the concrete launch schedule of the
selected runtime-clamp source. This is kernel IO; Tensor metadata is separate. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface
open VeriTile.Triton FP.Structural
open Kernel Kernel.Scalar Kernel.Math
noncomputable section

def inputs (c : Geometry) : List IOChannel :=
  [⟨.fp32, [c.batches, c.sequence, c.heads], "dt"⟩,
   ⟨.fp32, [c.heads], "rate"⟩, ⟨.real, [c.heads], "bias"⟩]

def outputs (c : Geometry) : List IOChannel :=
  [⟨.real, [c.batches, c.heads, c.chunks, c.chunk], "out"⟩,
   ⟨.real, [c.batches, c.heads, c.chunks, c.chunk], "sum"⟩]

abbrev Inputs (α : Type) (c : Geometry) := ChannelValues α (inputs c)
abbrev Outputs (α : Type) (c : Geometry) := ChannelValues α (outputs c)

def inputAddress (c : Geometry) : (i : Fin (inputs c).length) → TileIndex (inputs c)[i].shape → Nat
  | ⟨0, _⟩ => fun j => j.1.val * c.dtBatch + j.2.1.val * c.dtTime + j.2.2.1.val * c.dtHead
  | ⟨1, _⟩ => fun j => j.1.val * c.rateHead
  | ⟨2, _⟩ => fun j => j.1.val * c.biasHead
  | ⟨i + 3, h⟩ => False.elim (by simp [inputs] at h)

def inputMask (c : Geometry) (hasBias : Bool) : (i : Fin (inputs c).length) →
    TileIndex (inputs c)[i].shape → Prop
  | ⟨0, _⟩ | ⟨1, _⟩ => fun _ => True
  | ⟨2, _⟩ => fun _ => hasBias = Bool.true
  | ⟨i + 3, h⟩ => False.elim (by simp [inputs] at h)

def outputAddress (c : Geometry) : (o : Fin (outputs c).length) → TileIndex (outputs c)[o].shape → Nat
  | ⟨0, _⟩ | ⟨1, _⟩ => fun j => packed c j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
  | ⟨i + 2, h⟩ => False.elim (by simp [outputs] at h)

def memory (α : Type) (c : Geometry) (hasBias : Bool) : MemoryIO α where
  inputs := inputs c
  outputs := outputs c
  scratch := []
  buffers := ["dt", "rate", "bias", "out", "sum"]
  read := fun i _ j => inputAddress c i j
  inputMask := fun i _ j => inputMask c hasBias i j
  write := fun o _ j => outputAddress c o j
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def invocation {α : Type} (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool)
    (lower upper : α) (l : Launch) : Invocation (Term α) where
  kernel := kernel c hasBias useSoftplus
  backend := engine (graphAlgebra M c plan)
  arguments := [⟨"dt_min", .real, [], fun _ => .input lower⟩,
    ⟨"dt_max", .real, [], fun _ => .input upper⟩]
  pids := fun ax => if ax = 0 then l.batch else if ax = 1 then l.chunk else if ax = 2 then l.head else 0

def programIO {α : Type} (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α) :
    ProgramIO α (Term α) :=
  { memory α c hasBias with
    schedule := fun _ => c.grid.map (invocation M c plan hasBias useSoftplus lower upper)
    encode := Term.input
    observe := fun _ _ => interpret M }

/-- Extend finite logical input arrays outside their tensor domain only to give
the existing independent formulas their total mathematical function type. -/
def logicalInputs {α : Type} [Inhabited α] (c : Geometry) (values : Inputs α c) : LogicalInputs α where
  dt := fun b t h => if hb : b < c.batches then if ht : t < c.sequence then
    if hh : h < c.heads then values ⟨0, by simp [inputs]⟩ (⟨b,hb⟩,⟨t,ht⟩,⟨h,hh⟩,PUnit.unit)
    else default else default else default
  rate := fun h => if hh : h < c.heads then values ⟨1, by simp [inputs]⟩ (⟨h,hh⟩,PUnit.unit) else default
  bias := fun h => if hh : h < c.heads then values ⟨2, by simp [inputs]⟩ (⟨h,hh⟩,PUnit.unit) else default

def result {α : Type} (c : Geometry)
    (prepared cumulative : Nat → Nat → Nat → Nat → α) : Outputs α c
  | ⟨0, _⟩ => fun j => prepared j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
  | ⟨1, _⟩ => fun j => cumulative j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val
  | ⟨i + 2, h⟩ => False.elim (by simp [outputs] at h)

/-- Independent ordered native preparation and inclusive prefix recurrence. -/
def nativeTarget {α : Type} [Inhabited α] (M : NativeModel α) (c : Geometry)
    (plan : PrefixPlan c.BH c.BC) (hasBias useSoftplus : Bool) (lower upper : α)
    (values : Inputs α c) : Outputs α c :=
  result c (preparedValue M c (logicalInputs c values) hasBias useSoftplus lower upper)
    (fun b h k j => if hj : j < c.BC then
      cumulativeValue M c plan (logicalInputs c values) hasBias useSoftplus lower upper b h k ⟨j,hj⟩
      else default)

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Interface

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.LogicalTensors
open VeriTile Triton FP.Structural Kernel Kernel.Scalar
noncomputable section

/- Every finite tensor, including inactive optional channels, has a true
coordinate inverse. The original typed memory/schedule/observer is reused. -/
structure Inputs (c : Geometry) where
  dt : Fin (c.batches) → Fin (c.sequence) → Fin (c.heads) → ℝ
  rate : Fin (c.heads) → ℝ
  bias : Fin (c.heads) → ℝ

structure Outputs (c : Geometry) where
  prepared : Fin (c.batches) → Fin (c.heads) → Fin (c.chunks) → Fin (c.chunk) → ℝ
  cumulative : Fin (c.batches) → Fin (c.heads) → Fin (c.chunks) → Fin (c.chunk) → ℝ

def inputs (c : Geometry) : Interface.Inputs ℝ c ≃ Inputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .fp32 [c.batches,c.sequence,c.heads] (values (0 : Fin 3)),
    Values.curryEquiv ℝ .fp32 [c.heads] (values (1 : Fin 3)),
    Values.curryEquiv ℝ .real [c.heads] (values (2 : Fin 3))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .fp32 [c.batches,c.sequence,c.heads]).symm values.dt
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .fp32 [c.heads]).symm values.rate
    | ⟨2,_⟩ => (Values.curryEquiv ℝ .real [c.heads]).symm values.bias
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def outputs (c : Geometry) : Interface.Outputs ℝ c ≃ Outputs c where
  toFun := fun values => ⟨Values.curryEquiv ℝ .real [c.batches,c.heads,c.chunks,c.chunk] (values (0 : Fin 2)),
    Values.curryEquiv ℝ .real [c.batches,c.heads,c.chunks,c.chunk] (values (1 : Fin 2))⟩
  invFun := fun values channel => match channel with
    | ⟨0,_⟩ => (Values.curryEquiv ℝ .real [c.batches,c.heads,c.chunks,c.chunk]).symm values.prepared
    | ⟨1,_⟩ => (Values.curryEquiv ℝ .real [c.batches,c.heads,c.chunks,c.chunk]).symm values.cumulative
  left_inv := by
    intro values
    funext channel j
    fin_cases channel <;> exact congrFun (Equiv.symm_apply_apply _ _) j
  right_inv := by
    intro values
    cases values
    dsimp
    congr 1 <;> apply Equiv.apply_symm_apply

def inputArrays (c : Geometry) (values : Inputs c) :=
  Interface.logicalInputs c ((inputs c).symm values)

def io (c : Geometry) (plan : PrefixPlan c.BH c.BC) (hasBias reverseBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : LogicalProgramIO ℝ (Term ℝ) (Inputs c) (Outputs c) where
  core := Interface.programIO (RealTarget.model reverseBias upper) c plan hasBias useSoftplus lower (RealTarget.upperWord upper)
  inputs := inputs c
  outputs := outputs c

end
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.LogicalTensors
