import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_SourceDefinitions
import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_CandidateSourceDefinitions
import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_FPComplete
import bench.tritonbench_g.matmul_leakyrelu.Sources.Mathematical_FPFusedSource
import Mathlib.Data.Fintype.EquivFin

/-! Shared original source and independent whole-matrix specifications.
Default input is FP32; the source's accumulator and FP16 output are retained.
The primitive interface describes floating operations, never a kernel result.
-/
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section

structure Request where
  (A B C : RegionName)
  (rows inner innerB columns : Nat)
  (strideAM strideAK strideBK strideBN : Nat)
  aContiguous : Bool
  activation : String

def Request.parameters (r : Request) : FPValue.Parameters where
  A := r.A; B := r.B; C := r.C
  M := r.rows; N := r.columns; K := r.inner
  stride_am := r.strideAM; stride_ak := r.strideAK
  stride_bk := r.strideBK; stride_bn := r.strideBN
  stride_cm := max r.columns 1; stride_cn := 1
  BLOCK_SIZE_M := 32; BLOCK_SIZE_N := 32; BLOCK_SIZE_K := 32
  GROUP_SIZE_M := 4
  ACTIVATION := r.activation == "leaky_relu"

/-- No source transformation: both constexpr activation cases are reused. -/
def kernel (r : Request) : ComputeKernel := FPValue.source r.parameters .single

def Request.contractedParameters (r : Request) : FPFused.Parameters where
  A := r.A; B := r.B; C := r.C
  M := r.rows; N := r.columns; K := r.inner
  stride_am := r.strideAM; stride_ak := r.strideAK
  stride_bk := r.strideBK; stride_bn := r.strideBN
  stride_cm := max r.columns 1; stride_cn := 1
  BLOCK_SIZE_M := 32; BLOCK_SIZE_N := 32; BLOCK_SIZE_K := 32
  GROUP_SIZE_M := 4
  ACTIVATION := r.activation == "leaky_relu"

/-- The observed compiler profile, kept distinct from `kernel`. Its dot takes
the live accumulator and its slope is on the right. No floating equality of
the two graphs is assumed; both use the same source geometry and host metadata. -/
def contractedKernel (r : Request) : ComputeKernel :=
  FPFused.source r.contractedParameters .single
def tileRows (r : Request) := (r.rows + 31) / 32
def tileColumns (r : Request) := (r.columns + 31) / 32
def programs (r : Request) := tileRows r * tileColumns r

def instanceState {α : Type} (r : Request) (pid : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then pid else 0
          numPids := fun ax => if ax=0 then programs r else 1}

/-- Each program runs the actual shared source on the preceding memory. -/
def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.Dot α)
    (r : Request) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid::rest, s =>
      (FP.Structural.exec (FPValue.backend M D) (kernel r) (instanceState r pid s)).bind
        (runPrograms M D r rest)

inductive HostError where
  | incompatibleDimensions | noncontiguousA | executionFailure
  deriving DecidableEq

/-- Rank-two tensors and a successful fresh allocation are ordinary API
contracts. The two explicit source assertions retain their original order. -/
def host {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.Dot α)
    (r : Request) (s : State α) : Except HostError (RegionName × State α) :=
  if r.inner ≠ r.innerB then .error .incompatibleDimensions
  else if r.aContiguous = Bool.false then .error .noncontiguousA
  else match runPrograms M D r (List.range (programs r)) s with
    | none => .error .executionFailure
    | some final => .ok (r.C,final)

/-- A block dot is a primitive on one logical row and one logical column.
Its actual rounding/reduction policy is retained, not replaced by Real sum. -/
structure Numerics (α : Type) where
  zero : α
  add : α → α → α
  dot : (Fin 32 → α) → (Fin 32 → α) → α
  nonnegative : α → Bool
  slopeMul : α → α
  castHalf : α → α

def Numerics.activate {α : Type} (Q : Numerics α) (enabled : Bool) (x : α) : α :=
  if enabled then if Q.nonnegative x then x else Q.slopeMul x else x

/-- Only scalar/dot primitive correspondence, with the source's exact tags.
No memory, loop, host, grid, or complete output appears in these premises. -/
structure PrimitiveLaws {α : Type} (M : Algebra α) (D : FPValue.Dot α) (Q : Numerics α) : Prop where
  initialZero : M.literal (some .fp32) .real 0 = Q.zero
  loadZero : M.cast none .real .fp32 (M.literal none .real 0.0) = Q.zero
  add : ∀ x y, M.binary (some .fp32) .real .add x
    (M.cast (some .fp32) .fp32 .real y) = Q.add x y
  dot : ∀ (a b : TileIndex [32,32] → α) (i : TileIndex [32,32]),
    D.apply (some .fp32) .fp32 .fp32 .fp32 {} (batch := []) a b i =
      Q.dot (fun k => a (i.1,k,PUnit.unit)) (fun k => b (k,i.2.1,PUnit.unit))
  comparison : ∀ x, D.le (some .fp32) .real (M.literal (some .fp32) .real 0.0) x = Q.nonnegative x
  slope : ∀ x, M.binary (some .fp32) .real .mul
    (M.literal (some .fp32) .real 0.01) x = Q.slopeMul x
  halfCast : ∀ x, M.cast none .real .fp16 x = Q.castHalf x

def inputA {α : Type} [Inhabited α] (r : Request) (s : State α) (row k : Nat) : α :=
  (s.mem r.A (row*r.strideAM+k*r.strideAK)).read .fp32
def inputB {α : Type} [Inhabited α] (r : Request) (s : State α) (k col : Nat) : α :=
  (s.mem r.B (k*r.strideBK+col*r.strideBN)).read .fp32

/-- Independent logical matrix recurrence; no source execution, PID, tensor
registers, tile masks, or output memory occur in this definition. -/
def blockValue {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col block : Nat) : α :=
  Q.dot (fun k => if block*32+k.val<r.inner then inputA r s row (block*32+k.val) else Q.zero)
    (fun k => if block*32+k.val<r.inner then inputB r s (block*32+k.val) col else Q.zero)

def accumulator {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) : α :=
  foldRange (fun block x => Q.add x (blockValue Q r s row col block))
    0 ((r.inner+31)/32) 1 Q.zero

def reference {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) : α :=
  Q.castHalf (Q.activate (r.activation == "leaky_relu") (accumulator Q r s row col))

/-- Independent ideal mathematical target, separate from floating evaluation. -/
def mathematical (r : Request) (A : Fin r.rows → Fin r.inner → ℝ)
    (B : Fin r.inner → Fin r.columns → ℝ) (i : Fin r.rows) (j : Fin r.columns) : ℝ :=
  let value := ∑ k : Fin r.inner, A i k * B k j
  if r.activation == "leaky_relu" then if 0 ≤ value then value else (1/100 : ℝ)*value else value

def outputOffset (r : Request) (row col : Nat) := max r.columns 1 * row + col
def Footprint (r : Request) (region : RegionName) (offset : Nat) : Prop :=
  region=r.C ∧ ∃ row<r.rows, ∃ col<r.columns, offset=outputOffset r row col

def Result {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (initial final : State α) : Prop :=
  (∀ row<r.rows, ∀ col<r.columns,
    final.mem r.C (outputOffset r row col)=Cell.mk .fp16 (reference Q r initial row col)) ∧
  (∀ region offset, ¬ Footprint r region offset → final.mem region offset=initial.mem region offset)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.Whole
