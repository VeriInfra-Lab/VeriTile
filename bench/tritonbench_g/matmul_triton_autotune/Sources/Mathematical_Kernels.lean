import bench.tritonbench_g.matmul_triton_autotune.Sources.Mathematical_FPComplete
import Mathlib.Data.Fintype.EquivFin

/-! Original typed source, exact host choices and independent matrix reference.
The reference arrays are pinned to raw storage, not unresolved Torch view values.
Unspecified input precision is FP32; accumulation remains live-carry dotAcc and
output is explicitly FP16. Numerical primitive laws never assert execution. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange)
noncomputable section

inductive Config where | cuda128 | cuda64 | hip128
  deriving DecidableEq, Repr

def Config.bm : Config → Nat | .cuda128 => 128 | .cuda64 => 64 | .hip128 => 128
def Config.bn (_ : Config) : Nat := 256
def Config.bk : Config → Nat | .cuda128 => 64 | .cuda64 => 32 | .hip128 => 16
def Config.gm : Config → Nat | .cuda128 | .cuda64 => 8 | .hip128 => 1
def Config.stages : Config → Nat | .cuda128 => 3 | .cuda64 => 4 | .hip128 => 2
def Config.warps : Config → Nat | .cuda128 => 8 | .cuda64 | .hip128 => 4
def Config.waves : Config → Option Nat | .hip128 => some 2 | _ => none

def configurations (backend : String) : List Config :=
  if backend == "cuda" then [.cuda128,.cuda64] else [.hip128]

theorem Config.positive (c : Config) : 0<c.bm ∧ 0<c.bn ∧ 0<c.bk ∧ 0<c.gm := by
  cases c <;> decide

structure Request where
  (A B C : RegionName)
  (rows inner innerB columns : Nat)
  (strideAM strideAK strideBK strideBN : Nat)
  aContiguous : Bool
  activation : String
  target : String
  config : Config

def Request.parameters (r : Request) : FPValue.Parameters where
  A := r.A; B := r.B; C := r.C
  M := r.rows; N := r.columns; K := r.inner
  stride_am := r.strideAM; stride_ak := r.strideAK
  stride_bk := r.strideBK; stride_bn := r.strideBN
  stride_cm := max r.columns 1; stride_cn := 1
  BLOCK_SIZE_M := r.config.bm; BLOCK_SIZE_N := r.config.bn; BLOCK_SIZE_K := r.config.bk
  GROUP_SIZE_M := r.config.gm
  numKBlocks := (r.inner+r.config.bk-1)/r.config.bk
  ACTIVATION := r.activation == "leaky_relu"

def kernel (r : Request) : ComputeKernel := FPValue.source r.parameters .single

def tileRows (r : Request) := (r.rows+r.config.bm-1)/r.config.bm
def tileColumns (r : Request) := (r.columns+r.config.bn-1)/r.config.bn
def programs (r : Request) := tileRows r * tileColumns r

def instanceState {α : Type} (r : Request) (pid : Nat) (s : State α) : State α :=
  {s with pids := fun ax => if ax=0 then pid else 0
          numPids := fun ax => if ax=0 then programs r else 1}

def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) : List Nat → State α → Option (State α)
  | [], s => some s
  | pid::rest, s =>
      (FP.Structural.exec (FPValue.backend M D) (kernel r) (instanceState r pid s)).bind
        (runPrograms M D r rest)

inductive HostError where
  | incompatibleDimensions | noncontiguousA | executionFailure
  deriving DecidableEq

/-- Rank-two metadata and successful fresh allocation are API contracts.
The source assertions run in order; the returned region is the allocated C. -/
def host {α : Type} [Inhabited α] (M : Algebra α) (D : FPValue.DotAcc α)
    (r : Request) (s : State α) : Except HostError (RegionName × State α) :=
  if r.inner ≠ r.innerB then .error .incompatibleDimensions
  else if r.aContiguous = Bool.false then .error .noncontiguousA
  else match runPrograms M D r (List.range (programs r)) s with
    | none => .error .executionFailure
    | some final => .ok (r.C,final)

/-- Each configuration retains its own reduction primitive, with live carry.
No reassociation, separate dot/add, or final-only rounding is implied. -/
structure Numerics (α : Type) where
  zero : α
  dotAcc : (k : Nat) → (Fin k → α) → (Fin k → α) → α → α
  nonnegative : α → Bool
  slopeMul : α → α
  castHalf : α → α

def Numerics.activate {α : Type} (Q : Numerics α) (enabled : Bool) (x : α) : α :=
  if enabled then if Q.nonnegative x then x else Q.slopeMul x else x

/-- Local numeric correspondence only: DOTACC_PROFILE, ACTIVATION_PROFILE,
CONVERSIONS_AND_SPECIALS from the historical obligation registry. -/
structure PrimitiveLaws {α : Type} (M : Algebra α) (D : FPValue.DotAcc α) (Q : Numerics α) : Prop where
  initialZero : M.literal (some .fp32) .real 0 = Q.zero
  loadZero : M.cast none .real .fp32 (M.literal none .real 0.0) = Q.zero
  dotAcc : ∀ (m k n : Nat) (a : TileIndex [m,k] → α) (b : TileIndex [k,n] → α)
      (acc : TileIndex [m,n] → α) (i : TileIndex [m,n]),
    D.apply (some .fp32) .fp32 .fp32 .real {} (batch := []) a b acc i =
      Q.dotAcc k (fun j => a (i.1,j,PUnit.unit)) (fun j => b (j,i.2.1,PUnit.unit)) (acc i)
  comparison : ∀ x, D.le (some .fp32) .real (M.literal (some .fp32) .real 0.0) x = Q.nonnegative x
  slope : ∀ x, M.binary (some .fp32) .real .mul
    (M.literal (some .fp32) .real 0.01) x = Q.slopeMul x
  halfCast : ∀ x, M.cast none .real .fp16 x = Q.castHalf x

def inputA {α : Type} [Inhabited α] (r : Request) (s : State α) (row k : Nat) : α :=
  (s.mem r.A (row*r.strideAM+k*r.strideAK)).read .fp32
def inputB {α : Type} [Inhabited α] (r : Request) (s : State α) (k col : Nat) : α :=
  (s.mem r.B (k*r.strideBK+col*r.strideBN)).read .fp32

/-- Independent logical matrix recurrence, including every zero-padded lane. -/
def blockStep {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col block : Nat) (acc : α) : α :=
  Q.dotAcc r.config.bk
    (fun k => if block*r.config.bk+k.val<r.inner then inputA r s row (block*r.config.bk+k.val) else Q.zero)
    (fun k => if block*r.config.bk+k.val<r.inner then inputB r s (block*r.config.bk+k.val) col else Q.zero) acc

def accumulator {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) : α :=
  foldRange (fun block x => blockStep Q r s row col block x)
    0 ((r.inner+r.config.bk-1)/r.config.bk) 1 Q.zero

def reference {α : Type} [Inhabited α] (Q : Numerics α) (r : Request)
    (s : State α) (row col : Nat) : α :=
  Q.castHalf (Q.activate (r.activation == "leaky_relu") (accumulator Q r s row col))

/-- Independent ideal target; the source's FP32 and cast behavior is separate. -/
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
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Whole
