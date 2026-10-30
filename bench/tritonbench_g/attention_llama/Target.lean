import bench.tritonbench_g.attention_llama.Proofs.ScalarABI
import bench.tritonbench_g.attention_llama.Proofs.NativeSchedule

/-! Independent native precision target. The target consumes logical input
words and a raw ABI specialization; it never reads a source execution state.
All scalar arithmetic, conversions, exponentials and fixed K16 tensor-core
operations are ordinary primitive semantics. No finite-output hypothesis is
required: signed zero, underflow, overflow, infinities and NaNs remain in α. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Target
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionLlama NativeSchedule
noncomputable section

abbrev Ops := NativeGraph.Ops

structure Inputs (α : Type) (D : Nat) where
  query : Fin D → α
  halfKey : Nat → Fin D → α
  halfValue : Nat → Fin D → α
  byteKey : Nat → Fin D → UInt8
  byteValue : Nat → Fin D → UInt8

abbrev State := NativeGraph.State

def scaleProduct (A : Ops α) (C : ScalarABI.Conversions α) (arg : ScalarABI.Argument) (x : α) : α :=
  match arg with
  | .one => x
  | _ => A.mul (ScalarABI.convert C arg) x

def key (A : Ops α) (input : Inputs α D) (fp8 : Bool) (N k : Nat) (d : Fin D) : α :=
  if fp8 then A.e5m2ToHalf (if k<N then input.byteKey k d else 0)
  else if k<N then input.halfKey k d else A.zero

def value (A : Ops α) (input : Inputs α D) (fp8 : Bool) (N k : Nat) (d : Fin D) : α :=
  if fp8 then A.e5m2ToHalf (if k<N then input.byteValue k d else 0)
  else if k<N then input.halfValue k d else A.zero

def score (A : Ops α) (C : ScalarABI.Conversions α) (input : Inputs α D)
    (arg : ScalarABI.Argument) (fp8 causal : Bool) (N position query block : Nat) (col : Fin 64) : α :=
  let k := block*64+col.val
  let dot := mmaFold A.mma16 (mmaCoordinate fp8 D)
    (NativeGraph.padded A.zero input.query)
    (NativeGraph.padded A.zero (key A input fp8 N k)) A.zero (D/16)
  let masked := if k<N then scaleProduct A C arg dot else A.negativeInfinity
  if causal then if k+position≤query then masked else A.negativeInfinity else masked

def step (A : Ops α) (C : ScalarABI.Conversions α) (input : Inputs α D)
    (arg : ScalarABI.Argument) (fp8 causal : Bool) (N position query block : Nat)
    (state : State α D) : State α D :=
  let qk := score A C input arg fp8 causal N position query block
  let m := fun l => A.maximum ((selected D l).eval A.maximum qk) (state.maximum l)
  let alpha := fun l => NativeGraph.exp A (A.sub (state.maximum l) (m l))
  let previous := fun l => A.mul (state.denominator l) (alpha l)
  let probability := fun col : Fin 64 => NativeGraph.exp A (A.sub (qk col) (m (lane col.val)))
  let current := fun l => A.fma (state.denominator l) (alpha l) ((selected D l).eval A.add probability)
  let reciprocal := fun l => A.div A.one (current l)
  let halfProbability := fun col : Fin 64 => A.half (A.mul (probability col) (reciprocal (lane col.val)))
  { maximum := m
    denominator := current
    accumulator := fun d => mmaFold A.mma16 (mmaCoordinate fp8 64)
      (fun k => halfProbability (column k))
      (fun k => value A input fp8 N (block*64+(column k).val) d)
      (A.mul (state.accumulator d) (A.mul (previous (lane d.val)) (reciprocal (lane d.val)))) 4 }

def run (A : Ops α) (C : ScalarABI.Conversions α) (input : Inputs α D)
    (arg : ScalarABI.Argument) (fp8 causal : Bool) (N position query : Nat) : Nat → State α D
  | 0 => NativeGraph.initial A D
  | n+1 => step A C input arg fp8 causal N position query n
      (run A C input arg fp8 causal N position query n)

def output (A : Ops α) (C : ScalarABI.Conversions α) (input : Inputs α D)
    (arg : ScalarABI.Argument) (fp8 causal : Bool) (N position pid row : Nat) (d : Fin D) : α :=
  A.half ((run A C input arg fp8 causal N position (pid*64+row)
    (MaskCoverage.count (MaskCoverage.stop N causal position pid))).accumulator d)

#specNonCircular output avoiding [FP.Structural.exec,FP.Structural.run,NativeGraph.run,NativeGraph.returnedWord]
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Target
