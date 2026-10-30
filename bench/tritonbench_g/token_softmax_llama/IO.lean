import Mathlib.Tactic.FinCases
import bench.tritonbench_g.token_softmax_bloom.IO
import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.token_softmax_llama.Proofs.KernelSource

/-! Llama uses the same quoted strided source and public logical channels as
Bloom: two natural metadata channels and FP32 values for every selected PID,
with the actual head/token strides, masks and complete schedule. The shared
layout is defined in Bloom/IO.lean; source_is_shared proves source identity. -/
namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.PublicIO
open VeriTile.Triton FP.Structural Strided
noncomputable section

def realIO (p : Parameters) (work : List Grid.PID) : ProgramIO ℝ Real.Scalar :=
  TokenSoftmaxBloom.SoftmaxIO.realIO p work

def nativeIO (I : Native.Instructions) (V : Nat) (p : Parameters) (work : List Grid.PID) :
    ProgramIO Native.Word Native.Node := TokenSoftmaxBloom.SoftmaxIO.nativeIO I V p work

/-- Metadata and values of every selected segment/head event, in schedule order. -/
structure LogicalInputs (p : Parameters) (work : List Grid.PID) where
  starts : Fin work.length → Nat
  lengths : Fin work.length → Nat
  values : Fin work.length → Fin p.width → ℝ

def logicalInputs (p : Parameters) (work : List Grid.PID) :
    (realIO p work).Inputs ≃ LogicalInputs p work where
  toFun xs := ⟨fun r => xs (0 : Fin 3) (r,()),fun r => xs (1 : Fin 3) (r,()),fun r c => xs (2 : Fin 3) (r,c,())⟩
  invFun xs := fun port => match port with
    | ⟨0,_⟩ => fun i => xs.starts i.1
    | ⟨1,_⟩ => fun i => xs.lengths i.1
    | ⟨2,_⟩ => fun i => xs.values i.1 i.2.1
  left_inv := by
    intro xs; funext port i; fin_cases port
    · rcases i with ⟨r,u⟩; cases u; rfl
    · rcases i with ⟨r,u⟩; cases u; rfl
    · rcases i with ⟨r,c,u⟩; cases u; rfl
  right_inv := by intro xs; cases xs; rfl

def logicalOutputs (p : Parameters) (work : List Grid.PID) :
    (realIO p work).Outputs ≃ (Fin work.length → Fin p.width → ℝ) where
  toFun xs r c := xs (0 : Fin 1) (r,c,())
  invFun xs := Fin.cases (fun i => xs i.1 i.2.1) (fun i => Fin.elim0 i)
  left_inv := by intro xs; funext port i; fin_cases port; rcases i with ⟨r,c,u⟩; cases u; rfl
  right_inv := by intro xs; rfl

def logicalRealIO (p : Parameters) (work : List Grid.PID) :
    LogicalProgramIO ℝ Real.Scalar (LogicalInputs p work) (Fin work.length → Fin p.width → ℝ) where
  core := realIO p work
  inputs := logicalInputs p work
  outputs := logicalOutputs p work

/-- Same full coverage and output-event uniqueness, now over named metadata. -/
def logicalDomain (p : Parameters) (work : List Grid.PID) (xs : LogicalInputs p work) : Prop :=
  TokenSoftmaxBloom.SoftmaxIO.realDomain p work ((logicalInputs p work).symm xs)

/-- The finite active-segment accessor. The coverage domain excludes its zero
branch on every denominator coordinate. Padding remains in `values`. -/
def LogicalInputs.segment (p : Parameters) (work : List Grid.PID) (xs : LogicalInputs p work)
    (r : Fin work.length) (j : Fin (xs.lengths r)) : ℝ :=
  if h : j.val < p.width then xs.values r ⟨j.val,h⟩ else 0

end
end VeriTile.Bench.Repairs.TokenSoftmaxLlama.PublicIO
