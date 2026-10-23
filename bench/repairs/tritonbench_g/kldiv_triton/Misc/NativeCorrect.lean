import bench.repairs.tritonbench_g.kldiv_triton.Proofs.PublicIO
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.SourceNative

/-! Retained single-source native correctness. These ordinary theorems are
auxiliary results; their historical namespaces are preserved for callers. -/

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- Independent source-ordered native losses or per-chunk reduction trees. -/
def target (p : SourceNative.Primitives) (plans : SourceNative.Plans) (N B : Nat)
    (flag : Bool) (reduction : Nat) (prediction target : Nat → SourceNative.Word) (c : Nat) : SourceNative.Word :=
  if reduction=0 then SourceNative.loss p flag (prediction c) (target c)
  else SourceNative.chunk p plans flag N B c prediction target

/-- Actual FP32 primitives and ordered reduction plans select the native model;
the complete source, windows, masks and row launches are shared IO data. -/
def forwardIO (p : SourceNative.Primitives) (plans : SourceNative.Plans)
    (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (work : List Nat) :=
  IO.forward (SourceNative.model p plans) rows sy st so N B flag reduction work
def backwardIO (p : SourceNative.Primitives) (plans : SourceNative.Plans)
    (rows so st N B : Nat) (flag : Bool) (work : List Nat) :=
  IO.backward (SourceNative.model p plans) rows so st N B flag work

/-- Complete source-to-native output judgment on all FP32 words. Primitive
semantics and reduction topology are computational parameters, not FP rewrite laws. -/
theorem forward (p : SourceNative.Primitives) (plans : SourceNative.Plans)
    (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=outputCount N B reduction) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    forwardIO p plans rows sy st so N B flag reduction work ⊨
      IO.forwardFormula (target p plans N B flag reduction) := by
  apply PublicIO.forward (SourceNative.model p plans) rows sy st so N B flag reduction positive unique work order
  intro ys ts c
  unfold KernelInputs.forwardValue target
  split
  · exact SourceNative.loss_lowering p plans flag _ _
  · exact SourceNative.chunk_lowering p plans flag N B c ys ts

theorem backward (p : SourceNative.Primitives) (plans : SourceNative.Plans)
    (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=N) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    backwardIO p plans rows so st N B flag work ⊨ IO.backwardFormula (SourceNative.derivative p flag) := by
  apply PublicIO.backward (SourceNative.model p plans) rows so st N B flag positive unique work order
  exact SourceNative.derivative_lowering p plans flag

#specNonCircular target avoiding [forwardSource,backwardSource,runSchedule]
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.FPEquiv
