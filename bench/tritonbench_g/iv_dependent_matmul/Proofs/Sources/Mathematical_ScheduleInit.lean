import bench.tritonbench_g.iv_dependent_matmul.Proofs.SignedArithmetic
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_KernelSource
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_FPContract

/-! All five actual pointer schedules execute the same independent input-memory
FP recurrence. The source and compiled graphs are never equated with one another. -/
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Schedules
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul.SourceArithmetic FPValue Kernels
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

def aPointerAt (p : Parameters) (s : State Word) (cur : Nat) : Values Word .ptr [p.blockRows,p.blockInner] :=
  fun i => (p.input, aOffset p s cur i)
def bPointerAt (p : Parameters) (s : State Word) (cur : Nat) : Values Word .ptr [p.blockInner,p.blockColumns] :=
  fun i => (p.weight, bOffset p s cur i)

def Pointers (schedule : Schedule) (p : Parameters) (s t : State Word) (cur : Nat) : Prop :=
  let pair (an bn : RegName) (k : Nat) :=
    t.regs .ptr [p.blockRows,p.blockInner] an = some (aPointerAt p s k) ∧
    t.regs .ptr [p.blockInner,p.blockColumns] bn = some (bPointerAt p s k)
  match schedule with
  | .preLoad => True
  | .postLoad => pair "a_ptrs" "b_ptrs" cur
  | .mixed => t.regs .ptr [p.blockInner,p.blockColumns] "b_ptrs" = some (bPointerAt p s cur)
  | .two => pair "a_ptrs" "b_ptrs" cur ∧ pair "a_ptrs_next" "b_ptrs_next" (cur+1)
  | .three => pair "a_ptrs" "b_ptrs" cur ∧ pair "a_ptrs_next" "b_ptrs_next" (cur+1) ∧
      pair "a_ptrs_next_next" "b_ptrs_next_next" (cur+2)

def ReadyAt (schedule : Schedule) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p) : Prop :=
  FPValue.Ready p s t a ∧ Pointers schedule p s t cur

theorem prologueWithPointers_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (schedule : Schedule) (p : Parameters) (s : State Word) :
    ∃ t, run (backend rest cfg D F) (prologueWithPointers schedule p) s = some t ∧
      ReadyAt schedule p s t 0 (fun _ => FP.Value.zero .fp32) := by
  cases schedule <;>
    simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,prologueWithPointers, prologue,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,backend,
      SgmvExpandSlice.FPValue.backend,LoraExpandGemv.FPValue.backend]
  all_goals dsimp [ReadyAt, Pointers]
  all_goals repeat' constructor
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,State.setReg,rowBlock,colBlock,columnsGrid,
    aPointer,bPointer,aPointerAt,bPointerAt,aOffset,bOffset,aBase,bBase,row,col,bop]
  all_goals funext i
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,aPointer,bPointer,aPointerAt,bPointerAt,aOffset,bOffset,aBase,bBase,
    row,col,rowBlock,colBlock,columnsGrid,bop]

  all_goals simp only [nat_not_negative, if_false, mod_nat_pulled, ← Int.emod_def,
    ← Int.natCast_ediv, ← Int.natCast_emod, ← Int.natCast_mul, ← Int.natCast_add,
    Int.toNat_natCast, Nat.add_assoc]

#axiomsClean prologueWithPointers_run
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Schedules
