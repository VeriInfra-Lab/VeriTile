import bench.tritonbench_g.iv_dependent_matmul.Proofs.SignedArithmetic
import bench.tritonbench_g.iv_dependent_matmul.Proofs.SignedRange
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_FPReference
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul.SourceArithmetic
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def aPointer (p : Parameters) (s : State Word) : Values Word .ptr [p.blockRows,p.blockInner] :=
  fun i => (p.input,aBase p s i)
def bPointer (p : Parameters) (s : State Word) : Values Word .ptr [p.blockInner,p.blockColumns] :=
  fun i => (p.weight,bBase p s i)
structure Ready (p : Parameters) (s t : State Word) (a : Accumulator p) : Prop where
  mem : t.mem=s.mem
  pids : t.pids=s.pids
  undef : t.undef=s.undef
  pm : t.regs .int [] "pid_m"=some (fun _ => (rowBlock p s : Int))
  pn : t.regs .int [] "pid_n"=some (fun _ => (colBlock p s : Int))
  offk : t.regs .int [p.blockInner] "offs_k"=some (fun i => (i.1.val : Int))
  aptr : t.regs .ptr [p.blockRows,p.blockInner] "a_ptr"=some (aPointer p s)
  bptr : t.regs .ptr [p.blockInner,p.blockColumns] "b_ptr"=some (bPointer p s)
  acc : t.regs .real [p.blockRows,p.blockColumns] "accumulator"=some a

theorem prologue_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (p : Parameters) (s : State Word) :
    ∃ t, run (backend rest cfg D F) (prologue p) s=some t ∧
      Ready p s t (fun _ => FP.Value.zero .fp32) := by
  simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, prologue,source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,backend,
    SgmvExpandSlice.FPValue.backend,LoraExpandGemv.FPValue.backend]
  constructor <;> simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, State.setReg,rowBlock,colBlock,columnsGrid,aPointer,bPointer,aBase,bBase,row,col,bop]
  all_goals funext i
  all_goals simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, aPointer,bPointer,aBase,bBase,row,col,rowBlock,colBlock,columnsGrid,bop]
  all_goals simp only [nat_not_negative, if_false, mod_nat_pulled, ← Int.emod_def,
    ← Int.natCast_ediv, ← Int.natCast_emod, ← Int.natCast_mul, ← Int.natCast_add,
    Int.toNat_natCast]

theorem body_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t a) :
    ∃ u, run (backend rest cfg D F) (body g p) (t.setReg "k" .int [] (fun _ => (cur : Int)))=some u ∧
      Ready p s u (update cfg D F g p s cur a) := by
  cases g <;>
    simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, body,source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,fusedTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,backend,
      SgmvExpandSlice.FPValue.backend,SgmvExpandSlice.FPValue.installAcc,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  all_goals constructor
  all_goals simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, State.setReg,h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,
    update,aValues,bValues,aPointer,bPointer,aBase,bBase,aOffset,bOffset,bop,ofFloat,toFloat,Region.cast]
  all_goals try dsimp [aValues,bValues,aOffset,bOffset,aBase,bBase]
  all_goals simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, show (0.0 : ℝ)=0 by norm_num,LoraExpandGemv.FPValue.addWord,bop]
  all_goals try rfl

theorem loop_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (g : Graph) (p : Parameters) (s t : State Word) (cur : Nat) (a : Accumulator p)
    (h : Ready p s t a) :
    ∃ u, FP.Structural.rangeInt (backend rest cfg D F) "k" (cur : Int) (count p : Int) 1 (body g p) t=some u ∧
      Ready p s u (foldRange (update cfg D F g p s) cur (count p) 1 a) := by
  apply FPValue.rangeInt_values (backend rest cfg D F) "k" (body g p)
    (update cfg D F g p s) (fun _ a u => Ready p s u a)
    (fun a u => Ready p s u a) cur (count p) 1 a t
  · intro n a u hu _
    exact body_run rest cfg D F g p s u n a hu
  · intro n a u hu
    exact hu
  · exact h

theorem suffix_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (p : Parameters) (s t : State Word) (a : Accumulator p) (h : Ready p s t a) :
    ∃ u, run (backend rest cfg D F) (suffix p) t=some u ∧ u.mem=(written cfg p s a).mem := by
  simp [-Nat.cast_add,-Nat.cast_mul, nat_not_negative, ceil_nat, mod_nat, mask_nat, suffix,source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
    Region.cast,ofFloat,toFloat,State.setReg,store,h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,
    backend,SgmvExpandSlice.FPValue.backend,LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  simp only [← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast, Int.ofNat_lt]
  simp only [written,active,outputOffset,outputValues,row,col,decide_eq_true_eq,Bool.and_eq_true]
  apply SgmvExpandSlice.FPValue.fold_memory_congr
  rfl
#axiomsClean prologue_run
#axiomsClean body_run
#axiomsClean loop_run
#axiomsClean suffix_run
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.FPValue
