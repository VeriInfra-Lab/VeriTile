import bench.tritonbench_g.iv_dependent_matmul.Proofs.SignedArithmetic
import bench.tritonbench_g.iv_dependent_matmul.Proofs.RealExpressions
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_Schedules

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.RealExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul.SourceArithmetic
open TritonBenchG.IvDependentMatmul FPValue Kernels
open TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

def aPointerAt (p : Parameters) (s : State ℝ) (cur : Nat) : Values ℝ .ptr [p.blockRows,p.blockInner] :=
  fun i => (p.input, aOffset p s cur i)
def bPointerAt (p : Parameters) (s : State ℝ) (cur : Nat) : Values ℝ .ptr [p.blockInner,p.blockColumns] :=
  fun i => (p.weight, bOffset p s cur i)

def Pointers (schedule : Schedule) (p : Parameters) (s t : State ℝ) (cur : Nat) : Prop :=
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

def ReadyAt (schedule : Schedule) (p : Parameters) (s t : State ℝ) (cur : Nat) (a : Accumulator p) : Prop :=
  Ready p s t a ∧ Pointers schedule p s t cur

theorem prologueWithPointers_run (rest : Algebra ℝ) (schedule : Schedule) (p : Parameters) (s : State ℝ) :
    ∃ t, run (backend rest) (prologueWithPointers schedule p) s = some t ∧
      ReadyAt schedule p s t 0 (fun _ => 0) := by
  cases schedule <;>
    simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,prologueWithPointers, prologue,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,backend,
      sumDot]
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


theorem schedule_body_run (rest : Algebra ℝ) (schedule : Schedule) (g : Graph) (p : Parameters) (s t : State ℝ) (cur : Nat)
    (a : Accumulator p) (h : ReadyAt schedule p s t cur a) :
    ∃ u, run (backend rest) (Kernels.body schedule g p)
        (t.setReg "k" .int [] (fun _ => (cur : Int)))=some u ∧
      ReadyAt schedule p s u (cur+1) (update g p s cur a) := by
  obtain ⟨h,hp⟩ := h
  cases schedule <;> cases g <;>
    simp only [Pointers] at hp
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,Kernels.body,beforeLoads,afterDot,FPValue.body,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,fusedTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,hp,backend,
      sumDot]
  all_goals dsimp [ReadyAt, Pointers]
  all_goals repeat' constructor
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,State.setReg,h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,hp,
    update,aValues,bValues,aPointer,bPointer,aPointerAt,bPointerAt,aBase,bBase,aOffset,bOffset,
    bop,ofFloat,toFloat,Region.cast]
  all_goals try dsimp [aValues,bValues,aOffset,bOffset,aBase,bBase]
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,show (0.0 : ℝ)=0 by norm_num,sumDot,bop]
  all_goals try rfl
  all_goals simp only [bop,aValues,bValues,aOffset,bOffset,aBase,bBase,Nat.add_assoc]
  all_goals try rfl
  all_goals funext i
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,sumDot,aValues,bValues,aOffset,bOffset,aBase,bBase,ite_mul,mul_ite,bop]

  all_goals simp only [nat_not_negative, if_false, mod_nat_pulled, ← Int.emod_def,
    ← Int.natCast_ediv, ← Int.natCast_emod, ← Int.natCast_mul, ← Int.natCast_add,
    Int.toNat_natCast, Nat.add_assoc]

theorem schedule_loop_run (rest : Algebra ℝ) (schedule : Schedule) (g : Graph) (p : Parameters) (s t : State ℝ) (cur : Nat)
    (a : Accumulator p) (h : ReadyAt schedule p s t cur a) :
    ∃ u, FP.Structural.rangeInt (backend rest) "k" (cur : Int) (count p : Int) 1
        (Kernels.body schedule g p) t=some u ∧
      Ready p s u (foldRange (update g p s) cur (count p) 1 a) := by
  apply FPValue.rangeInt_values (backend rest) "k" (Kernels.body schedule g p)
    (update g p s) (fun n a u => ReadyAt schedule p s u n a)
    (fun a u => Ready p s u a) cur (count p) 1 a t
  · intro n a u hu _
    exact schedule_body_run rest schedule g p s u n a hu
  · intro n a u hu
    exact hu.1
  · exact h

/-- Successful execution of every original pointer schedule, arbitrary ceil-K
and M/N edge masks; only numerical dot primitives remain parameters. -/
theorem execution_memory (rest : Algebra ℝ) (schedule : Schedule) (g : Graph) (p : Parameters) (s : State ℝ) :
    ∃ final, FP.Structural.exec (backend rest) (Kernels.source schedule g p) s=some final ∧
      final.mem=(written p s (accumulated g p s)).mem := by
  obtain ⟨t,ht,htr⟩ := prologueWithPointers_run rest schedule p s
  obtain ⟨u,hu,hur⟩ := schedule_loop_run rest schedule g p s t 0 _ htr
  obtain ⟨final,hf,hfm⟩ := suffix_run rest p s u _ hur
  refine ⟨final,?_,hfm⟩
  rw [SourceArithmetic.exec_surface]
  rw [Kernels.source_split]
  rw [run_append,run_append,ht]
  dsimp only [Option.bind]
  have ran : run (backend rest)
      [ComputeStmt.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (Kernels.body schedule g p)] t=some u := by
    rw [SourceArithmetic.loop_run (backend rest) p (Kernels.body schedule g p) t]
    exact hu
  rw [ran]
  exact hf


#axiomsClean prologueWithPointers_run
#axiomsClean schedule_body_run
#axiomsClean schedule_loop_run
#axiomsClean execution_memory
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.RealExecution
