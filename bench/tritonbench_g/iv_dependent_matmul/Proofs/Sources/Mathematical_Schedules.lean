import bench.tritonbench_g.iv_dependent_matmul.Proofs.Sources.Mathematical_ScheduleInit
namespace VeriTile.Bench.TritonBenchG.IvDependentMatmul.Schedules
open VeriTile.Triton FP.Structural FPValue Kernels
open TritonBenchG.IvDependentMatmul.SourceArithmetic
open ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

theorem body_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (schedule : Schedule) (g : Graph) (p : Parameters) (s t : State Word) (cur : Nat)
    (a : Accumulator p) (h : ReadyAt schedule p s t cur a) :
    ∃ u, run (backend rest cfg D F) (Kernels.body schedule g p)
        (t.setReg "k" .int [] (fun _ => (cur : Int)))=some u ∧
      ReadyAt schedule p s u (cur+1) (update cfg D F g p s cur a) := by
  obtain ⟨h,hp⟩ := h
  cases schedule <;> cases g <;>
    simp only [Pointers] at hp
  all_goals simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,Kernels.body,beforeLoads,afterDot,FPValue.body,FPValue.source,typed,commonView,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.quotation,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.prefixLength,VeriTile.Bench.TritonBenchG.IvDependentMatmul.Kernels.Schedule.name,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.originalKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.nativeKernel,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadSource,VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel.preLoadNative,fusedTyped,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,hp,backend,
      SgmvExpandSlice.FPValue.backend,SgmvExpandSlice.FPValue.installAcc,
      LoraExpandGemv.FPValue.backend,LoraExpandGemv.FPValue.formatOf]
  all_goals dsimp [ReadyAt, Pointers]
  all_goals repeat' constructor
  all_goals simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,State.setReg,h.mem,h.pids,h.undef,h.pm,h.pn,h.offk,h.aptr,h.bptr,h.acc,hp,
    update,aValues,bValues,aPointer,bPointer,aPointerAt,bPointerAt,aBase,bBase,aOffset,bOffset,
    bop,ofFloat,toFloat,Region.cast]
  all_goals try dsimp [aValues,bValues,aOffset,bOffset,aBase,bBase]
  all_goals try simp [-Nat.cast_add,-Nat.cast_mul,nat_not_negative,ceil_nat,mod_nat,mask_nat,show (0.0 : ℝ)=0 by norm_num,LoraExpandGemv.FPValue.addWord,bop]
  all_goals try rfl
  all_goals simp only [bop,FPValue.aValues,FPValue.bValues,aOffset,bOffset,aBase,bBase,Nat.add_assoc]
  all_goals simp only [mask_nat, ← Int.natCast_mul, ← Int.natCast_add,
    Int.toNat_natCast, Nat.add_assoc]
  all_goals try rfl

theorem loop_run [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (schedule : Schedule) (g : Graph) (p : Parameters) (s t : State Word) (cur : Nat)
    (a : Accumulator p) (h : ReadyAt schedule p s t cur a) :
    ∃ u, FP.Structural.rangeInt (backend rest cfg D F) "k" (cur : Int) (count p : Int) 1
        (Kernels.body schedule g p) t=some u ∧
      FPValue.Ready p s u (foldRange (update cfg D F g p s) cur (count p) 1 a) := by
  apply FPValue.rangeInt_values (backend rest cfg D F) "k" (Kernels.body schedule g p)
    (update cfg D F g p s) (fun n a u => ReadyAt schedule p s u n a)
    (fun a u => FPValue.Ready p s u a) cur (count p) 1 a t
  · intro n a u hu _
    exact body_run rest cfg D F schedule g p s u n a hu
  · intro n a u hu
    exact hu.1
  · exact h

/-- Successful execution of every original pointer schedule, arbitrary ceil-K
and M/N edge masks; only numerical dot primitives remain parameters. -/
theorem execution_memory [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (schedule : Schedule) (g : Graph) (p : Parameters) (s : State Word) :
    ∃ final, FP.Structural.exec (backend rest cfg D F) (Kernels.source schedule g p) s=some final ∧
      final.mem=(written cfg p s (accumulated cfg D F g p s)).mem := by
  obtain ⟨t,ht,htr⟩ := prologueWithPointers_run rest cfg D F schedule p s
  obtain ⟨u,hu,hur⟩ := loop_run rest cfg D F schedule g p s t 0 _ htr
  obtain ⟨final,hf,hfm⟩ := suffix_run rest cfg D F p s u _ hur
  refine ⟨final,?_,hfm⟩
  rw [SourceArithmetic.exec_surface]
  rw [Kernels.source_split]
  rw [run_append,run_append,ht]
  dsimp only [Option.bind]
  have ran : run (backend rest cfg D F)
      [ComputeStmt.forRangeInt "k" (.constInt 0) (stopOp p) (.constNat 1) (Kernels.body schedule g p)] t=some u := by
    rw [SourceArithmetic.loop_run (backend rest cfg D F) p (Kernels.body schedule g p) t]
    exact hu
  rw [ran]
  exact hf

theorem schedule_memory_equivalence [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (D : Dot) (F : DotAcc) (a b : Schedule) (g : Graph) (p : Parameters) (s : State Word) :
    ∃ x y, FP.Structural.exec (backend rest cfg D F) (Kernels.source a g p) s=some x ∧
      FP.Structural.exec (backend rest cfg D F) (Kernels.source b g p) s=some y ∧ x.mem=y.mem := by
  obtain ⟨x,hx,hmx⟩ := execution_memory rest cfg D F a g p s
  obtain ⟨y,hy,hmy⟩ := execution_memory rest cfg D F b g p s
  exact ⟨x,y,hx,hy,hmx.trans hmy.symm⟩

#axiomsClean prologueWithPointers_run
#axiomsClean body_run
#axiomsClean loop_run
#axiomsClean execution_memory
#axiomsClean schedule_memory_equivalence
end
end VeriTile.Bench.TritonBenchG.IvDependentMatmul.Schedules
