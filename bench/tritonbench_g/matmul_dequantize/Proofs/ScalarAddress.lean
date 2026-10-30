import bench.tritonbench_g.matmul_dequantize.CandidateIO
import bench.tritonbench_g.matmul_dequantize.Proofs.FirstPublicResult
import bench.tritonbench_g.matmul_dequantize.Proofs.SecondPublicResult
import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant32IOResult
import bench.tritonbench_g.matmul_dequantize.Proofs.Dequant16IOResult

namespace VeriTile.Bench.Repairs.MatmulDequantize.ScalarAddress
open VeriTile Triton FP.Structural FP.Structural.Safety
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.TritonBenchG.MatmulDequantize
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

/-- Task-local transport across a pure source frontRun; the remainder is the
same complete quoted kernel, including every branch, load, dot and store. -/
theorem prefix_execution (M : Algebra A) (original candidate : ComputeKernel) (n : Nat)
    (frontRun : ∀ s, run M (original.surfaceBody.take n) s=run M (candidate.surfaceBody.take n) s)
    (suffix : original.surfaceBody.drop n=candidate.surfaceBody.drop n) (s : State A) :
    exec M original s=exec M candidate s := by
  change run M original.surfaceBody s=run M candidate.surfaceBody s
  rw [←List.take_append_drop n original.surfaceBody,←List.take_append_drop n candidate.surfaceBody,
    run_append,run_append,frontRun,suffix]

theorem prefix_safety (M : Algebra A) (original candidate : ComputeKernel) (n : Nat)
    (frontRun : ∀ s, run M (original.surfaceBody.take n) s=run M (candidate.surfaceBody.take n) s)
    (suffix : original.surfaceBody.drop n=candidate.surfaceBody.drop n)
    (pureOriginal : ∀ bounds s,traceSafeList M bounds (original.surfaceBody.take n) s)
    (pureCandidate : ∀ bounds s,traceSafeList M bounds (candidate.surfaceBody.take n) s)
    (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds original s ↔ TraceSafe M bounds candidate s := by
  unfold TraceSafe
  rw [←List.take_append_drop n original.surfaceBody,←List.take_append_drop n candidate.surfaceBody,
    traceSafeList_append,traceSafeList_append]
  simp only [pureOriginal,pureCandidate,true_and,frontRun,suffix]

namespace First
abbrev original := FPTypedFirst.source
abbrev candidate := Candidate.First.source

theorem frontRun (M : Algebra A) (p : Candidate.First.Parameters) (s : State A) :
    run M ((original p).surfaceBody.take 7) s=run M ((candidate p).surfaceBody.take 7) s := by
  simp [original,candidate,FPTypedFirst.source,Candidate.First.source,
    VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped.matmul4Typed,KernelSource.groupedGPTQ,
    ComputeKernel.surfaceBody,run,step,evalExpr,FP.Structural.evalOp_unfold,
    FP.Structural.numeric,FP.Structural.bop,Option.bind,bind,integralFloorDiv]
  congr 1

theorem suffix (p : Candidate.First.Parameters) :
    (original p).surfaceBody.drop 7=(candidate p).surfaceBody.drop 7 := rfl

theorem pure_prefix (M : Algebra A) (p : Candidate.First.Parameters) (bounds : RegionName → Nat) (s : State A) :
    traceSafeList M bounds ((original p).surfaceBody.take 7) s ∧
      traceSafeList M bounds ((candidate p).surfaceBody.take 7) s := by
  simp [original,candidate,FPTypedFirst.source,Candidate.First.source,
    VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped.matmul4Typed,KernelSource.groupedGPTQ,
    ComputeKernel.surfaceBody,traceSafeList_cons,stepSafe_unfold,safeExpr_unfold,safeOp_unfold]

theorem execution (M : Algebra A) (p : Candidate.First.Parameters) (s : State A) :
    exec M (original p) s=exec M (candidate p) s :=
  prefix_execution M _ _ 7 (frontRun M p) (suffix p) s

theorem safety (M : Algebra A) (p : Candidate.First.Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (original p) s ↔ TraceSafe M bounds (candidate p) s :=
  prefix_safety M _ _ 7 (frontRun M p) (suffix p)
    (fun b t => (pure_prefix M p b t).1) (fun b t => (pure_prefix M p b t).2) bounds s
end First

namespace Second
abbrev original := FPTypedSecond.source
abbrev candidate := Candidate.Second.source

theorem frontRun (M : Algebra A) (p : Candidate.Second.Parameters) (s : State A) :
    run M ((original p).surfaceBody.take 6) s=run M ((candidate p).surfaceBody.take 6) s := by
  simp [original,candidate,FPTypedSecond.source,Candidate.Second.source,
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPTyped.matmulTyped,KernelSource.groupedS2,
    ComputeKernel.surfaceBody,run,step,evalExpr,FP.Structural.evalOp_unfold,
    FP.Structural.numeric,FP.Structural.bop,Option.bind,bind,integralFloorDiv]
  congr 1

theorem suffix (p : Candidate.Second.Parameters) :
    (original p).surfaceBody.drop 6=(candidate p).surfaceBody.drop 6 := rfl

theorem pure_prefix (M : Algebra A) (p : Candidate.Second.Parameters) (bounds : RegionName → Nat) (s : State A) :
    traceSafeList M bounds ((original p).surfaceBody.take 6) s ∧
      traceSafeList M bounds ((candidate p).surfaceBody.take 6) s := by
  simp [original,candidate,FPTypedSecond.source,Candidate.Second.source,
    VeriTile.Bench.Repairs.MatmulDequantizeWhole001.FPTyped.matmulTyped,KernelSource.groupedS2,
    ComputeKernel.surfaceBody,traceSafeList_cons,stepSafe_unfold,safeExpr_unfold,safeOp_unfold]

theorem execution (M : Algebra A) (p : Candidate.Second.Parameters) (s : State A) :
    exec M (original p) s=exec M (candidate p) s :=
  prefix_execution M _ _ 6 (frontRun M p) (suffix p) s

theorem safety (M : Algebra A) (p : Candidate.Second.Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (original p) s ↔ TraceSafe M bounds (candidate p) s :=
  prefix_safety M _ _ 6 (frontRun M p) (suffix p)
    (fun b t => (pure_prefix M p b t).1) (fun b t => (pure_prefix M p b t).2) bounds s
end Second

namespace D32
abbrev original := Dequant32.source
abbrev candidate := Candidate.D32.source

theorem frontRun (M : Algebra A) (p : Candidate.D32.Parameters) (s : State A) :
    run M ((original p).surfaceBody.take 5) s=run M ((candidate p).surfaceBody.take 5) s := by
  simp [original,candidate,Dequant32.source,Dequant32.typed,Candidate.D32.source,
    VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped.dequantTyped,KernelSource.affineDequant,
    ComputeKernel.surfaceBody,run,step,evalExpr,FP.Structural.evalOp_unfold,
    FP.Structural.numeric,FP.Structural.bop,Option.bind,bind,integralFloorDiv]
  congr 1
  funext i
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.mul_assoc]
  ac_rfl

theorem suffix (p : Candidate.D32.Parameters) :
    (original p).surfaceBody.drop 5=(candidate p).surfaceBody.drop 5 := rfl

theorem pure_prefix (M : Algebra A) (p : Candidate.D32.Parameters) (bounds : RegionName → Nat) (s : State A) :
    traceSafeList M bounds ((original p).surfaceBody.take 5) s ∧
      traceSafeList M bounds ((candidate p).surfaceBody.take 5) s := by
  simp [original,candidate,Dequant32.source,Dequant32.typed,Candidate.D32.source,
    VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped.dequantTyped,KernelSource.affineDequant,
    ComputeKernel.surfaceBody,traceSafeList_cons,stepSafe_unfold,safeExpr_unfold,safeOp_unfold]

theorem execution (M : Algebra A) (p : Candidate.D32.Parameters) (s : State A) :
    exec M (original p) s=exec M (candidate p) s :=
  prefix_execution M _ _ 5 (frontRun M p) (suffix p) s

theorem safety (M : Algebra A) (p : Candidate.D32.Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (original p) s ↔ TraceSafe M bounds (candidate p) s :=
  prefix_safety M _ _ 5 (frontRun M p) (suffix p)
    (fun b t => (pure_prefix M p b t).1) (fun b t => (pure_prefix M p b t).2) bounds s
end D32

namespace D16
abbrev original := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.source
abbrev candidate := Candidate.D16.source

theorem frontRun (M : Algebra A) (p : Candidate.D16.Parameters) (s : State A) :
    run M ((original p).surfaceBody.take 5) s=run M ((candidate p).surfaceBody.take 5) s := by
  simp [original,candidate,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.source,Candidate.D16.source,
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,KernelSource.affineDequant,
    ComputeKernel.surfaceBody,run,step,evalExpr,FP.Structural.evalOp_unfold,
    FP.Structural.numeric,FP.Structural.bop,Option.bind,bind,integralFloorDiv]
  congr 1
  funext i
  simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.mul_assoc]
  ac_rfl

theorem suffix (p : Candidate.D16.Parameters) :
    (original p).surfaceBody.drop 5=(candidate p).surfaceBody.drop 5 := rfl

theorem pure_prefix (M : Algebra A) (p : Candidate.D16.Parameters) (bounds : RegionName → Nat) (s : State A) :
    traceSafeList M bounds ((original p).surfaceBody.take 5) s ∧
      traceSafeList M bounds ((candidate p).surfaceBody.take 5) s := by
  simp [original,candidate,VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.source,Candidate.D16.source,
    VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.typed,KernelSource.affineDequant,
    ComputeKernel.surfaceBody,traceSafeList_cons,stepSafe_unfold,safeExpr_unfold,safeOp_unfold]

theorem execution (M : Algebra A) (p : Candidate.D16.Parameters) (s : State A) :
    exec M (original p) s=exec M (candidate p) s :=
  prefix_execution M _ _ 5 (frontRun M p) (suffix p) s

theorem safety (M : Algebra A) (p : Candidate.D16.Parameters) (bounds : RegionName → Nat) (s : State A) :
    TraceSafe M bounds (original p) s ↔ TraceSafe M bounds (candidate p) s :=
  prefix_safety M _ _ 5 (frontRun M p) (suffix p)
    (fun b t => (pure_prefix M p b t).1) (fun b t => (pure_prefix M p b t).2) bounds s
end D16

theorem schedule_execution (left right : Nat → Invocation A)
    (same : ∀ pid s,(left pid).run s=(right pid).run s) (jobs : List Nat) (s : State A) :
    runSchedule (jobs.map left) s=runSchedule (jobs.map right) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons,runSchedule,same]
    congr 1
    funext t
    exact ih t

theorem schedule_safety (left right : Nat → Invocation A)
    (same : ∀ pid s,(left pid).run s=(right pid).run s)
    (safe : ∀ pid bounds s,(left pid).Safe bounds s ↔ (right pid).Safe bounds s)
    (jobs : List Nat) (bounds : RegionName → Nat) (s : State A) :
    ScheduleSafe bounds (jobs.map left) s ↔ ScheduleSafe bounds (jobs.map right) s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons pid rest ih => simp only [List.map_cons,ScheduleSafe,same,safe,ih]

namespace First
theorem launch_execution (M : Algebra A) (D : Candidate.First.Dot A) (p : Candidate.First.Parameters) (pid : Nat) (s : State A) :
    (FirstIO.invocation M D p pid).run s=(Candidate.First.invocation M D p pid).run s :=
  execution (FPTypedFirst.backend M D) p ((FirstIO.invocation M D p pid).initial s)

theorem launch_safety (M : Algebra A) (D : Candidate.First.Dot A) (p : Candidate.First.Parameters) (pid : Nat)
    (bounds : RegionName → Nat) (s : State A) :
    (FirstIO.invocation M D p pid).Safe bounds s ↔ (Candidate.First.invocation M D p pid).Safe bounds s :=
  safety (FPTypedFirst.backend M D) p bounds ((FirstIO.invocation M D p pid).initial s)

theorem equivalent (M : Algebra A) (D : Candidate.First.Dot A) (p : Candidate.First.Parameters)
    (valid : VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range
      (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.count (FirstGrid.geometry p)))) :
    (FirstIO.program M D p jobs).Equivalent (Candidate.First.variant M D p jobs) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    FirstIO.expression_implements M D p valid ha hb hs hz jobs schedule vals domain bounds bounded initial binding
  have safeTransport := schedule_safety (FirstIO.invocation M D p) (Candidate.First.invocation M D p)
    (launch_execution M D p) (launch_safety M D p) jobs bounds initial
  have runTransport := schedule_execution (FirstIO.invocation M D p) (Candidate.First.invocation M D p)
    (launch_execution M D p) jobs initial
  refine ⟨safe,safeTransport.mp safe,final,final,ran,runTransport.symm.trans ran,?_,frame,frame⟩
  intro o i active
  rfl
#axiomsClean execution
#axiomsClean safety
#axiomsClean equivalent
end First

namespace Second
theorem launch_execution (M : Algebra A) (D : Candidate.Second.Dot A) (p : Candidate.Second.Parameters) (pid : Nat) (s : State A) :
    (SecondIO.invocation M D p pid).run s=(Candidate.Second.invocation M D p pid).run s :=
  execution (FPTypedSecond.backend M D) p ((SecondIO.invocation M D p pid).initial s)

theorem launch_safety (M : Algebra A) (D : Candidate.Second.Dot A) (p : Candidate.Second.Parameters) (pid : Nat)
    (bounds : RegionName → Nat) (s : State A) :
    (SecondIO.invocation M D p pid).Safe bounds s ↔ (Candidate.Second.invocation M D p pid).Safe bounds s :=
  safety (FPTypedSecond.backend M D) p bounds ((SecondIO.invocation M D p pid).initial s)

theorem equivalent (M : Algebra A) (D : Candidate.Second.Dot A) (p : Candidate.Second.Parameters)
    (valid : VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range
      (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.count (SecondGrid.geometry p)))) :
    (SecondIO.program M D p jobs).Equivalent (Candidate.Second.variant M D p jobs) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    SecondIO.expression_implements M D p valid ha hb hs hz jobs schedule vals domain bounds bounded initial binding
  have safeTransport := schedule_safety (SecondIO.invocation M D p) (Candidate.Second.invocation M D p)
    (launch_execution M D p) (launch_safety M D p) jobs bounds initial
  have runTransport := schedule_execution (SecondIO.invocation M D p) (Candidate.Second.invocation M D p)
    (launch_execution M D p) jobs initial
  refine ⟨safe,safeTransport.mp safe,final,final,ran,runTransport.symm.trans ran,?_,frame,frame⟩
  intro o i active
  rfl
#axiomsClean execution
#axiomsClean safety
#axiomsClean equivalent
end Second

namespace D32
theorem launch_execution (M : Algebra A) (p : Candidate.D32.Parameters) (pid : Nat) (s : State A) :
    (D32IO.invocation M p pid).run s=(Candidate.D32.invocation M p pid).run s :=
  execution (M) p ((D32IO.invocation M p pid).initial s)

theorem launch_safety (M : Algebra A) (p : Candidate.D32.Parameters) (pid : Nat)
    (bounds : RegionName → Nat) (s : State A) :
    (D32IO.invocation M p pid).Safe bounds s ↔ (Candidate.D32.invocation M p pid).Safe bounds s :=
  safety (M) p bounds ((D32IO.invocation M p pid).initial s)

theorem equivalent (M : Algebra A) (p : Candidate.D32.Parameters)
    (valid : VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range
      (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.count (DequantGrid32.geometry p)))) :
    (D32IO.program M p jobs).Equivalent (Candidate.D32.variant M p jobs) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    D32IO.expression_implements M p valid hb hs hz jobs schedule vals domain bounds bounded initial binding
  have safeTransport := schedule_safety (D32IO.invocation M p) (Candidate.D32.invocation M p)
    (launch_execution M p) (launch_safety M p) jobs bounds initial
  have runTransport := schedule_execution (D32IO.invocation M p) (Candidate.D32.invocation M p)
    (launch_execution M p) jobs initial
  refine ⟨safe,safeTransport.mp safe,final,final,ran,runTransport.symm.trans ran,?_,frame,frame⟩
  intro o i active
  rfl
#axiomsClean execution
#axiomsClean safety
#axiomsClean equivalent
end D32

namespace D16
theorem launch_execution (M : Algebra A) (p : Candidate.D16.Parameters) (pid : Nat) (s : State A) :
    (D16IO.invocation M p pid).run s=(Candidate.D16.invocation M p pid).run s :=
  execution (M) p ((D16IO.invocation M p pid).initial s)

theorem launch_safety (M : Algebra A) (p : Candidate.D16.Parameters) (pid : Nat)
    (bounds : RegionName → Nat) (s : State A) :
    (D16IO.invocation M p pid).Safe bounds s ↔ (Candidate.D16.invocation M p pid).Safe bounds s :=
  safety (M) p bounds ((D16IO.invocation M p pid).initial s)

theorem equivalent (M : Algebra A) (p : Candidate.D16.Parameters)
    (valid : VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.Valid (DequantGrid16.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (jobs : List Nat) (schedule : jobs.Perm (List.range
      (VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Geometry.count (DequantGrid16.geometry p)))) :
    (D16IO.program M p jobs).Equivalent (Candidate.D16.variant M p jobs) := by
  intro vals domain bounds bounded initial binding
  obtain ⟨safe,final,ran,output,frame⟩ :=
    D16IO.expression_implements M p valid hb hs hz jobs schedule vals domain bounds bounded initial binding
  have safeTransport := schedule_safety (D16IO.invocation M p) (Candidate.D16.invocation M p)
    (launch_execution M p) (launch_safety M p) jobs bounds initial
  have runTransport := schedule_execution (D16IO.invocation M p) (Candidate.D16.invocation M p)
    (launch_execution M p) jobs initial
  refine ⟨safe,safeTransport.mp safe,final,final,ran,runTransport.symm.trans ran,?_,frame,frame⟩
  intro o i active
  rfl
#axiomsClean execution
#axiomsClean safety
#axiomsClean equivalent
end D16

end
end VeriTile.Bench.Repairs.MatmulDequantize.ScalarAddress
