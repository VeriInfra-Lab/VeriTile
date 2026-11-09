import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_FPPrelude

/-! Successful execution of the complete repaired source for arbitrary program
IDs. This proof does not assume a successful execution, valid output geometry,
or any floating-point numerical equation. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue.Total
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

structure Ready {α : Type} (p : Parameters) (current : State α) : Prop where
  pm : ∃ v, current.regs .int [] "pid_m"=some v
  pn : ∃ v, current.regs .int [] "pid_n"=some v
  offk : ∃ v, current.regs .int [BK p] "offs_k"=some v
  am : ∃ v, current.regs .int [BM p] "offs_am"=some v
  bn : ∃ v, current.regs .int [BN p] "offs_bn"=some v
  acc : ∃ v, current.regs .real [BM p,BN p] "accumulator"=some v

theorem prelude_ready {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) : Ready p (preludeExecution M D p .single s).val := by
  constructor <;> simp [preludeExecution]

theorem body_step {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) (k : Int) (ctx : Ready p s) :
    ∃ t,run (backend M D) (body p .single) (s.setReg "k" .int [] (fun _=>k))=some t ∧ Ready p t := by
  rcases ctx.pm with ⟨pm,hpm⟩
  rcases ctx.pn with ⟨pn,hpn⟩
  rcases ctx.offk with ⟨offk,ho⟩
  rcases ctx.am with ⟨am,ha⟩
  rcases ctx.bn with ⟨bn,hb⟩
  rcases ctx.acc with ⟨acc,hacc⟩
  refine ⟨?_,?_,?_⟩
  swap
  · simp [body,source,typed,Pasted.originalKernel,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
      evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericEq,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
      ho,ha,hb,hacc,backend]
    rfl
  · constructor <;> simp [hpm,hpn,ho,ha,hb]

theorem loop_executes {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (cur stop : Int) (stride : Nat) (s : State α) (ctx : Ready p s) :
    ∃ t,rangeInt (backend M D) "k" cur stop stride (body p .single) s=some t ∧ Ready p t := by
  by_cases hz:stride=0
  · exact ⟨s,by simp [rangeInt,hz],ctx⟩
  · by_cases hc:cur<stop
    · obtain ⟨next,he,hn⟩:=body_step M D p s cur ctx
      obtain ⟨final,hf,hr⟩:=loop_executes M D p (cur+stride) stop stride next hn
      refine ⟨final,?_,hr⟩
      rw [rangeInt]
      simpa only [hz,hc,↓reduceIte,he,Option.bind_some] using hf
    · exact ⟨s,by simp [rangeInt,hz,hc],ctx⟩
termination_by (stop-cur).toNat
 decreasing_by omega

theorem epilogue_executes {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) (ctx : Ready p s) :
    ∃ t,run (backend M D) (epilogue p .single) s=some t := by
  rcases ctx.pm with ⟨pm,hpm⟩
  rcases ctx.pn with ⟨pn,hpn⟩
  rcases ctx.acc with ⟨acc,hacc⟩
  cases he:p.ACTIVATION <;>
    (refine ⟨?_,?_⟩
     swap
     simp [epilogue,source,typed,Pasted.originalKernel,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericEq,numericLe,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       hpm,hpn,hacc,backend,he]
     rfl)

theorem executes {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) : ∃ t,FP.Structural.exec (backend M D) (source p .single) s=some t := by
  let start:=preludeExecution M D p .single s
  obtain ⟨mid,hm,hc⟩:=loop_executes M D p 0 (loopStop p) 1 start.val (prelude_ready M D p s)
  obtain ⟨finish,hf⟩:=epilogue_executes M D p mid hc
  refine ⟨finish,?_⟩
  change run (backend M D) (source p .single).surfaceBody s=some finish
  rw [source_decomposition,run_append,run_append,start.property]
  simp only [Option.bind_some]
  rw [FPValue.loop_run,hm]
  exact hf

#axiomsClean executes
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue.Total
