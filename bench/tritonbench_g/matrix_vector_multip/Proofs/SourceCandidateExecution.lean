import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceCandidateParts
import bench.tritonbench_g.matrix_vector_multip.Proofs.SourceFullExecution
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem prelude_run (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (s : State Payload) :
    run (model native p g) (prelude p) s=run (WholeNative.algebra native g) (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.prelude p) s := by
  rw [prelude_eq]
  simp [VeriTile.Bench.Repairs.MatrixVectorMultip.Source.prelude,VeriTile.Bench.Repairs.MatrixVectorMultip.Source.source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    model,VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA.algebra,WholeNative.algebra]

theorem epilogue_run (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (s : State Payload) :
    run (model native p g) (epilogue p) s=run (WholeNative.algebra native g) (VeriTile.Bench.Repairs.MatrixVectorMultip.Source.epilogue p) s := by
  rw [epilogue_eq]
  simp [VeriTile.Bench.Repairs.MatrixVectorMultip.Source.epilogue,VeriTile.Bench.Repairs.MatrixVectorMultip.Source.source,Kernels.fromParameters,arguments,VeriTile.Bench.Repairs.MatrixVectorMultip.Kernels.body,
    Kernels.arguments,Kernels.fromParameters,Kernels.fmaFromParameters,ComputeKernel.surfaceBody,ComputeKernel.withScalarArguments,ScalarArgument.integer,ScalarArgument.statement,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,FP.Structural.bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,
    Region.cast,Option.bind,integralFloorDiv,integralMod,
    model,VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA.algebra,WholeNative.algebra]

  rfl

theorem loopStatement_eq (p : Parameters) :
    loopStatement p=.forRangeDyn "m" (.ref .nat [] "m") (.ref .nat [] "M")
      (.constNat p.blockColumns) (loopBody p) := by rfl

theorem dynamicLoop (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (initial current : State Payload) (off : Nat) (acc : WholeSource.Acc Payload p)
    (h : RuntimeInvariant p initial off acc current) :
    step (model native p g) (loopStatement p) current=
      range (model native p g) "m" off p.columns p.blockColumns (loopBody p) current := by
  have bound := h.metadata "M" p.columns (by simp [scalarParameters])
  have counter := AffineWhile.normalized_read current "m" off h.counter
  rw [loopStatement_eq]
  simp [step,evalOp_unfold,bound,counter]

theorem rangeExecution (native : Scalars) (p : Parameters) (g : Graph p.blockRows p.blockColumns)
    (width : 0<p.blockColumns) (initial current : State Payload) (off : Nat)
    (acc : WholeSource.Acc Payload p) (h : RuntimeInvariant p initial off acc current) :
    ∃ last final, range (model native p g) "m" off p.columns p.blockColumns (loopBody p) current=some final ∧
      RuntimeInvariant p initial last (WholeSource.accumulated (WholeNative.algebra native g) p
        (initial.pids 0) (viewMemory initial.mem) off acc) final := by
  by_cases active : off<p.columns
  · let next := bodyExecution native p g initial current off acc h
    obtain ⟨last,final,ran,inv⟩ := rangeExecution native p g width initial next.val (off+p.blockColumns)
      (WholeSource.nextAcc (WholeNative.algebra native g) p (initial.pids 0) off (viewMemory initial.mem) acc)
      (bodyInvariant (WholeNative.algebra native g) p initial current off acc h)
    refine ⟨last,final,?_,?_⟩
    · rw [range,if_neg (by omega),if_pos active,
        show current.setReg "m" .nat [] (fun _ => off)=current from h.counter,next.property]
      exact ran
    · rw [WholeSource.accumulated]
      simpa only [active,width,and_self,if_true] using inv
  · refine ⟨off,current,?_,?_⟩
    · rw [range,if_neg (by omega),if_neg active]
    · rw [WholeSource.accumulated]
      simpa only [active,false_and,if_false] using h
termination_by p.columns-off

#axiomsClean prelude_run
#axiomsClean epilogue_run
#axiomsClean dynamicLoop
#axiomsClean rangeExecution
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source.Candidate
