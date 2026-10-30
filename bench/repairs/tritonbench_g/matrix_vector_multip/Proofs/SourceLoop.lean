import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceBody

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.SourceLoop
open VeriTile.Triton FP.Structural SourceBody
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def condition {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (s : State α) : Option Bool :=
  (evalOp A none (SourceCode.original N M BN BM rowShape columnShape).condition s).map
    (fun c => c PUnit.unit)

theorem condition_value {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (ctx : Context N M BN BM pid off memory acc s) :
    condition A N M BN BM rowShape columnShape s = some (decide (off<M)) := by
  simp [condition,SourceCode.original,Source.whileLoop?,Kernels.source,ComputeKernel.surfaceBody,
    evalOp_unfold,ctx.columns,ctx.counter,natLt,numericLt,bop,Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- All iterations of the original while. The measure proves termination
inside the source domain; no execution fuel is inserted in the source. -/
theorem runs {α : Type} [Inhabited α] (A : Algebra α) (N M BN BM pid off : Nat)
    (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM)
    (memory : RegionName → Nat → Cell α) (acc : Acc α BN BM) (s : State α)
    (domain : SourceDomain N M BN BM pid)
    (ctx : Context N M BN BM pid off memory acc s) :
    ∃ last t, Control.While.run (condition A N M BN BM rowShape columnShape)
      (run A (SourceCode.original N M BN BM rowShape columnShape).body) s = some t ∧
      Context N M BN BM pid last memory
        (WholeSource.accumulated A (parameters N M BN BM) pid memory off acc) t := by
  by_cases active : off<M
  · have next := SourceBody.preserves A N M BN BM pid off memory acc s ctx
    obtain ⟨last,t,ran,final⟩ := runs A N M BN BM pid (off+BM) rowShape columnShape memory
      (WholeSource.nextAcc A (parameters N M BN BM) pid off memory acc) _ domain next
    refine ⟨last,t,?_,?_⟩
    · rw [Control.While.run,condition_value A N M BN BM pid off rowShape columnShape memory acc s ctx]
      simp only [active,decide_true,Option.bind_some,if_true]
      rw [SourceBody.runs A N M BN BM pid off rowShape columnShape memory acc s domain active ctx]
      exact ran
    · rw [WholeSource.accumulated]
      simpa only [parameters,active,columnShape.1,and_self,if_true] using final
  · refine ⟨off,s,?_,?_⟩
    · rw [Control.While.run,condition_value A N M BN BM pid off rowShape columnShape memory acc s ctx]
      simp [active]
    · rw [WholeSource.accumulated]
      simpa only [parameters,active,false_and,if_false] using ctx
termination_by M-off
decreasing_by
  simp_wf
  have := columnShape.1
  omega

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.SourceLoop
