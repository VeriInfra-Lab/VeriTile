import bench.tritonbench_g.triton_matmul.Proofs.Mathematical_FPComplete
import bench.tritonbench_g.triton_matmul.Proofs.SourcePreludeContext
import bench.tritonbench_g.triton_matmul.Proofs.SourceOutputExecution
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 1000000

theorem execution_safe_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) (hm : 0<BM p) (hn : 0<BN p)
    (hf : SourceControl.first p initial≤SourceControl.rows p) (hk : ceilCount p.K (BK p)=p.numKBlocks) :
    ∃ final,exec (backend M D) (typed p st) initial=some final ∧
      final.mem=(writtenSafe M D p st initial).mem := by
  let start:=preludeExecution M D p st initial
  have hc:=preludeContext M D p st initial hm hn hf
  obtain ⟨middle,he,ctx,acc⟩:=loopExecution M D p st initial start.val hc.1 hc.2
  let finish:=epilogueExecution M D p st initial middle ctx acc
  refine ⟨finish.val,?_,epilogueMemory M D p st initial middle ctx acc⟩
  change run (backend M D) (typed p st).surfaceBody initial=some finish.val
  rw [decomposition,run_append,run_append]
  simp only [start.property,Option.bind_some]
  have hr : run (backend M D) [.forRangeInt "k" (.constInt 0) (.ref .int [] "blocks_k")
      (.constNat 1) (iteration p st)] start.val=some middle := by
    have hb : start.val.regs .int [] "blocks_k"=some (fun _=>(p.numKBlocks : Int)) := by
      simpa only [hk] using hc.1.blocks
    simp only [run,step,evalOp_unfold,hb,Option.bind_some,Option.bind_eq_bind,Option.pure_def]
    rw [he]
    rfl
  simp only [hr,Option.bind_some]
  exact finish.property

theorem writtenSafe_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : writtenSafe M D p st initial=written M D p st initial := by
  have stepEq (t : State α) (i : CIndex p) :
      (if active p initial i then t.write p.C (cSafeOffset p initial i)
        (stored st (outputValue M D p st initial i)) else t) =
      (if active p initial i then t.write p.C (cOffset p initial i)
        (stored st (outputValue M D p st initial i)) else t) := by
    by_cases h : active p initial i
    · simp [h,cSafeOffset,cOffset,aRow,bCol,h.1,h.2]
    · simp [h]
  unfold writtenSafe written sourceOutputValue
  simp_rw [stepEq]



theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) (hm : 0<BM p) (hn : 0<BN p)
    (hf : SourceControl.first p initial≤SourceControl.rows p) (hk : ceilCount p.K (BK p)=p.numKBlocks) :
    ∃ final,exec (backend M D) (typed p st) initial=some final ∧
      final.mem=(written M D p st initial).mem := by
  obtain ⟨final,he,hm⟩:=execution_safe_memory M D p st initial hm hn hf hk
  exact ⟨final,he,by simpa only [writtenSafe_eq] using hm⟩
#axiomsClean execution_memory

#axiomsClean execution_safe_memory
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
