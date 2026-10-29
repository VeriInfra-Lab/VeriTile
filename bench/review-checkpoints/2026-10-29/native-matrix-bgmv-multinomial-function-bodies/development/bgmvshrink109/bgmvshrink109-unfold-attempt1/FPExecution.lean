import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.FPTail
import bench.repairs.tritonbench_g.bgmv_expand_slice.Proofs.FPControl
namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPExecution
open VeriTile.Triton FP.Structural FPProgram VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPValues FPSetup FPBody
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
variable {α : Type} [Inhabited α]
namespace Control
abbrev single := @VeriTile.Bench.Repairs.BgmvExpandSlice.FPControl.single
abbrev pass := @VeriTile.Bench.Repairs.BgmvExpandSlice.FPControl.pass
abbrev skip := @VeriTile.Bench.Repairs.BgmvExpandSlice.FPControl.skip
abbrev integer_guard := @VeriTile.Bench.Repairs.BgmvExpandSlice.FPControl.integer_guard
end Control

theorem exec_code (M : Algebra α) (h : Call) (s : State α) :
    FP.Structural.exec M (Surface.source h) s=run M (code h) s := by
  unfold code
  cases eq : h.storage <;> simp only [Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,eq]
  all_goals rfl

theorem preguard_memory (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) :
    (preguard M P h batch split data).val.mem=(entry h batch split data).mem := by
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> rfl

theorem condition (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α) :
    FP.Structural.evalOp (machine M P) none guard (preguard M P h batch split data).val=
      some (fun _ => decide (data.indices batch ≠ -1)) := by
  rw [FPProgram.guard,SourceControl.eval_not_eq]
  apply Control.integer_guard
  rcases h with ⟨batches,adapters,N,K,BN,columns,positive,covered,outputCovered,storage,outputType,scaling⟩
  cases storage <;> rfl

/-- Complete typed source execution, including sentinel and all range steps.
Every active source atomic event has the explicit local FP payload. No source
success, arithmetic output, address or memory hypothesis is deferred numerically. -/
theorem source_executes (M : Algebra α) (P : Primitives α) (h : Call)
    (batch split : Nat) (data : Data α)
    (valid : data.indices batch=-1 ∨ 0 ≤ data.indices batch) :
    ∃ t,FP.Structural.exec (machine M P) (Surface.source h) (entry h batch split data)=some t ∧
      t.mem=(if data.indices batch=-1 then (entry h batch split data).mem else
        (VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile h P batch
          (payload M P h data batch (data.indices batch).toNat split) (entry h batch split data)).mem) := by
  rw [exec_code,complete_code,run_append,(preguard M P h batch split data).property,Option.bind_some]
  by_cases sentinel : data.indices batch=-1
  · refine ⟨(preguard M P h batch split data).val,?_,?_⟩
    · apply Control.single
      apply Control.skip
      simpa only [sentinel,ne_self_iff_false,decide_false] using condition M P h batch split data
    · simpa only [sentinel,if_true] using preguard_memory M P h batch split data
  · have nonnegative := valid.resolve_left sentinel
    have metadata : data.indices batch=((data.indices batch).toNat:Int) := (Int.toNat_of_nonneg nonnegative).symm
    obtain ⟨after,loopRun,ready⟩ := FPLoop.loop_runs M P h batch split (data.indices batch).toNat data metadata
    let final := FPTail.finish M P h batch split (data.indices batch).toNat data after ready
    refine ⟨final.val,?_,?_⟩
    · apply Control.single
      apply Control.pass
      · simpa [sentinel] using condition M P h batch split data
      · rw [complete_active,run_append,(preloop M P h batch split (data.indices batch).toNat data metadata).property,Option.bind_some]
        simp only [run,loopRun,Option.bind_some]
        exact final.property
    · simp only [sentinel,if_false]
      rw [FPTail.finish_memory]
      unfold VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.tile
      apply VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.AtomicScatter.fold_memory
      · exact ready.memory
      · intro i u v same
        by_cases active : i.1.val < h.N <;> simp only [active,if_true,if_false]
        · simp only [State.write,same]
        · exact same


#axiomsClean source_executes
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.FPExecution
