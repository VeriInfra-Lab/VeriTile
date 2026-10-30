import bench.tritonbench_g.matmul_kernel.Proofs.KernelExecution
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active scatter_memory)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial current : State α)
    (ctx : Context p initial 0 current)
    (ha : HasAcc p (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, FP.Structural.range (backend M D) "k" 0 p.steps 1 (body p) current=some t ∧
      ∃ k, Context p initial k t ∧ HasAcc p (accumulation M D p initial) t := by
  let I := fun n (a : CIndex p → α) (s : State α) => Context p initial n s ∧ HasAcc p a s
  exact range_values (backend M D) "k" (body p) (update M D p initial)
    I (fun a s => ∃ k, Context p initial k s ∧ HasAcc p a s) 0 p.steps 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p initial s k a h.1 h.2⟩)
    (by intro n a s h; exact ⟨n,h⟩) ⟨ctx,ha⟩

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p initial) current) :
    {t : State α // FP.Structural.run (backend M D) (epilogue p) current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_, ?_⟩
  swap
  simp [epilogue,source,typed,ComputeKernel.surfaceBody,FP.Structural.run,FP.Structural.step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,VeriTile.Triton.Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.pm,ctx.pn,ha,backend]
  rfl

def written {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write "C" (cOffset p initial i) (stored (outputValue M D p initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p initial) current) :
    (epilogueExecution M D p initial current k ctx ha).val.mem=(written M D p initial).mem := by
  simp [epilogueExecution,written,masked_scatter_memory,ctx.memory,
    active,cOffset,row,col,outputValue,stored,backend,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp only [scatter_memory,State.setReg_mem,ctx.memory]

def Post {α : Type} (p : Config) (initial final : State α) (v : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem "C" (cOffset p initial i)=stored (v i)) ∧
  (∀ (r : RegionName) o, (r≠"C" ∨ ∀ i, active p initial i → o≠cOffset p initial i) → final.mem r o=initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Config) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j → cOffset p initial i=cOffset p initial j → i=j) :
    ∃ final, FP.Structural.exec (backend M D) (source p) initial=some final ∧
      final.mem=(written M D p initial).mem ∧ Post p initial final (outputValue M D p initial) := by
  let start := preludeExecution M D p initial
  obtain ⟨mid,hm,k,hc,ha⟩ := loopExecution M D p initial start.val
    (preludeContext M D p initial).1 (preludeContext M D p initial).2
  let finish := epilogueExecution M D p initial mid k hc ha
  refine ⟨finish.val,?_,epilogueMemory M D p initial mid k hc ha,?_,?_⟩
  · change FP.Structural.run (backend M D) (source p).surfaceBody initial=some finish.val
    rw [source_decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some]
    have hr : FP.Structural.run (backend M D) [.forRange "k" 0 p.steps 1 (body p)] start.val=some mid := by
      simpa [FP.Structural.run,FP.Structural.step] using hm
    rw [hr]
    exact finish.property
  · intro i hi
    rw [epilogueMemory]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o h
    rw [epilogueMemory]
    exact State.masked_scatter_frame ("C" : RegionName) (cOffset p initial)
      (fun i=>stored (outputValue M D p initial i)) (active p initial)
      (TileShape.allIndices [BM p,BN p]) r o (h.imp id (fun miss i _ hi=>miss i hi)) initial

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelSource
