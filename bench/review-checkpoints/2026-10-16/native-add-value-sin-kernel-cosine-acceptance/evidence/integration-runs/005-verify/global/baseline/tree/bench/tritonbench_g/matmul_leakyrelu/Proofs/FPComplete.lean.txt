import bench.tritonbench_g.matmul_leakyrelu.Proofs.FPExecution
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial 0 current)
    (ha : HasAcc p (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, FP.Structural.range (backend M D) "k" 0 (trips p) 1 (body p st) current=some t ∧
      ∃ k, Context p initial k t ∧ HasAcc p (accumulation M D p st initial) t := by
  let I := fun n (a : CIndex p → α) (s : State α) => Context p initial n s ∧ HasAcc p a s
  exact range_values (backend M D) "k" (body p st) (update M D p st initial)
    I (fun a s => ∃ k, Context p initial k s ∧ HasAcc p a s) 0 (trips p) 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p st initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p st initial s k a h.1 h.2⟩)
    (by intro n a s h; exact ⟨n,h⟩) ⟨ctx,ha⟩

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p st initial) current) :
    {t : State α // run (backend M D) (epilogue p st) current=some t} := by
  simp only [HasAcc] at ha
  cases st <;> cases he : p.ACTIVATION <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [epilogue,source,typed,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLe,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.pm,ctx.pn,ha,backend,he]
     rfl)

def written {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cOffset p initial i) (stored st (outputValue M D p st initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p st initial) current) :
    (epilogueExecution M D p st initial current k ctx ha).val.mem=(written M D p st initial).mem := by
  cases p
  rename_i enabled
  cases enabled <;> cases st <;> simp [epilogueExecution,written,masked_scatter_memory,ctx.memory,
    active,cOffset,row,col,outputValue,activation,stored,dtype,backend,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

def Post {α : Type} (p : Parameters) (st : Storage) (initial final : State α) (v : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.C (cOffset p initial i)=stored st (v i)) ∧
  (∀ r o, (r≠p.C ∨ ∀ i, active p initial i → o≠cOffset p initial i) → final.mem r o=initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (st : Storage) (initial : State α)
    (hw : ∀ i j, active p initial i → active p initial j → cOffset p initial i=cOffset p initial j → i=j) :
    ∃ final, FP.Structural.exec (backend M D) (source p st) initial=some final ∧
      final.mem=(written M D p st initial).mem ∧ Post p st initial final (outputValue M D p st initial) := by
  let start := preludeExecution M D p st initial
  obtain ⟨mid,hm,k,hc,ha⟩ := loopExecution M D p st initial start.val
    (preludeContext M D p st initial).1 (preludeContext M D p st initial).2
  let finish := epilogueExecution M D p st initial mid k hc ha
  refine ⟨finish.val,?_,epilogueMemory M D p st initial mid k hc ha,?_,?_⟩
  · change run (backend M D) (source p st).surfaceBody initial=some finish.val
    rw [source_decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some]
    have hr : run (backend M D) [.forRangeDyn "k" (.constNat 0) (tripExpr p) (.constNat 1) (body p st)] start.val=some mid := by
      simpa [run,step,tripExpr,trips,evalOp_unfold,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex] using hm
    rw [hr]
    exact finish.property
  · intro i hi
    rw [epilogueMemory]
    exact masked_readback_active _ _ _ _ _ i hi hw
  · intro r o h
    rw [epilogueMemory]
    exact State.masked_scatter_frame _ _ _ _ _ _ _ (h.imp id (fun h i _ => h i)) _

#axiomsClean loopExecution
#axiomsClean epilogueExecution
#axiomsClean epilogueMemory
#axiomsClean correctness
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPValue
