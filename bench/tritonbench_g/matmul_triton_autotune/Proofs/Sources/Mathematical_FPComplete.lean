import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_FPExecution
import bench.tritonbench_g.matmul_triton_autotune.Proofs.Sources.Mathematical_FPRange
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (foldRange range_values)
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial 0 current)
    (ha : HasAcc p (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, FP.Structural.rangeInt (backend M D) "k" 0 (p.numKBlocks : Int) 1 (body p st) current=some t ∧
      ∃ k, Context p initial k t ∧ HasAcc p (accumulation M D p st initial) t := by
  let I := fun n (a : CIndex p → α) (s : State α) => Context p initial n s ∧ HasAcc p a s
  exact nat_range_values (backend M D) "k" (body p st) (update M D p st initial)
    I (fun a s => ∃ k, Context p initial k s ∧ HasAcc p a s) 0 p.numKBlocks 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p st initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p st initial s k a h.1 h.2⟩)
    (by intro n a s h; exact ⟨n,h⟩) ⟨ctx,ha⟩

theorem epilogue_profiles (p : Parameters) (st : Storage) : epilogue p st=epilogue p .single := by
  cases st <;> rfl

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p st initial) current) :
    {t : State α // run (backend M D) (epilogue p st) current=some t} := by
  simp only [HasAcc] at ha
  cases he : p.ACTIVATION <;>
    (refine ⟨?_, ?_⟩
     swap
     rw [epilogue_profiles]
     simp [epilogue,source,typed,Pasted.originalKernel,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericLt,numericEq,numericLe,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.pm,ctx.pn,ha,backend,he]
     rfl)

def written {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cOffset p initial i) (stored st (outputValue M D p st initial i)) else t) initial

private theorem ite_natCast_zero (P : Prop) [Decidable P] (n : Nat) :
    (if P then (n : Int) else 0)=((if P then n else 0 : Nat) : Int) := by
  by_cases h:P <;> simp only [h,↓reduceIte,Int.natCast_zero]

private theorem fold_conditional_offsets {α ι : Type}
    (indices : List ι) (region : RegionName) (rows cols sm sn : Nat)
    (row col : ι→Nat) (value : ι→Cell α) (memory : RegionName→Nat→Cell α) :
    indices.foldl (fun mem i => if row i<rows ∧ col i<cols then
      fun r o => if r=region ∧ o=(if row i<rows then sm*row i else 0)+(if col i<cols then sn*col i else 0)
        then value i else mem r o else mem) memory =
    indices.foldl (fun mem i => if row i<rows ∧ col i<cols then
      fun r o => if r=region ∧ o=sm*row i+sn*col i then value i else mem r o else mem) memory := by
  apply congrArg (fun f=>indices.foldl f memory)
  funext mem i
  by_cases hr:row i<rows <;> by_cases hc:col i<cols <;> simp [hr,hc]

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p st initial) current) :
    (epilogueExecution M D p st initial current k ctx ha).val.mem=(written M D p st initial).mem := by
  cases p
  rename_i enabled
  cases enabled <;> simp [epilogueExecution,written,masked_scatter_memory,ctx.memory,
    active,cOffset,row,col,outputValue,activation,stored,dtype,backend,bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simp only [←Int.natCast_add,←Int.natCast_mul,Int.toNat_natCast,Int.ofNat_lt,ite_natCast_zero]
  all_goals try simp only [show (0.0 : ℝ)=0 by norm_num]
  all_goals apply fold_conditional_offsets

def Post {α : Type} (p : Parameters) (st : Storage) (initial final : State α) (v : CIndex p → α) : Prop :=
  (∀ i, active p initial i → final.mem p.C (cOffset p initial i)=stored st (v i)) ∧
  (∀ r o, (r≠p.C ∨ ∀ i, active p initial i → o≠cOffset p initial i) → final.mem r o=initial.mem r o)

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α)
    (geo : Geometry p initial)
    (hw : ∀ i j, active p initial i → active p initial j → cOffset p initial i=cOffset p initial j → i=j) :
    ∃ final, FP.Structural.exec (backend M D) (source p st) initial=some final ∧
      final.mem=(written M D p st initial).mem ∧ Post p st initial final (outputValue M D p st initial) := by
  let start := preludeExecution M D p st initial
  obtain ⟨mid,hm,k,hc,ha⟩ := loopExecution M D p st initial start.val
    (preludeContext M D p st initial geo).1 (preludeContext M D p st initial geo).2
  let finish := epilogueExecution M D p st initial mid k hc ha
  refine ⟨finish.val,?_,epilogueMemory M D p st initial mid k hc ha,?_,?_⟩
  · change run (backend M D) (source p st).surfaceBody initial=some finish.val
    rw [source_decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some]
    have hr : run (backend M D) [loopStatement p st] start.val=some mid := by
      rw [loop_run,loopStop_value]
      exact hm
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
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.FPValue
