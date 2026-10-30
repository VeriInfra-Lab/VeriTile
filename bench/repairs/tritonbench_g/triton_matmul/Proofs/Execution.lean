import bench.repairs.tritonbench_g.triton_matmul.Proofs.Model
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
open VeriTile.Triton FP.Structural
open FPValue
open ParallelAttention.FPStages (foldRange range_values)
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem min_form (a b : Nat) : min a b=if a<b then a else b := by
  by_cases h : a<b
  · rw [if_pos h,min_eq_left (Nat.le_of_lt h)]
  · rw [if_neg h,min_eq_right (Nat.le_of_not_gt h)]

theorem mask_iff (limit base lane : Nat) : lane<limit-base ↔ base+lane<limit := by omega

theorem clamp_add (limit base lane : Nat) :
    (if lane<limit-base then base else 0)+(if lane<limit-base then lane else 0)=
      (if base+lane<limit then base+lane else 0) := by
  split_ifs <;> omega

private theorem clamp_add' (limit base lane : Nat) :
    (if base+lane<limit then base else 0)+(if base+lane<limit then lane else 0)=
      (if base+lane<limit then base+lane else 0) := by
  by_cases h : base+lane<limit <;> simp [h]

structure Context {α : Type} (p : Parameters) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  pm : current.regs .nat [] "pid_m"=some (fun _ => pidM p initial)
  pn : current.regs .nat [] "pid_n"=some (fun _ => pidN p initial)
  offk : current.regs .nat [BK p] "offs_k"=some (fun i => i.1.val)
  am : current.regs .nat [BM p] "offs_am"=some (fun i => aRow p initial i.1)
  bn : current.regs .nat [BN p] "offs_bn"=some (fun i => bCol p initial i.1)
  mm : current.regs .bool [BM p] "mask_m"=some (fun i => decide (row p initial i.1<p.M))
  nn : current.regs .bool [BN p] "mask_n"=some (fun i => decide (col p initial i.1<p.N))

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    {t : State α // run (backend M D) (prelude p st) s=some t} := by
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [prelude,typed,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]
     rfl)

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (s : State α) :
    Context p s (preludeExecution M D p st s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p st s).val := by
  cases st <;> constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,blocksM,blocksN,perGroup,firstM,
    groupSize,pidM,pidN,row,col,aRow,bCol,min_form,HasAcc,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,clamp_add,mask_iff]
  all_goals try rfl
  all_goals funext i
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,mask_iff,←ite_add]
  all_goals exact clamp_add' _ _ _

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (body p st)
      (current.setReg "kk" .nat [] (fun _ => k))=some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [body,typed,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.offk,ctx.am,ctx.bn,ctx.mm,ctx.nn,ha,backend]
     rfl)

private theorem candidate_aValue {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) (k : Nat) (i : AIndex p) :
    (if i.2.1.val<p.K-k*BK p then
       FPValue.read st (s.mem p.A (aRow p s i.1*p.stride_am+
         ((if i.2.1.val<p.K-k*BK p then k*BK p else 0)+
          (if i.2.1.val<p.K-k*BK p then i.2.1.val else 0))*p.stride_ak))
     else M.cast none .real (dtype st) (M.literal none .real 0.0)) = aValue M p st s k i := by
  by_cases h : i.2.1.val<p.K-k*BK p
  · simp only [h,if_true,aValue]
    congr 2
    unfold aOffset
    ring
  · simp [aValue,h]

private theorem candidate_bValue {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) (k : Nat) (i : BIndex p) :
    (if i.1.val<p.K-k*BK p then
       FPValue.read st (s.mem p.B (((if i.1.val<p.K-k*BK p then k*BK p else 0)+
          (if i.1.val<p.K-k*BK p then i.1.val else 0))*p.stride_bk+
         bCol p s i.2.1*p.stride_bn))
     else M.cast none .real (dtype st) (M.literal none .real 0.0)) = bValue M p st s k i := by
  by_cases h : i.1.val<p.K-k*BK p
  · simp only [h,if_true,bValue]
    congr 2
    unfold bOffset
    ring
  · simp [bValue,h]

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    let t := (bodyExecution M D p st initial current k a ctx ha).val
    Context p initial t ∧ HasAcc p (update M D p st initial k a) t := by
  cases st <;> constructor
  all_goals try (constructor <;> simp [bodyExecution,ctx.memory,ctx.pm,ctx.pn,ctx.offk,ctx.am,ctx.bn,ctx.mm,ctx.nn,
      bop,Broadcast.leftIndex,Broadcast.rightIndex])
  all_goals simp [HasAcc,bodyExecution,update,dtype,backend,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals apply congrArg₂ (fun (av : AIndex p → α) (bv : BIndex p → α) =>
      D.apply (some .fp32) _ _ .real {} (batch := []) av bv a)
  all_goals funext i
  · simpa [FPValue.read,dtype] using candidate_aValue M p .half initial k i
  · simpa [FPValue.read,dtype] using candidate_bValue M p .half initial k i
  · simpa [FPValue.read,dtype] using candidate_aValue M p .e4m3 initial k i
  · simpa [FPValue.read,dtype] using candidate_bValue M p .e4m3 initial k i

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current)
    (ha : HasAcc p (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, FP.Structural.range (backend M D) "kk" 0 p.numKBlocks 1 (body p st) current=some t ∧
      Context p initial t ∧ HasAcc p (accumulation M D p st initial) t := by
  let I := fun (_ : Nat) (a : CIndex p → α) (s : State α) => Context p initial s ∧ HasAcc p a s
  exact range_values (backend M D) "kk" (body p st) (update M D p st initial)
    I (fun a s => Context p initial s ∧ HasAcc p a s) 0 p.numKBlocks 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p st initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p st initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulation M D p st initial) current) :
    {t : State α // run (backend M D) (epilogue p st) current=some t} := by
  simp only [HasAcc] at ha
  cases st <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [epilogue,typed,dtype,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLe,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.am,ctx.bn,ctx.mm,ctx.nn,ha,backend]
     rfl)

def cSafeOffset {α : Type} (p : Parameters) (initial : State α) (i : CIndex p) :=
  p.stride_cm*(if row p initial i.1<p.M then row p initial i.1 else 0) +
  p.stride_cn*(if col p initial i.2.1<p.N then col p initial i.2.1 else 0)

def writtenSafe {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cSafeOffset p initial i) (stored st (outputValue M D p st initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulation M D p st initial) current) :
    (epilogueExecution M D p st initial current ctx ha).val.mem=(writtenSafe M D p st initial).mem := by
  cases st <;> simp [epilogueExecution,writtenSafe,masked_scatter_memory,ctx.memory,
    active,cSafeOffset,aRow,bCol,row,col,outputValue,stored,dtype,backend,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]

theorem writtenSafe_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) : writtenSafe M D p st initial=written M D p st initial := by
  have stepEq (t : State α) (i : CIndex p) :
      (if active p initial i then t.write p.C (cSafeOffset p initial i)
        (stored st (outputValue M D p st initial i)) else t) =
      (if active p initial i then t.write p.C (cOffset p initial i)
        (stored st (outputValue M D p st initial i)) else t) := by
    by_cases h : active p initial i
    · simp [h,cSafeOffset,cOffset,h.1,h.2]
    · simp [h]
  unfold writtenSafe written
  simp_rw [stepEq]

/-- Exact new register sequence executes successfully and has the same full
memory transformer as the unchanged source. Floating primitives stay identical. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (st : Storage) (initial : State α) :
    ∃ final, FP.Structural.exec (backend M D) (typed p st) initial=some final ∧
      final.mem=(written M D p st initial).mem := by
  let start := preludeExecution M D p st initial
  obtain ⟨mid,hm,hc,ha⟩ := loopExecution M D p st initial start.val
    (preludeContext M D p st initial).1 (preludeContext M D p st initial).2
  let finish := epilogueExecution M D p st initial mid hc ha
  refine ⟨finish.val,?_,?_⟩
  · change run (backend M D) (typed p st).surfaceBody initial=some finish.val
    rw [decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some]
    have hr : run (backend M D) [.forRange "kk" 0 p.numKBlocks 1 (body p st)] start.val=some mid := by
      simpa [run,step] using hm
    rw [hr]
    exact finish.property
  · rw [epilogueMemory,writtenSafe_eq]

#axiomsClean execution_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Source
