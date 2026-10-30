import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Source
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Addresses

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
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

structure Context {α : Type} (p : Parameters) (initial current : State α) : Prop where
  memory : current.mem=initial.mem
  pm : current.regs .nat [] "pid_m"=some (fun _ => pidM p initial)
  pn : current.regs .nat [] "pid_n"=some (fun _ => pidN p initial)
  offk : current.regs .nat [BK p] "offs_k"=some (fun i => i.1.val)
  am : current.regs .nat [BM p] "offs_am"=some (fun i => aRow p initial i.1)
  bn : current.regs .nat [BN p] "offs_bn"=some (fun i => bCol p initial i.1)

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) :
    {t : State α // run (backend M D) (prelude p) s=some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [prelude,kernel,typed,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]
  rfl

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (s : State α) :
    Context p s (preludeExecution M D p s).val ∧
      HasAcc p (fun _ => M.literal (some .fp32) .real 0) (preludeExecution M D p s).val := by
  constructor
  all_goals try constructor
  all_goals simp [preludeExecution,backend,blocksM,blocksN,perGroup,firstM,
    groupSize,pidM,pidN,row,col,aRow,bCol,min_form,HasAcc,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals try rfl

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    {t : State α // run (backend M D) (body p)
      (current.setReg "kk" .nat [] (fun _ => k))=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_, ?_⟩
  swap
  simp [body,kernel,typed,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
    ctx.memory,ctx.offk,ctx.am,ctx.bn,ha,backend]
  rfl

private theorem candidate_aValue {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α) (k : Nat) (i : AIndex p) :
    (if k*BK p+i.2.1.val<p.K then
       (s.mem p.A (aRow p s i.1*p.stride_am+
         (if k*BK p+i.2.1.val<p.K then k*BK p+i.2.1.val else 0)*p.stride_ak)).read .fp32
     else M.cast none .real .fp32 (M.literal none .real 0.0)) = aValue M p .single s k i := by
  have hm : (k*BK p+i.2.1.val<p.K) ↔ i.2.1.val<p.K-k*BK p := by omega
  by_cases h : k*BK p+i.2.1.val<p.K
  · have h' := hm.mp h
    simp only [h,if_true,aValue,h',FPValue.read,dtype]
    congr 2
    unfold aOffset
    ring
  · have h' : ¬i.2.1.val<p.K-k*BK p := fun h' => h (hm.mpr h')
    simp [aValue,h,h',dtype]

private theorem candidate_bValue {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α) (k : Nat) (i : BIndex p) :
    (if k*BK p+i.1.val<p.K then
       (s.mem p.B ((if k*BK p+i.1.val<p.K then k*BK p+i.1.val else 0)*p.stride_bk+
         bCol p s i.2.1*p.stride_bn)).read .fp32
     else M.cast none .real .fp32 (M.literal none .real 0.0)) = bValue M p .single s k i := by
  have hm : (k*BK p+i.1.val<p.K) ↔ i.1.val<p.K-k*BK p := by omega
  by_cases h : k*BK p+i.1.val<p.K
  · have h' := hm.mp h
    simp only [h,if_true,bValue,h',FPValue.read,dtype]
    congr 2
    unfold bOffset
    ring
  · have h' : ¬i.1.val<p.K-k*BK p := fun h' => h (hm.mpr h')
    simp [bValue,h,h',dtype]

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current) :
    let t := (bodyExecution M D p initial current k a ctx ha).val
    Context p initial t ∧ HasAcc p (update M D p .single initial k a) t := by
  constructor
  · constructor
    all_goals simp [bodyExecution,ctx.memory,ctx.pm,ctx.pn,ctx.offk,ctx.am,ctx.bn,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  · simp [HasAcc,bodyExecution,update,dtype,backend,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
    apply congrArg₂ (fun (av : AIndex p → α) (bv : BIndex p → α) =>
      D.apply (some .fp32) .fp32 .fp32 .real {} (batch := []) av bv a)
    · funext i
      simpa only [ite_mul,zero_mul] using candidate_aValue M p initial k i
    · funext i
      simpa only [ite_mul,zero_mul] using candidate_bValue M p initial k i

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α)
    (ctx : Context p initial current)
    (ha : HasAcc p (fun _ => M.literal (some .fp32) .real 0) current) :
    ∃ t, FP.Structural.range (backend M D) "kk" 0 p.numKBlocks 1 (body p) current=some t ∧
      Context p initial t ∧ HasAcc p (accumulation M D p .single initial) t := by
  let I := fun (_ : Nat) (a : CIndex p → α) (s : State α) => Context p initial s ∧ HasAcc p a s
  exact range_values (backend M D) "kk" (body p) (update M D p .single initial)
    I (fun a s => Context p initial s ∧ HasAcc p a s) 0 p.numKBlocks 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D p initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D p initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulation M D p .single initial) current) :
    {t : State α // run (backend M D) (epilogue p) current=some t} := by
  simp only [HasAcc] at ha
  cases he : p.ACTIVATION <;>
    (refine ⟨?_, ?_⟩
     swap
     simp [epilogue,kernel,typed,ComputeKernel.surfaceBody,run,step,evalExpr,
       evalComputeOp,evalOp_unfold,evalOffsets,store,numeric,bop,natLt,numericEq,numericLe,
       ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
       FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
       ctx.memory,ctx.pm,ctx.pn,ha,backend,he]
     rfl)

def cSafeOffset {α : Type} (p : Parameters) (initial : State α) (i : CIndex p) :=
  p.stride_cm*(if row p initial i.1<p.M then row p initial i.1 else 0) +
  p.stride_cn*(if col p initial i.2.1<p.N then col p initial i.2.1 else 0)

def writtenSafe {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) : State α :=
  (TileShape.allIndices [BM p,BN p]).foldl (fun t i => if active p initial i then
    t.write p.C (cSafeOffset p initial i) (stored .single (outputValue M D p .single initial i)) else t) initial

theorem epilogueMemory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulation M D p .single initial) current) :
    (epilogueExecution M D p initial current ctx ha).val.mem=(writtenSafe M D p initial).mem := by
  cases p
  rename_i enabled
  cases enabled <;> simp [epilogueExecution,writtenSafe,masked_scatter_memory,ctx.memory,
    active,cSafeOffset,row,col,outputValue,activation,stored,dtype,backend,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

theorem writtenSafe_eq {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) : writtenSafe M D p initial=written M D p .single initial := by
  have stepEq (t : State α) (i : CIndex p) :
      (if active p initial i then t.write p.C (cSafeOffset p initial i)
        (stored .single (outputValue M D p .single initial i)) else t) =
      (if active p initial i then t.write p.C (cOffset p initial i)
        (stored .single (outputValue M D p .single initial i)) else t) := by
    by_cases h : active p initial i
    · simp [h,cSafeOffset,cOffset,h.1,h.2]
    · simp [h]
  unfold writtenSafe written
  simp_rw [stepEq]

/-- Exact new register sequence executes successfully and has the same full
memory transformer as the unchanged source. Floating primitives stay identical. -/
theorem execution_memory {α : Type} [Inhabited α] (M : Algebra α) (D : DotAcc α)
    (p : Parameters) (initial : State α) :
    ∃ final, FP.Structural.exec (backend M D) (kernel p) initial=some final ∧
      final.mem=(written M D p .single initial).mem := by
  let start := preludeExecution M D p initial
  obtain ⟨mid,hm,hc,ha⟩ := loopExecution M D p initial start.val
    (preludeContext M D p initial).1 (preludeContext M D p initial).2
  let finish := epilogueExecution M D p initial mid hc ha
  refine ⟨finish.val,?_,?_⟩
  · change run (backend M D) (kernel p).surfaceBody initial=some finish.val
    rw [decomposition,run_append,run_append,start.property]
    simp only [Option.bind_some]
    have hr : run (backend M D) [.forRange "kk" 0 p.numKBlocks 1 (body p)] start.val=some mid := by
      simpa [run,step] using hm
    rw [hr]
    exact finish.property
  · rw [epilogueMemory,writtenSafe_eq]

#axiomsClean execution_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
