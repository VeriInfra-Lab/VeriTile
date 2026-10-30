import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPMerge
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPMerge
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

theorem loopExecution (M : Algebra α) (p : Layout) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (fun _ => zero M) current) :
    ∃ t, range (engine M) "i_k" 0 p.NK 1 (parts p).body current=some t ∧
      Context p initial t ∧ HasAcc p (accumulated M p initial) t := by
  let I := fun (_ : Nat) (a : Index p → α) (s : State α) => Context p initial s ∧ HasAcc p a s
  exact range_values (engine M) "i_k" (parts p).body (update M p initial)
    I (fun a s => Context p initial s ∧ HasAcc p a s) 0 p.NK 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M p initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M p initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def finishExecution (M : Algebra α) (p : Layout) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulated M p initial) current) :
    {t : State α // run (engine M) (parts p).after current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.it,ctx.ic,ctx.bh,ctx.nh,ha,FP.Structural.store]
  rfl

def written (M : Algebra α) (p : Layout) (s : State α) : State α :=
  (TileShape.allIndices [p.BC,p.BC]).foldl (fun t i =>
    if active p s i then t.write "attention" (outputAddress p s i) (.mk .fp32 (value M p s i)) else t) s

theorem finishMemory (M : Algebra α) (p : Layout) (initial current : State α)
    (ctx : Context p initial current) (ha : HasAcc p (accumulated M p initial) current) :
    (finishExecution M p initial current ctx ha).val.mem=(written M p initial).mem := by
  simp [finishExecution,written,masked_scatter_memory,ctx.memory,active,outputAddress,row,column,
    value,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

def prologue (M : Algebra α) (p : Layout) (s : State α) :
    {t : State α // run (engine M) (branch p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem condition_eval (M : Algebra α) (p : Layout) (s : State α) :
    FP.Structural.evalOp (engine M) none (branch p).condition (prologue M p s).val =
      some (fun _ => decide (enabled p s)) := by
  simp [branch,kernel,TypedMerge.original,ComputeKernel.surfaceBody,Source.conditional?,
    evalOp_unfold,numeric,bop,prologue,natLe,natLt,enabled,row]
  funext i
  by_cases h : s.pids 0*p.BT+s.pids 1*p.BC < p.T
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem source_parts (p : Layout) : (kernel p).surfaceBody = (branch p).before ++
    [.ifThen (branch p).condition ((parts p).before ++
      [.forRangeDyn "i_k" (parts p).start (parts p).stop (parts p).stride (parts p).body] ++ (parts p).after)] := rfl

/-- Actual typed execution has exactly the independently defined FP32 carry
recurrence and masked stores; this holds for every scalar algebra. -/
theorem source_memory (M : Algebra α) (p : Layout) (s : State α) :
    ∃ t, FP.Structural.exec (engine M) (kernel p) s=some t ∧
      t.mem=(if enabled p s then written M p s else s).mem := by
  let start := entryExecution M p s
  have hs : run (engine M) (parts p).before (prologue M p s).val=some start.val := by
    simpa only [run_append,(prologue M p s).property,Option.bind_some] using start.property
  change ∃ t, run (engine M) (kernel p).surfaceBody s=some t ∧ _
  rw [source_parts,run_append,(prologue M p s).property]
  simp only [Option.bind_some,run,step,condition_eval]
  by_cases hen : enabled p s
  · simp only [hen,decide_true,↓reduceIte]
    obtain ⟨mid,hm,hc,ha⟩ := loopExecution M p s start.val
      (entryContext M p s).1 (entryContext M p s).2
    let finish := finishExecution M p s mid hc ha
    refine ⟨finish.val,?_,finishMemory M p s mid hc ha⟩
    have hl : step (engine M) (.forRangeDyn "i_k" (parts p).start (parts p).stop
        (parts p).stride (parts p).body) start.val=some mid := by
      have h0 : (parts p).start = .constNat 0 := rfl
      have hn : (parts p).stop = .constNat p.NK := rfl
      have h1 : (parts p).stride = .constNat 1 := rfl
      simp only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some]
      exact hm
    rw [List.append_assoc,run_append,hs]
    simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property,↓reduceIte]
  · simp only [hen,decide_false,Bool.false_eq_true,↓reduceIte,bind,Option.bind_some]
    refine ⟨(prologue M p s).val,rfl,?_⟩
    simp [prologue]

theorem readback (M : Algebra α) (p : Layout) (s : State α) (i : Index p)
    (hi : active p s i)
    (hw : ∀ i j, active p s i → active p s j → outputAddress p s i=outputAddress p s j → i=j) :
    (written M p s).mem "attention" (outputAddress p s i)=.mk .fp32 (value M p s i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem frame (M : Algebra α) (p : Layout) (s : State α) (r : RegionName) (off : Nat)
    (h : r≠"attention" ∨ ∀ i, active p s i → off≠outputAddress p s i) :
    (written M p s).mem r off=s.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

#specNonCircular value avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPMerge
