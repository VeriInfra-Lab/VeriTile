import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPInterValues
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPInter
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
open ProducerGeometry (Shape)
open FPOutput (DotPrimitive withDot)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def enabled (c : Shape) (s : State α) : Prop := rowStart c s<c.T ∧ s.pids 1%c.NC<s.pids 1/c.NC
def active (c : Shape) (s : State α) (i : Index) : Prop := row c s i.1.val<c.T ∧ column c s i.2.1.val<c.BT
instance (c : Shape) (s : State α) : Decidable (enabled c s) := inferInstanceAs (Decidable (_∧_))
instance (c : Shape) (s : State α) : DecidablePred (active c s) := fun _ => inferInstanceAs (Decidable (_∧_))
def outputAddress (c : Shape) (s : State α) (i : Index) := s.pids 2*c.T*c.BT+row c s i.1.val*c.BT+column c s i.2.1.val

theorem loopExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context c initial current) (ha : HasAcc (fun _ => zero M) current) :
    ∃ t, range (engine (withDot M D)) "i_k" 0 (Geometry.ceilDiv c.K c.BK) 1 (parts c scale).body current=some t ∧
      Context c initial t ∧ HasAcc (accumulated M D c scale initial) t := by
  let I := fun (_ : Nat) (a : Index → α) (s : State α) => Context c initial s ∧ HasAcc a s
  exact range_values (engine (withDot M D)) "i_k" (parts c scale).body (update M D c scale initial)
    I (fun a s => Context c initial s ∧ HasAcc a s) 0 (Geometry.ceilDiv c.K c.BK) 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D c scale initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D c scale initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def finishExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context c initial current) (ha : HasAcc (accumulated M D c scale initial) current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).after current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,TypedInter.fp32Original,ProducerGeometry.layout,ProducerGeometry.regions,
    ComputeKernel.surfaceBody,Source.conditional?,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.it,ctx.ic,ctx.bh,ctx.ii,ctx.ij,ha,FP.Structural.store]
  rfl

def written (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) : State α :=
  (TileShape.allIndices [16,16]).foldl (fun t i =>
    if active c s i then t.write "attention" (outputAddress c s i) (.mk .fp32 (value M D c scale s i)) else t) s

theorem finishMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context c initial current) (ha : HasAcc (accumulated M D c scale initial) current) :
    (finishExecution M D c scale initial current ctx ha).val.mem=(written M D c scale initial).mem := by
  simp [finishExecution,written,masked_scatter_memory,ctx.memory,active,outputAddress,row,column,rowStart,
    value,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

def prologue (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    {t : State α // run (engine (withDot M D)) (outer c scale).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,TypedInter.fp32Original,ProducerGeometry.layout,ProducerGeometry.regions,
    ComputeKernel.surfaceBody,Source.conditional?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem outer_condition (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    FP.Structural.evalOp (engine (withDot M D)) none (outer c scale).condition (prologue M D c scale s).val =
      some (fun _ => decide (rowStart c s<c.T)) := by
  simp [outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,TypedInter.fp32Original,ProducerGeometry.layout,ProducerGeometry.regions,
    ComputeKernel.surfaceBody,Source.conditional?,evalOp_unfold,numeric,bop,prologue,natLe,natLt,rowStart]
  funext i
  by_cases h : s.pids 0*c.BT+s.pids 1/c.NC*16<c.T
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem inner_condition (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    FP.Structural.evalOp (engine (withDot M D)) none (inner c scale).condition (prologue M D c scale s).val =
      some (fun _ => decide (s.pids 1%c.NC<s.pids 1/c.NC)) := by
  simp [inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,TypedInter.fp32Original,ProducerGeometry.layout,ProducerGeometry.regions,
    ComputeKernel.surfaceBody,Source.conditional?,evalOp_unfold,numeric,bop,prologue,natLe,natLt,numericLe]
  funext i
  by_cases h : s.pids 1%c.NC<s.pids 1/c.NC
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem source_parts (c : Shape) (scale : ℝ) : (kernel c scale).surfaceBody = (outer c scale).before ++
    [.ifThen (outer c scale).condition [.ifThen (inner c scale).condition ((parts c scale).before ++
      [.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop (parts c scale).stride (parts c scale).body] ++
      (parts c scale).after)]] := rfl

theorem source_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel c scale) s=some t ∧
      t.mem=(if enabled c s then written M D c scale s else s).mem := by
  let start := entryExecution M D c scale s
  have hs : run (engine (withDot M D)) (parts c scale).before (prologue M D c scale s).val=some start.val := by
    simpa only [run_append,(prologue M D c scale s).property,Option.bind_some] using start.property
  change ∃ t, run (engine (withDot M D)) (kernel c scale).surfaceBody s=some t ∧ _
  rw [source_parts,run_append,(prologue M D c scale s).property]
  simp only [Option.bind_some,run,step,outer_condition]
  by_cases ht : rowStart c s<c.T
  · simp only [ht,decide_true,↓reduceIte,inner_condition]
    by_cases hl : s.pids 1%c.NC<s.pids 1/c.NC
    · simp only [hl,decide_true,↓reduceIte,enabled,ht,and_self,ite_true]
      obtain ⟨mid,hm,hc,ha⟩ := loopExecution M D c scale s start.val
        (entryContext M D c scale s).1 (entryContext M D c scale s).2
      let finish := finishExecution M D c scale s mid hc ha
      refine ⟨finish.val,?_,finishMemory M D c scale s mid hc ha⟩
      have hloop : step (engine (withDot M D)) (.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop
          (parts c scale).stride (parts c scale).body) start.val=some mid := by
        have h0 : (parts c scale).start = .constNat 0 := rfl
        have hn : (parts c scale).stop = .div .nat .nil
          (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1)) (.constNat c.BK) := rfl
        have h1 : (parts c scale).stride = .constNat 1 := rfl
        simp only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop]
        exact hm
      rw [List.append_assoc,run_append,hs]
      simp only [Option.bind_some,List.singleton_append,run,hloop,bind,finish.property,↓reduceIte]
    · simp only [hl,decide_false,Bool.false_eq_true,↓reduceIte,enabled,ht,true_and,ite_false,bind,Option.bind_some]
      exact ⟨(prologue M D c scale s).val,rfl,by simp [prologue]⟩
  · simp only [ht,decide_false,Bool.false_eq_true,↓reduceIte,enabled,false_and,ite_false,bind,Option.bind_some]
    exact ⟨(prologue M D c scale s).val,rfl,by simp [prologue]⟩

theorem readback (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (i : Index)
    (hi : active c s i)
    (hw : ∀ i j, active c s i → active c s j → outputAddress c s i=outputAddress c s j → i=j) :
    (written M D c scale s).mem "attention" (outputAddress c s i)=.mk .fp32 (value M D c scale s i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem frame (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (r : RegionName) (off : Nat)
    (h : r≠"attention" ∨ ∀ i, active c s i → off≠outputAddress c s i) :
    (written M D c scale s).mem r off=s.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPInter
