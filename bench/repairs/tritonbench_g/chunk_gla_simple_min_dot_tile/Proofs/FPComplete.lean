import bench.repairs.tritonbench_g.chunk_gla_simple_min_dot_tile.Proofs.FPExecution

namespace VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.Host
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

theorem loopExecution {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial current : State α) (ctx : Context c initial current)
    (ha : HasAcc c (fun _ => zero M,fun _ => zero M) current) :
    ∃ t, FP.Structural.range (engine M) "i_k" 0 ((c.K+c.BK-1)/c.BK) 1 (Typed.parts c scale).body current=some t ∧
      Context c initial t ∧ HasAcc c (accumulated M c d initial) t := by
  let I := fun (_ : Nat) (a : Acc α c) (s : State α) => Context c initial s ∧ HasAcc c a s
  exact range_values (engine M) "i_k" (Typed.parts c scale).body (update M c d initial)
    I (fun a s => Context c initial s ∧ HasAcc c a s) 0 ((c.K+c.BK-1)/c.BK) 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M c scale d initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M c scale d initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def finishExecution {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial current : State α) (ctx : Context c initial current)
    (ha : HasAcc c (accumulated M c d initial) current) :
    {t : State α // run (engine M) (Typed.parts c scale).after current=some t} := by
  have ho := ha.1
  have hs := ha.2
  refine ⟨?_,?_⟩
  swap
  simp [Typed.parts,kernel,ChunkGlaSimpleMinDotTile.source,ComputeKernel.surfaceBody,Source.dynamicRange?,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.bh,ctx.it,ctx.rows,ctx.cols,ctx.mask,ctx.memory,ho,hs,d.sv_eq,FP.Structural.store]
  rfl

def written {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) : State α :=
  (TileShape.allIndices [c.BT,c.BV]).foldl (fun t i => if active c initial i then
    t.write "o" (oOffset c initial i) (.mk .fp32 (outputValue M c d initial scale i)) else t) initial

theorem finishMemory {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial current : State α) (ctx : Context c initial current)
    (ha : HasAcc c (accumulated M c d initial) current) :
    (finishExecution M c scale d initial current ctx ha).val.mem=(written M c scale d initial).mem := by
  simp [finishExecution,written,masked_scatter_memory,ctx.memory,active,oOffset,row,column,
    outputValue,gatedO,gatedS,vValue,gValue,gOffset,zero,FP.Structural.bop,
    Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

def Post {α : Type} (c : Shape) (initial final : State α) (v : OIndex c → α) : Prop :=
  (∀ i, active c initial i → final.mem "o" (oOffset c initial i)=.mk .fp32 (v i)) ∧
  (∀ (r : RegionName) o, (r≠"o" ∨ ∀ i, active c initial i → o≠oOffset c initial i) → final.mem r o=initial.mem r o)

theorem readback {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) (i : OIndex c) (hi : active c initial i)
    (hw : ∀ i j, active c initial i → active c initial j → oOffset c initial i=oOffset c initial j → i=j) :
    (written M c scale d initial).mem "o" (oOffset c initial i)=.mk .fp32 (outputValue M c d initial scale i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) (r : RegionName) (off : Nat)
    (h : r≠"o" ∨ ∀ i, active c initial i → off≠oOffset c initial i) :
    (written M c scale d initial).mem r off=initial.mem r off := by
  apply State.masked_scatter_frame
  rcases h with hr | ho
  · exact Or.inl hr
  · right
    intro i _ hi
    exact ho i hi

theorem source_memory {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α) :
    ∃ final, FP.Structural.exec (engine M) (kernel c .fp32 .fp32 .fp32 .fp32 scale) initial=some final ∧
      final.mem=(written M c scale d initial).mem := by
  let start := entryExecution M c scale initial
  obtain ⟨mid,hm,hc,ha⟩ := loopExecution M c scale d initial start.val
    (entryContext M c scale initial).1 (entryContext M c scale initial).2
  let finish := finishExecution M c scale d initial mid hc ha
  refine ⟨finish.val,?_,finishMemory M c scale d initial mid hc ha⟩
  have hl : step (engine M) (.forRangeDyn "i_k" (Typed.parts c scale).start (Typed.parts c scale).stop
      (Typed.parts c scale).stride (Typed.parts c scale).body) start.val = some mid := by
    have ha : (Typed.parts c scale).start = .constNat 0 := rfl
    have hb : (Typed.parts c scale).stop = .div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1)) (.constNat c.BK) := rfl
    have hd : (Typed.parts c scale).stride = .constNat 1 := rfl
    simp only [step,ha,hb,hd,evalOp_unfold,numeric,bop,Option.bind,Option.bind_some]
    exact hm
  change run (engine M) (kernel c .fp32 .fp32 .fp32 .fp32 scale).surfaceBody initial = _
  rw [Typed.source_parts]
  simp only [List.append_assoc,run_append,start.property,Option.bind_some,List.singleton_append]
  rw [run]
  simp only [hl,bind,Option.bind_some]
  exact finish.property

theorem correctness {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (scale : ℝ)
    (d : Typed.Dots M c) (initial : State α)
    (hw : ∀ i j, active c initial i → active c initial j → oOffset c initial i=oOffset c initial j → i=j) :
    ∃ final, FP.Structural.exec (engine M) (kernel c .fp32 .fp32 .fp32 .fp32 scale) initial=some final ∧
      Post c initial final (outputValue M c d initial scale) := by
  obtain ⟨final,he,hm⟩ := source_memory M c scale d initial
  refine ⟨final,he,?_,?_⟩
  · intro i hi
    rw [hm]
    exact readback M c scale d initial i hi hw
  · intro r o h
    rw [hm]
    exact frame M c scale d initial r o h

#axiomsClean correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.FPValue
