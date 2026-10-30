import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPOutputValues
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def active (c : OutputGrid.Shape) (s : State α) (i : Index c) : Prop :=
  row c s i.1.val<c.T ∧ column c s i.2.1.val<c.V
instance (c : OutputGrid.Shape) (s : State α) : DecidablePred (active c s) :=
  fun _ => inferInstanceAs (Decidable (_∧_))
def outputAddress (c : OutputGrid.Shape) (s : State α) (i : Index c) :=
  s.pids 2*(c.T*c.V)+row c s i.1.val*c.V+column c s i.2.1.val

def attentionLoad (M : Algebra α) (c : OutputGrid.Shape) (s : State α)
    (i : TileIndex [c.BT,c.BT]) : α :=
  M.cast (some .fp32) .real .real
    (if i.2.1.val ≤ i.1.val then
      M.fp32Load (if row c s i.1.val<c.T then
        (s.mem "attention" (s.pids 2*c.T*c.BT+row c s i.1.val*c.BT+i.2.1.val)).read .real else zero M)
    else zero M)
def valueLoad (M : Algebra α) (c : OutputGrid.Shape) (s : State α) (i : Index c) : α :=
  M.fp32Load (if active c s i then (s.mem "v" (outputAddress c s i)).read .real else zero M)
def value (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) (i : Index c) : α :=
  M.cast (some .fp32) .real .fp32 (M.binary (some .fp32) .real .add (accumulated M D c scale s i)
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (attentionLoad M c s) (valueLoad M c s) i))

theorem loopExecution (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context c initial current) (ha : HasAcc c (fun _ => zero M) current) :
    ∃ t, range (engine (withDot M D)) "i_k" 0 (Geometry.ceilDiv c.K c.BK) 1 (parts c scale).body current=some t ∧
      Context c initial t ∧ HasAcc c (accumulated M D c scale initial) t := by
  let I := fun (_ : Nat) (a : Index c → α) (s : State α) => Context c initial s ∧ HasAcc c a s
  exact range_values (engine (withDot M D)) "i_k" (parts c scale).body (update M D c scale initial)
    I (fun a s => Context c initial s ∧ HasAcc c a s) 0 (Geometry.ceilDiv c.K c.BK) 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D c scale initial s k a h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D c scale initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

def finishExecution (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context c initial current) (ha : HasAcc c (accumulated M D c scale initial) current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).after current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,TypedOutput.fp32Original,OutputGrid.layout,OutputGrid.regions,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.causal,ha,FP.Structural.store]
  rfl

def written (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) : State α :=
  (TileShape.allIndices [c.BT,c.BV]).foldl (fun t i =>
    if active c s i then t.write "o" (outputAddress c s i) (.mk .fp32 (value M D c scale s i)) else t) s

theorem finishMemory (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context c initial current) (ha : HasAcc c (accumulated M D c scale initial) current) :
    (finishExecution M D c scale initial current ctx ha).val.mem=(written M D c scale initial).mem := by
  simp [finishExecution,written,masked_scatter_memory,ctx.memory,active,outputAddress,row,column,
    value,attentionLoad,valueLoad,zero,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    try dsimp [attentionLoad,valueLoad,active,outputAddress,row,column,zero]
    norm_num only
  all_goals rfl

theorem source_parts (c : OutputGrid.Shape) (scale : ℝ) : (kernel c scale).surfaceBody = (parts c scale).before ++
    [.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop (parts c scale).stride (parts c scale).body] ++
    (parts c scale).after := rfl

theorem source_memory (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel c scale) s=some t ∧
      t.mem=(written M D c scale s).mem := by
  let start := entryExecution M D c scale s
  obtain ⟨mid,hm,hc,ha⟩ := loopExecution M D c scale s start.val
    (entryContext M D c scale s).1 (entryContext M D c scale s).2
  let finish := finishExecution M D c scale s mid hc ha
  refine ⟨finish.val,?_,finishMemory M D c scale s mid hc ha⟩
  have hl : step (engine (withDot M D)) (.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop
      (parts c scale).stride (parts c scale).body) start.val=some mid := by
    have h0 : (parts c scale).start = .constNat 0 := rfl
    have hn : (parts c scale).stop = .div .nat .nil
      (.sub .nat .nil (.add .nat .nil (.constNat c.K) (.constNat c.BK)) (.constNat 1)) (.constNat c.BK) := rfl
    have h1 : (parts c scale).stride = .constNat 1 := rfl
    simp only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop]
    exact hm
  change run (engine (withDot M D)) (kernel c scale).surfaceBody s=some finish.val
  rw [source_parts,List.append_assoc,run_append,start.property]
  simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]

theorem readback (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) (i : Index c)
    (hi : active c s i)
    (hw : ∀ i j, active c s i → active c s j → outputAddress c s i=outputAddress c s j → i=j) :
    (written M D c scale s).mem "o" (outputAddress c s i)=.mk .fp32 (value M D c scale s i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem frame (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) (r : RegionName) (off : Nat)
    (h : r≠"o" ∨ ∀ i, active c s i → off≠outputAddress c s i) :
    (written M D c scale s).mem r off=s.mem r off := by
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
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput
