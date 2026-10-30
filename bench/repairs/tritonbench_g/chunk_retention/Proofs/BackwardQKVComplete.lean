import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardQKVValues

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

abbrev LoopAccum (c : Shape) (α : Type) := Accum c α × State α
def advance (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (f : Nat) (a : LoopAccum c α) : LoopAccum c α :=
  ⟨update M D c s f a.1,storedPartial M D c s a.2 scale f⟩
def trajectory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (n : Nat) : LoopAccum c α :=
  KernelFold.sequenceValue (advance M D c s scale) ⟨⟨fun _=>zero M,fun _=>zero M,fun _=>zero M⟩,s⟩ n
def scoreGradient (M : Algebra α) (c : Shape) (s : State α) (scale : ℝ)
    (a : Accum c α) (i : ScoreIndex) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul (a.2.2 i) (scoreDecay M c s scale i))
def queryValue (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (a : Accum c α) (i : KeyIndex c) : α :=
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (a.1 i) (queryDecay M c s scale ⟨i.1,()⟩))
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (scoreGradient M c s scale a) (keyLoad M c s) i))
def keyValue (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (scale : ℝ) (a : Accum c α) (i : KeyIndex c) : α :=
  M.cast (some .fp32) .real .real (M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (a.2.1 i) (keyDecay M c s ⟨i.1,()⟩))
    (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
      (queryLoad M c s) (scoreGradient M c s scale a) ⟨i.2.1,i.1,()⟩))
def keyActive (c : Shape) (s : State α) (i : KeyIndex c) : Prop :=
  row s i.1.val<c.T ∧ feature c s i.2.1.val<c.K
instance (c : Shape) (s : State α) : DecidablePred (keyActive c s) :=
  fun _ => inferInstanceAs (Decidable (_∧_))
def keyAddress (c : Shape) (s : State α) (i : KeyIndex c) :=
  s.pids 2*(c.T*c.K)+row s i.1.val*c.K+feature c s i.2.1.val
def storedKey (c : Shape) (initial current : State α) (r : RegionName) (value : KeyIndex c → α) : State α :=
  (TileShape.allIndices [64,c.BK]).foldl (fun t i =>
    if keyActive c initial i then t.write r (keyAddress c initial i) (.mk .real (value i)) else t) current
def finishStored (M : Algebra α) (D : DotPrimitive α) (c : Shape) (initial current : State α)
    (scale : ℝ) (a : Accum c α) : State α :=
  storedKey c initial (storedKey c initial current "dq" (queryValue M D c initial scale a))
    "dk" (keyValue M D c initial scale a)
def written (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α) (scale : ℝ) : State α :=
  let a := trajectory M D c s scale c.valueBlocks
  finishStored M D c s a.2 scale a.1

theorem loopExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context M D c scale initial current)
    (ha : HasAcc c ⟨fun _=>zero M,fun _=>zero M,fun _=>zero M⟩ current)
    (hm : current.mem=initial.mem) :
    ∃ t, range (engine (withDot M D)) "i_v" 0 c.valueBlocks 1 (parts c scale).body current=some t ∧
      Context M D c scale initial t ∧ HasAcc c (trajectory M D c initial scale c.valueBlocks).1 t ∧
      t.mem=(trajectory M D c initial scale c.valueBlocks).2.mem := by
  let I := fun (_ : Nat) (a : LoopAccum c α) (s : State α) =>
    Context M D c scale initial s ∧ HasAcc c a.1 s ∧ s.mem=a.2.mem
  have hloop := range_values (engine (withDot M D)) "i_v" (parts c scale).body
    (advance M D c initial scale) I (fun a s => Context M D c scale initial s ∧ HasAcc c a.1 s ∧ s.mem=a.2.mem)
    0 c.valueBlocks 1 ⟨⟨fun _=>zero M,fun _=>zero M,fun _=>zero M⟩,initial⟩ current
    (by
      intro f a s h hf
      let next := bodyExecution M D c scale initial s f a.1 h.1 h.2.1
      refine ⟨next.val,next.property,bodyContext M D c scale initial s f a.1 h.1 h.2.1,
        bodyAcc M D c scale initial s f a.1 h.1 h.2.1,?_⟩
      rw [bodyMemory]
      simp only [advance,storedPartial,masked_scatter_memory,h.2.2])
    (by intro n a s h; exact h) ⟨ctx,ha,hm⟩
  simpa only [KernelFold.from_zero,trajectory] using hloop

def finishExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).after current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,Kernels.backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,numericEq,
    natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.ik,ctx.it,ctx.bh,ctx.dq,ctx.dk,ctx.ds,ctx.query,ctx.key,ha.1,ha.2.1,ha.2.2,FP.Structural.store]
  rfl

theorem finishMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (a : Accum c α)
    (ctx : Context M D c scale initial current) (ha : HasAcc c a current) :
    (finishExecution M D c scale initial current a ctx ha).val.mem=
      (finishStored M D c initial current scale a).mem := by
  simp [finishExecution,finishStored,storedKey,masked_scatter_memory,keyActive,keyAddress,
    queryValue,keyValue,scoreGradient,row,feature,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  rfl

theorem source_parts (c : Shape) (scale : ℝ) : (kernel c scale).surfaceBody=(parts c scale).before ++
    [.forRangeDyn "i_v" (parts c scale).start (parts c scale).stop (parts c scale).stride (parts c scale).body] ++
    (parts c scale).after := rfl

theorem source_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine (withDot M D)) (kernel c scale) s=some t ∧
      t.mem=(written M D c s scale).mem := by
  let start := entryExecution M D c scale s
  have hc := entryContext M D c scale s
  obtain ⟨mid,hm,ctx,ha,hmem⟩ := loopExecution M D c scale s start.val hc.1 hc.2 (entryMemory M D c scale s)
  let finish := finishExecution M D c scale s mid (trajectory M D c s scale c.valueBlocks).1 ctx ha
  refine ⟨finish.val,?_,?_⟩
  · have hl : step (engine (withDot M D)) (.forRangeDyn "i_v" (parts c scale).start (parts c scale).stop
        (parts c scale).stride (parts c scale).body) start.val=some mid := by
      have h0 : (parts c scale).start = .constNat 0 := rfl
      have hn : (parts c scale).stop = .div .nat .nil
        (.sub .nat .nil (.add .nat .nil (.constNat c.V) (.constNat c.BV)) (.constNat 1)) (.constNat c.BV) := rfl
      have h1 : (parts c scale).stride = .constNat 1 := rfl
      simp only [step,h0,hn,h1,evalOp_unfold,Option.bind,Option.bind_some,numeric,bop]
      exact hm
    change run (engine (withDot M D)) (kernel c scale).surfaceBody s=some finish.val
    rw [source_parts,List.append_assoc,run_append,start.property]
    simp only [Option.bind_some,List.singleton_append,run,hl,bind,finish.property]
  · rw [finishMemory]
    simp only [written,finishStored,storedKey,masked_scatter_memory,hmem]

#specNonCircular written avoiding [run,FP.Structural.exec]
#axiomsClean source_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.BackwardQKVSource
