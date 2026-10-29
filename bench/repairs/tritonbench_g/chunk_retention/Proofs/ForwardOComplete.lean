import bench.repairs.tritonbench_g.chunk_retention.Proofs.ForwardOValues

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def active (c : Shape) (s : State α) (i : Index c) : Prop :=
  row s i.1.val<c.T ∧ column c s i.2.1.val<c.V
instance (c : Shape) (s : State α) : DecidablePred (active c s) :=
  fun _ => inferInstanceAs (Decidable (_∧_))
def outputAddress (c : Shape) (s : State α) (i : Index c) :=
  s.pids 2*(c.T*c.V)+row s i.1.val*c.V+column c s i.2.1.val
def attention (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (i : ScoreIndex) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul ((accumulated M D c s).2 i) (scoreDecay M c s i))
def valueLoad (M : Algebra α) (c : Shape) (s : State α) (i : Index c) : α :=
  M.fp32Load (if active c s i then (s.mem "v" (outputAddress c s i)).read .real else zero M)
def value (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (i : Index c) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .add ((accumulated M D c s).1 i)
        (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
          (attention (α := α) M D c s) (valueLoad (α := α) M c s) i))
      (M.literal (some .fp32) .real scale))

def finishExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context M c initial current)
    (ha : HasAcc c (accumulated M D c initial).1 (accumulated M D c initial).2 current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).after current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,Kernels.forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,
    Source.prefixBefore,Option.map,Option.getD,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,boundaryZero,Option.bind,engine,withDot,Algebra.withDefaultPrecision,
    resolvePrecision,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,
    natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.di,ctx.ds,ha.1,ha.2,FP.Structural.store]
  rfl

def written (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) : State α :=
  (TileShape.allIndices [64,c.BV]).foldl (fun t i =>
    if active c s i then t.write "o" (outputAddress c s i) (.mk .real (value M D c scale s i)) else t) s

theorem finishMemory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context M c initial current)
    (ha : HasAcc c (accumulated M D c initial).1 (accumulated M D c initial).2 current) :
    (finishExecution M D c scale initial current ctx ha).val.mem=(written M D c scale initial).mem := by
  simp [finishExecution,written,masked_scatter_memory,ctx.memory,active,outputAddress,row,column,
    value,attention,valueLoad,zero,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  rfl

theorem source_parts (c : Shape) (scale : ℝ) : (kernel c scale).surfaceBody = (parts c scale).before ++
    [.forRangeDyn "i_k" (parts c scale).start (parts c scale).stop (parts c scale).stride (parts c scale).body] ++
    (parts c scale).after := rfl

theorem source_memory (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
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

theorem readback (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (i : Index c)
    (hi : active c s i)
    (hw : ∀ i j, active c s i → active c s j → outputAddress c s i=outputAddress c s j → i=j) :
    (written M D c scale s).mem "o" (outputAddress c s i)=.mk .real (value M D c scale s i) :=
  masked_readback_active _ _ _ _ _ i hi hw

theorem frame (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (r : RegionName) (off : Nat)
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
#axiomsClean readback
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
