import bench.tritonbench_g.chunk_retention.Sources.ForwardOResult
import bench.tritonbench_g.chunk_retention.Sources.KernelStaging

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel ForwardOSource
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_readback_active)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

def kernel (c : Shape) (scale : ℝ) :=
  Kernels.forwardOScaleFirst "q" "k" "v" "h" "o"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    scale c.H c.T c.K c.V 64 c.BK c.BV

def parts (c : Shape) (scale : ℝ) : Source.DynamicRange :=
  (Source.dynamicRange? (kernel c scale).surfaceBody "i_k").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def value (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (i : Index c) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul (M.literal (some .fp32) .real scale)
      (M.binary (some .fp32) .real .add ((accumulated M D c s).1 i)
        (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
          (attention (α := α) M D c s) (valueLoad (α := α) M c s) i)))

def finishExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context M c initial current)
    (ha : HasAcc c (accumulated M D c initial).1 (accumulated M D c initial).2 current) :
    {t : State α // run (engine (withDot M D)) (parts c scale).after current=some t} := by
  simp only [HasAcc] at ha
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,Kernels.forwardOScaleFirst,ComputeKernel.surfaceBody,Source.dynamicRange?,
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

theorem before_eq (c : Shape) (scale : ℝ) :
    (parts c scale).before=(ForwardOSource.parts c scale).before := rfl

theorem body_eq (c : Shape) (scale : ℝ) :
    (parts c scale).body=(ForwardOSource.parts c scale).body := rfl

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
  rw [source_parts,List.append_assoc,run_append,before_eq,start.property]
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
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ScaleFirstSource
