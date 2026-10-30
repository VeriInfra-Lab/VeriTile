import bench.tritonbench_g.chunk_gated_attention.Sources.NativeLoop

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (scatter_memory masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def finishExecution (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (acc : HIndex c→NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc) :
    {t : State NativeGraph.Expr // run (engine algebra) (parts c base gateK hasSeed storeFinal).after current=some t} := by
  have hacc:=ha
  unfold HasAcc at hacc
  cases storeFinal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (config := {maxSteps:=150000}) [parts,source,NativeSource.hOriginal,NativeSource.hSource,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg,integralFloorDiv,integralMod,ctx.bh,ctx.ik,ctx.iv,hacc,FP.Structural.store,
      algebra,ScalarGraph.algebra,NativeGraph.algebra]
    rfl

def finalWrite (c : Shape) (base : Nat) (initial current : State NativeGraph.Expr)
    (storeFinal : Bool) (acc : HIndex c→NativeGraph.Expr) : State NativeGraph.Expr :=
  if storeFinal then (TileShape.allIndices [c.BK,c.BV]).foldl (fun t i=>t.write "ht"
    (stateOffset c base initial i) (.mk .fp32 (.cast (some .fp32) .real .fp32 (acc i)))) current
  else current

theorem finishMemory (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (acc : HIndex c→NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc) :
    (finishExecution c base gateK hasSeed storeFinal initial current acc ctx ha).val.mem=
      (finalWrite c base initial current storeFinal acc).mem := by
  cases storeFinal <;> simp [finishExecution,finalWrite,stateOffset,scatter_memory,masked_scatter_memory,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]

/-- Full finite history/final reference, computed from the original input
snapshot by the carried-dot recurrence and explicit typed stores. -/
def result (c : Shape) (base : Nat) (s : State NativeGraph.Expr) (gateK hasSeed storeFinal : Bool) : State NativeGraph.Expr :=
  let a:=reference c base s gateK hasSeed
  finalWrite c base s {s with mem:=a.2} storeFinal a.1

theorem finalWrite_memory (c : Shape) (base : Nat) (initial a b : State NativeGraph.Expr)
    (storeFinal : Bool) (acc : HIndex c→NativeGraph.Expr) (same : a.mem=b.mem) :
    (finalWrite c base initial a storeFinal acc).mem=(finalWrite c base initial b storeFinal acc).mem := by
  cases storeFinal <;> simp [finalWrite,scatter_memory,same]

theorem source_parts (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool) :
    (source c base gateK hasSeed storeFinal).surfaceBody=(parts c base gateK hasSeed storeFinal).before ++
      [.forRange "i_t" 0 c.NT 1 (parts c base gateK hasSeed storeFinal).body] ++
      (parts c base gateK hasSeed storeFinal).after := by
  cases gateK <;> cases hasSeed <;> cases storeFinal <;> rfl

/-- Successful full typed source execution is constructed for every input;
its exact memory equals the independent gated FMA recurrence. -/
theorem source_memory (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool) (s : State NativeGraph.Expr) :
    ∃ final,FP.Structural.exec (engine algebra) (source c base gateK hasSeed storeFinal) s=some final ∧
      final.mem=(result c base s gateK hasSeed storeFinal).mem := by
  let start:=entryExecution algebra c base gateK hasSeed storeFinal s
  have startContext:=entryContext algebra c base gateK hasSeed storeFinal s
  obtain ⟨mid,loop,ctx,ha,hm⟩:=loopExecution c base gateK hasSeed storeFinal s start.val
    startContext.1 startContext.2 (entryMemory c base gateK hasSeed storeFinal s)
  let finish:=finishExecution c base gateK hasSeed storeFinal s mid _ ctx ha
  refine ⟨finish.val,?_,?_⟩
  · change run (engine algebra) (source c base gateK hasSeed storeFinal).surfaceBody s=_
    rw [source_parts]
    simp only [List.append_assoc,run_append,start.property,Option.bind_some,List.singleton_append]
    rw [run]
    simp only [step,loop,bind,Option.bind_some]
    exact finish.property
  · exact (finishMemory c base gateK hasSeed storeFinal s mid _ ctx ha).trans
      (finalWrite_memory c base s mid {s with mem:=(reference c base s gateK hasSeed).2}
        storeFinal _ hm)

#specNonCircular result avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.range]
#axiomsClean source_memory
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
