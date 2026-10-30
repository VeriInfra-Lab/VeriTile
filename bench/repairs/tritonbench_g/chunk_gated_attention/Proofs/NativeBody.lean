import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeBodySource

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def historyWrite (c : Shape) (base : Nat) (initial current : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c→NativeGraph.Expr) : State NativeGraph.Expr :=
  (TileShape.allIndices [c.BK,c.BV]).foldl (fun t i=>t.write "h"
    (historyOffset c base initial chunk i) (.mk .real (.cast (some .fp32) .real .real (acc i)))) current

theorem historyWrite_frame (c : Shape) (base : Nat) (initial current : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c→NativeGraph.Expr) (r : RegionName) (off : Nat) (hr : r≠"h") :
    (historyWrite c base initial current chunk acc).mem r off=current.mem r off :=
  State.scatter_frame _ _ _ _ _ _ (Or.inl hr) _

theorem bodyMemory (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (chunk : Nat) (acc : HIndex c→NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc) :
    (bodyExecution c base gateK hasSeed storeFinal initial current chunk acc ctx ha).val.mem=
      (historyWrite c base initial current chunk acc).mem := by
  cases gateK <;>
    simp [bodyExecution,historyWrite,historyOffset,
      VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_memory,
      VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.scatter_memory,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex] <;> rfl

theorem bodyContext (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (chunk : Nat) (acc : HIndex c→NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc) :
    Context c base initial (bodyExecution c base gateK hasSeed storeFinal initial current chunk acc ctx ha).val := by
  constructor
  · intro off
    rw [bodyMemory,historyWrite_frame c base initial current chunk acc "k" off (by decide),ctx.key]
  · intro off
    rw [bodyMemory,historyWrite_frame c base initial current chunk acc "v" off (by decide),ctx.value]
  · intro off
    rw [bodyMemory,historyWrite_frame c base initial current chunk acc "g" off (by decide),ctx.gate]
  · intro off
    rw [bodyMemory,historyWrite_frame c base initial current chunk acc "h0" off (by decide),ctx.seed]
  all_goals cases gateK <;> simp [bodyExecution,State.scatter_regs,
    VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy.masked_scatter_regs,ctx.bh,ctx.ik,ctx.iv]

theorem bodyAcc (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (chunk : Nat) (acc : HIndex c→NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc) :
    HasAcc c (bodyExecution c base gateK hasSeed storeFinal initial current chunk acc ctx ha).val
      (update c base initial gateK chunk acc) := by
  cases gateK <;>
    simp [HasAcc,bodyExecution,update,keyValue,valueValue,gated,load,
      keyOffset,valueOffset,keyEndpoint,valueEndpoint,algebra,ScalarGraph.algebra,NativeGraph.algebra,
      NativeGraph.consume,NativeGraph.options,State.scatter_frame,TileShape.indexToList,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,ctx.key,ctx.value,ctx.gate] <;> rfl

#axiomsClean bodyExecution
#axiomsClean bodyMemory
#axiomsClean bodyContext
#axiomsClean bodyAcc
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
