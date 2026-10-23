import bench.repairs.tritonbench_g.context_attn_llama.Proofs.NativeHost
import bench.tritonbench_g.context_attn_llama.Proofs.WholeProgram

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnLlama
open FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Exact occurrences in the shared typed FP32 program. These equalities bind
the integer observation lemmas to Program.kernel, including its real loads,
causal predicate and output store; floating statements stay in their source order. -/
def Sites (p : Layout) (r : Regions) (scale : ℝ) : Prop :=
    (FPSuccess.parts p r scale).before[13]? = some (.assign .int [p.BM,p.D] "off_q" (.alg (PreludeProgram.offset p))) ∧
    (FPSuccess.parts p r scale).before[14]? = some (.assign .real [p.BM,p.D] "q"
      (.compute (.load .fp32 (.region r.query (Op.ref .int [p.BM,p.D] "off_q").castIntToNat)
        (.maskOther (PreludeProgram.queryMask p) (.broadcast (.const (0.0 : ℝ)) [p.BM,p.D]))))) ∧
    (FPSuccess.parts p r scale).before[19]? = some (.assign .int [] "block_end_loc" (.alg (PreludeProgram.ending p))) ∧
    (FPSuccess.parts p r scale).body[1]? = some (.assign .int [p.BN] "kv_loc"
      (.alg (.load .int (.region r.pages (LoopProgram.pageIndex p).castIntToNat)
        (.maskOther (LoopProgram.pageMask p) (.broadcast (.constInt 0) [p.BN]))))) ∧
    (FPSuccess.parts p r scale).body[2]? = some (.assign .int [p.D,p.BN] "off_k" (.alg (LoopProgram.keyOffsets p))) ∧
    (FPSuccess.parts p r scale).body[3]? = some (.assign .real [p.D,p.BN] "k"
      (.compute (.load .fp32 (.region r.key (Op.ref .int [p.D,p.BN] "off_k").castIntToNat)
        (.maskOther (LoopProgram.keyMask p) (.broadcast (.const (0.0 : ℝ)) [p.D,p.BN]))))) ∧
    (FPSuccess.parts p r scale).body[5]? = some (.assign .bool [p.BM,p.BN] "mask" (.alg (LoopProgram.causalMask p))) ∧
    (FPSuccess.parts p r scale).body[14]? = some (.assign .int [p.BN,p.D] "off_v" (.alg (LoopProgram.valueOffsets p))) ∧
    (FPSuccess.parts p r scale).body[15]? = some (.assign .real [p.BN,p.D] "v"
      (.compute (.load .fp32 (.region r.value (Op.ref .int [p.BN,p.D] "off_v").castIntToNat)
        (.maskOther (LoopProgram.valueMask p) (.broadcast (.const (0.0 : ℝ)) [p.BN,p.D]))))) ∧
    (FPSuccess.parts p r scale).after[1]? = some (.assign .int [p.BM,p.D] "off_o" (.alg (EpilogueProgram.offsets p)))

theorem source_sites (p : Layout) (r : Regions) (scale : ℝ) : Sites p r scale := by
  exact ⟨rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl,rfl⟩

/-- Every integer register needed by the source observations. Page lanes are
bound immediately after the actual masked metadata load. -/
structure Discrete {α : Type} [Inhabited α] (c : Host.Call) (idx : GridIndex c.grid)
    (step : Nat) (s : State α) : Prop where
  start : s.regs .int [] "cur_batch_in_all_start_index" = some (fun _ => (c.start (batchAt c idx) : Int))
  head : s.regs .nat [] "cur_head" = some (fun _ => Formula.head c.layout (c.launchAt idx))
  kv : s.regs .nat [] "cur_kv_head" = some (fun _ => Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)
  request : s.regs .int [] "cur_batch_req_idx" = some (fun _ => (c.request (batchAt c idx) : Int))
  rows : s.regs .nat [c.layout.BM] "offs_m" = some (fun i => Formula.queryRow c.layout (c.launchAt idx) i.1.val)
  dims : s.regs .nat [c.layout.D] "offs_d" = some (fun i => i.1.val)
  lanes : s.regs .nat [c.layout.BN] "offs_n" = some (fun i => i.1.val)
  length : s.regs .int [] "cur_batch_seq_len" = some (fun _ => (c.length (batchAt c idx) : Int))
  prompt : s.regs .int [] "prompt_cache_len" = some (fun _ => (c.prompt (batchAt c idx) : Int))
  ending : s.regs .int [] "block_end_loc" = some (fun _ => (endAt c idx : Int))
  index : s.regs .int [] "start_n" = some (fun _ => ((step*c.layout.BN : Nat) : Int))
  pages : s.regs .int [c.layout.BN] "kv_loc" = some (fun i => (mathPage c idx step i.1.val : Int))

def Observed {α : Type} [Inhabited α] (c : Host.Call) (idx : GridIndex c.grid)
    (step : Nat) (M : Algebra α) (s : State α) : Prop :=
    evalOp M none (PreludeProgram.offset c.layout) s = some (fun i =>
      (((c.start (batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) i.1.val)*c.layout.qRow+
        Formula.head c.layout (c.launchAt idx)*c.layout.qHead+i.2.1.val*c.layout.qDim : Nat) : Int)) ∧
    evalOp M none (PreludeProgram.queryMask c.layout) s = some (fun i =>
      decide (Formula.queryRow c.layout (c.launchAt idx) i.1.val < c.length (batchAt c idx))) ∧
    evalOp M none (LoopProgram.pageIndex c.layout) s = some (fun i =>
      ((c.layout.reqRow*c.request (batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+i.1.val) : Nat) : Int)) ∧
    evalOp M none (LoopProgram.pageMask c.layout) s = some (fun i => decide (step*c.layout.BN+i.1.val < endAt c idx)) ∧
    evalOp M none (LoopProgram.keyOffsets c.layout) s = some (fun i =>
      ((mathPage c idx step i.2.1.val*c.layout.kRow+
        (Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)*c.layout.kHead+i.1.val*c.layout.kDim : Nat) : Int)) ∧
    evalOp M none (LoopProgram.keyMask c.layout) s = some (fun i => decide (step*c.layout.BN+i.2.1.val < endAt c idx)) ∧
    evalOp M none (LoopProgram.causalMask c.layout) s = some (fun i =>
      decide (step*c.layout.BN+i.2.1.val ≤ Formula.queryRow c.layout (c.launchAt idx) i.1.val+c.prompt (batchAt c idx))) ∧
    evalOp M none (LoopProgram.valueOffsets c.layout) s = some (fun i =>
      ((mathPage c idx step i.1.val*c.layout.vRow+
        (Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup)*c.layout.vHead+i.2.1.val*c.layout.vDim : Nat) : Int)) ∧
    evalOp M none (LoopProgram.valueMask c.layout) s = some (fun i => decide (step*c.layout.BN+i.1.val < endAt c idx)) ∧
    evalOp M none (EpilogueProgram.offsets c.layout) s = some (fun i =>
      ((c.output (batchAt c idx) (Formula.head c.layout (c.launchAt idx))
        (Formula.queryRow c.layout (c.launchAt idx) i.1.val) i.2.1.val : Nat) : Int))

theorem source_observations {α : Type} [Inhabited α] (c : Host.Call) (idx : GridIndex c.grid)
    (step : Nat) (M : Algebra α) (s : State α) (h : Discrete c idx step s) : Observed c idx step M s := by
  unfold Observed
  rcases h with ⟨hs,hh,hkv,hr,hm,hd,hn,hl,hp,he,hi,hpg⟩
  simp [PreludeProgram.offset,PreludeProgram.queryMask,PreludeProgram.rowView,PreludeProgram.dimView,
    PreludeProgram.rows,PreludeProgram.dimensions,PreludeProgram.n,PreludeProgram.z,
    LoopProgram.pageIndex,LoopProgram.pageMask,LoopProgram.keyOffsets,LoopProgram.keyMask,
    LoopProgram.causalMask,LoopProgram.valueOffsets,LoopProgram.valueMask,LoopProgram.pageKeyView,
    LoopProgram.pageValueView,LoopProgram.pageLanes,LoopProgram.keyView,LoopProgram.valueView,
    LoopProgram.keyChannel,LoopProgram.valueChannel,LoopProgram.lanes,EpilogueProgram.offsets,
    evalOp_unfold,hs,hh,hkv,hr,hm,hd,hn,hl,hp,he,hi,hpg,Option.bind,numeric,numericLt,numericLe,
    Broadcast.leftIndex,Broadcast.rightIndex,Host.Call.output,Nat.cast_add,Nat.cast_mul,
    Int.ofNat_lt,Int.ofNat_le,bop]
  repeat' apply And.intro
  all_goals first | rfl | (funext i; simp only [bop,Broadcast.leftIndex,Broadcast.rightIndex,←Nat.cast_add,←Nat.cast_mul,Int.ofNat_lt,Int.ofNat_le])

end
end VeriTile.Bench.Repairs.ContextAttnLlama.Native
