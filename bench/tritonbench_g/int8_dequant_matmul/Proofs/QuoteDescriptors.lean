import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteExecution

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem epilogue_descriptors (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (base current : State (Sum F H))
    (divisor : F) (hasDivisor : base.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor))
    (ctx : Context s c base (steps s c) current)
    (acc : HasAcc c (accumulation q s c base) current) :
    let final:=(epilogueExecution (floating q) q s c bias base current (.inl divisor) hasDivisor
      (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) rfl ctx acc).val
    (register final .ptr [c.BM,c.BN] "C"=
      fun i : TileIndex [c.BM,c.BN]=>(("C":RegionName),offset s c base i)) ∧
    (register final .bool [c.BM,c.BN] "mask"=fun i=>decide (active s c base i)) := by
  dsimp only
  by_cases split : c.S=1
  all_goals
    cases bias
    all_goals
      by_cases first : register base .int [] "pid_z" PUnit.unit=0
      all_goals
        simp only [register] at first
        simp [epilogueExecution,split,first,register,masked_write_regs,
          floating,offset,active,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    constructor
    all_goals funext i
    all_goals simp [bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

structure Result (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (base : State (Sum F H)) (divisor : F)
    (final : State (Sum F H)) : Prop where
  out : register final .fp16 [c.BM,c.BN] "out"=fun i=>.inr (value q s c bias base divisor i)
  address : register final .ptr [c.BM,c.BN] "C"=
    fun i : TileIndex [c.BM,c.BN]=>(("C":RegionName),offset s c base i)
  mask : register final .bool [c.BM,c.BN] "mask"=fun i=>decide (active s c base i)
  memory : final.mem=stores q s c bias base divisor

theorem fullResult (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (initial : State (Sum F H))
    (divisor : F) (hasDivisor : initial.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor)) :
    ∃ final, execute (floating q) q s.M s.N (source s c bias) initial=some final ∧
      Result q s c bias (preludeExecution (floating q) q s c bias initial).val divisor final := by
  let base:=(preludeExecution (floating q) q s c bias initial).val
  have start:=preludeContext (floating q) q s c bias initial
  obtain ⟨current,hloop,ctx,acc⟩:=rangeExecution (floating q) q s c bias base base 0 (steps s c)
    (Nat.zero_le _) (fun _=>0) start.1 start.2
  have hd : base.regs .fp32 [] "divfactor"=some (fun _=>.inl divisor) := by
    simpa [base,preludeExecution] using hasDivisor
  let final:=epilogueExecution (floating q) q s c bias base current (.inl divisor) hd
    (fun x y=>.inr (q.ops.hadd (asH q.ops x) (asH q.ops y))) rfl ctx acc
  have descriptors:=epilogue_descriptors q s c bias base current divisor hd ctx acc
  refine ⟨final.val,?_,⟨epilogue_value q s c bias base current divisor hd ctx acc,
    descriptors.1,descriptors.2,epilogue_memory q s c bias base current divisor hd ctx acc⟩⟩
  unfold execute
  rw [source_decomposition,run_append,run_append,(preludeExecution (floating q) q s c bias initial).property]
  change ((run (floating q) q s.M s.N [.forRange "k" 0 (steps s c) 1 (body s c bias)] base).bind
      (run (floating q) q s.M s.N (epilogue s c bias)))=some final.val
  simp only [run,step,hloop,Option.bind_some]
  exact final.property

#axiomsClean epilogue_descriptors
#axiomsClean fullResult

/-- Replay only the final source store statement from its actual registers.
This is the per-program semantics that the grid scheduler interleaves. -/
def replay (q : IntegerModel I F H) [Inhabited (Sum F H)] (c : Config)
    (final : State (Sum F H)) (mem : RegionName→Nat→Cell (Sum F H)) :=
  (TileShape.allIndices [c.BM,c.BN]).foldl (fun current i=>
    let pointer:=register final .ptr [c.BM,c.BN] "C" i
    let value:=register final .fp16 [c.BM,c.BN] "out" i
    if register final .bool [c.BM,c.BN] "mask" i then
      writeCell current pointer.1 pointer.2 (.mk .fp16
        (if c.S=1 then value else .inr (q.ops.hadd
          (asH q.ops ((current pointer.1 pointer.2).read .fp16)) (asH q.ops value))))
    else current) mem

theorem Result.replay_eq (q : IntegerModel I F H) [Inhabited (Sum F H)]
    (s : Shape) (c : Config) (bias : Bool) (base final : State (Sum F H)) (divisor : F)
    (result : Result q s c bias base divisor final) : replay q c final base.mem=final.mem := by
  rw [result.memory]
  unfold replay stores
  simp only [result.address,result.out,result.mask,Prod.fst,Prod.snd,decide_eq_true_eq,asH]
  congr 1
  funext mem i
  by_cases h : active s c base i <;> by_cases hs : c.S=1 <;> simp [h,hs]

#axiomsClean Result.replay_eq
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
