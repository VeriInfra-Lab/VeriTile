import bench.tritonbench_g.multinomial_sampling.Proofs.Core.FPInvariant
namespace VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def bodyResult (P : Primitives α) (d : Storage) (p : Parameters) (initial current : State α)
    (b : Nat) (st : Accumulator α p) : State α :=
  let t0 := current.setReg "b_idx" .nat [] (fun _ => b)
  let t1 := t0.setReg "s_off" .nat [p.blockTokens] (fun i => b+i.1.val)
  let t2 := t1.setReg "s_mask" .bool [p.block,p.blockTokens]
    (fun i => decide (active p initial (i.1,PUnit.unit)) && decide (b+i.2.1.val<p.tokens))
  let t3 := t2.setReg "scores" .real [p.block,p.blockTokens] (scores P d p initial b)
  let t4 := t3.setReg "c_scores" .real [p.block,p.blockTokens] (scanned P d p initial b)
  let t5 := t4.setReg "cum_scores" .real [p.block,p.blockTokens] (cumulative P d p initial b st)
  let t6 := t5.setReg "acc" .real [p.block] (advance P d p initial b st).acc
  let t7 := t6.setReg "pre_cum_scores" .real [p.block,p.blockTokens]
    (fun i => P.sub (cumulative P d p initial b st i) (scores P d p initial b i))
  let t8 := t7.setReg "valid_mask" .bool [p.block,p.blockTokens] (valid P d p initial b st)
  let t9 := t8.setReg "found_mask" .bool [p.block] (found P d p initial b st)
  let t10 := t9.setReg "valid_pos" .nat [p.block] (selected P d p initial b st)
  let t11 := t10.setReg "indices" .int [p.block] (selectedIndex P d p initial b st)
  t11.setReg "output" .int [p.block] (advance P d p initial b st).output

/-- Every original loop statement executes, including all padded lanes and
its full scan/maximum/Bool-sum/argmax operations; no normalized-distribution
premise or first-hit shortcut is used. -/
theorem bodyExecution (rest : Algebra α) (P : Primitives α) (d : Storage) (p : Parameters)
    (initial current : State α) (b : Nat) (st : Accumulator α p)
    (h : Invariant P p initial st current) :
    run (backend rest P) (body d p) (current.setReg "b_idx" .nat [] (fun _ => b)) =
      some (bodyResult P d p initial current b st) := by
  have hn : 0<p.blockTokens := p.tokensDomain.1
  cases d <;>
    simp [body,source,original,original16,originalBfloat16,original64,Source.original,Source.original16,Source.originalBfloat16,Source.original64,ComputeKernel.surfaceBody,Random.i32ArangeSize,
      run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,
      h.memory,h.off,h.n_off,h.mask,h.samp,h.acc,h.output,
      FP.Structural.numeric,FP.Structural.natLt,FP.Structural.numericLt,FP.Structural.numericLe,
      FP.Structural.ofFloat,FP.Structural.toFloat,ComputeDType.eraseDType,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,Option.bind,
      TileShape.axisDim,hn,bodyResult,scores,padding,Storage.dtype,readScore,
      scoreOffset,scanned,cumulative,advance,valid,found,selected,selectedIndex,min_def_lt,ite_mul]
  all_goals delta scores padding Storage.dtype readScore scoreOffset scanned cumulative advance valid found selected selectedIndex FP.Structural.bop
  all_goals simp only [Broadcast.leftIndex,Broadcast.rightIndex,Bool.and_eq_true,decide_eq_true_eq,min_def_lt,ite_mul]
  all_goals try rfl

theorem bodyInvariant (P : Primitives α) (d : Storage) (p : Parameters)
    (initial current : State α) (b : Nat) (st : Accumulator α p)
    (h : Invariant P p initial st current) :
    Invariant P p initial (advance P d p initial b st) (bodyResult P d p initial current b st) := by
  constructor <;> simp [bodyResult,h.memory,h.off,h.n_off,h.mask,h.samp]

#axiomsClean bodyExecution
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.MultinomialSampling.SourceValue
