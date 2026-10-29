import bench.tritonbench_g.token_attn_mistral.Proofs.FPModel
namespace VeriTile.Bench.TritonBenchG.TokenAttnMistral.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages (SumPrimitive withSum)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

private theorem natural_window (n : Nat) : (if 0 < n then n else 0) = n := by
  split <;> omega

structure Ready {α : Type} [Inhabited α] (p : Parameters) (s t : State α)
    (acc : TileIndex [p.width] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  batch : t.regs .nat [] "cur_batch" = some (fun _ => s.pids 0)
  head : t.regs .nat [] "cur_head" = some (fun _ => s.pids 1)
  offsN : t.regs .nat [p.block] "offs_n" = some (fun i => i.1.val)
  offsD : t.regs .nat [p.width] "offs_d" = some (fun i => i.1.val)
  length : t.regs .nat [] "cur_att_seq_len" = some (fun _ => count p s)
  sequence : t.regs .nat [] "cur_batch_seq_len" = some (fun _ => sequenceLength p s)
  window : t.regs .nat [] "cur_batch_start_index" = some (fun _ => windowStart p s)
  prob : t.regs .nat [p.block] "p_offs" = some (fun i => probOffset p s i.1.val)
  gather : t.regs .nat [p.block] "v_loc_off" = some (fun i => gatherOffset p s i.1.val)
  value : t.regs .nat [1,p.width] "v_offs" = some (fun i => valueOffset p s i.2.1.val)
  accumulator : t.regs .real [p.width] "acc" = some acc

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, run (withSum M R) (prelude p) s = some t ∧ Ready p s t (initial M p) := by
  simp [prelude, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, integralFloorDiv,
    withSum, Region.cast, natLt, numericLt, natural_window]
  constructor <;> try simp [State.setReg, count, startIndex, requestIndex, probOffset,
    gatherOffset, valueOffset, initial, withSum, bop, numeric, integralFloorDiv, sequenceLength, windowStart]
  all_goals try rfl
  all_goals try (funext i; simp [windowStart, sequenceLength, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Region.cast])
  all_goals first | (intro h; exact (Nat.sub_eq_zero_of_le h).symm) |
    (left; intro h; exact (Nat.sub_eq_zero_of_le h).symm)

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (n : Nat) (a : TileIndex [p.width] → α)
    (h : Ready p s t a) :
    ∃ u, run (withSum M R) (body p) (t.setReg "start_n" .nat [] (fun _ => n)) = some u ∧
      Ready p s u (update M R p s n a) := by
  simp [body, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, withSum,
    h.mem, h.pids, h.undef, h.batch, h.head, h.offsN, h.offsD,
    h.length, h.sequence, h.window, h.prob, h.gather, h.value, h.accumulator]
  constructor <;> try simp [State.setReg, h.mem, h.pids, h.undef, h.batch, h.head,
    h.offsN, h.offsD, h.length, h.sequence, h.window, h.prob, h.gather, h.value, update, products,
    probability, value, gather, padding, bop, numeric, withSum, ofFloat, toFloat]
  all_goals try rfl
  funext i
  rcases i with ⟨i,⟨⟩⟩
  apply congrArg (fun xs : TileIndex [p.block,p.width] → α =>
    M.binary (some .fp32) .real .add (a (i,PUnit.unit))
      (M.cast (some .fp32) .fp32 .real
        (R.apply (some .fp32) ⟨0, by simp⟩ Bool.false xs (i,PUnit.unit))))
  funext j
  rcases j with ⟨j,k,⟨⟩⟩
  change M.binary (some .fp32) .fp32 .mul _ _ = _
  by_cases live : n+j.val < count p s <;>
    by_cases inWindow : n+j.val+windowStart p s < sequenceLength p s <;>
    simp [products, probability, value, gather, padding, Region.cast,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, live, inWindow]

  all_goals norm_num

theorem range_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (start stop : Nat) (a : TileIndex [p.width] → α)
    (h : Ready p s t a) :
    ∃ u, FP.Structural.range (withSum M R) "start_n" start stop p.block (body p) t = some u ∧
      Ready p s u (foldRange (update M R p s) start stop p.block a) := by
  apply range_values (withSum M R) "start_n" (body p) (update M R p s)
    (fun _ a t => Ready p s t a) (fun a t => Ready p s t a) start stop p.block a t
  · intro n a u hu _
    exact body_run M R p s u n a hu
  · intro _ _ _ hu
    exact hu
  · exact h

#axiomsClean prelude_run
#axiomsClean body_run
#axiomsClean range_run
end
end VeriTile.Bench.TritonBenchG.TokenAttnMistral.FPStages
