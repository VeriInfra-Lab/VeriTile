import bench.repairs.tritonbench_g.token_attn_reduceV.Model
namespace VeriTile.Bench.Repairs.TokenAttnReduceV
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Natural metadata remains nonnegative after the source's signed local
arithmetic. Normalize only its pointer conversion, not every integer node. -/
theorem gather_index_toNat (base start stride : Nat) :
    ((base : Int) + (start : Int) * (stride : Int)).toNat = base + start*stride := by
  rw [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]

structure Ready {α : Type} [Inhabited α] (p : Parameters) (s t : State α)
    (acc : TileIndex [p.width] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  batch : t.regs .nat [] "cur_batch" = some (fun _ => s.pids 0)
  head : t.regs .nat [] "cur_head" = some (fun _ => s.pids 1)
  offsN : t.regs .nat [p.block] "offs_n" = some (fun i => i.1.val)
  offsD : t.regs .nat [p.width] "offs_d" = some (fun i => i.1.val)
  length : t.regs .nat [] "cur_batch_seq_len" = some (fun _ => count p s)
  prob : t.regs .nat [p.block] "p_offs" = some (fun i => probOffset p s i.1.val)
  gather : t.regs .int [p.block] "v_loc_off" = some (fun i => (gatherOffset p s i.1.val : Int))
  value : t.regs .nat [1,p.width] "v_offs" = some (fun i => valueOffset p s i.2.1.val)
  accumulator : t.regs .real [p.width] "acc" = some acc

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    ∃ t, run (withSum M R) (prelude p) s = some t ∧ Ready p s t (initial M p) := by
  simp [prelude, kernel, source, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, integralFloorDiv,
    withSum, Region.cast]
  constructor <;> simp [State.setReg, count, startIndex, requestIndex, probOffset,
    gatherOffset, valueOffset, initial, withSum, bop, numeric, integralFloorDiv]
  all_goals try rfl
  funext i
  simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]



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
    h.length, h.prob, h.gather, h.value, h.accumulator]
  constructor <;> simp [State.setReg, h.mem, h.pids, h.undef, h.batch, h.head,
    h.offsN, h.offsD, h.length, h.prob, h.gather, h.value, update, products,
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
    simp [products, probability, value, gather, padding, Region.cast,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, live, gather_index_toNat]

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
end VeriTile.Bench.Repairs.TokenAttnReduceV
