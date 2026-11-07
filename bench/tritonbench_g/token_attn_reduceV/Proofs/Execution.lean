import bench.tritonbench_g.token_attn_reduceV.Proofs.SignedRange
namespace VeriTile.Bench.Repairs.TokenAttnReduceV
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Ready {α : Type} [Inhabited α] (p : Parameters) (s t : State α)
    (acc : TileIndex [p.width] → α) : Prop where
  mem : t.mem = s.mem
  pids : t.pids = s.pids
  undef : t.undef = s.undef
  batch : t.regs .nat [] "cur_batch" = some (fun _ => s.pids 0)
  head : t.regs .nat [] "cur_head" = some (fun _ => s.pids 1)
  offsN : t.regs .nat [p.block] "offs_n" = some (fun i => i.1.val)
  offsD : t.regs .nat [p.width] "offs_d" = some (fun i => i.1.val)
  valid : ValidMetadata p s
  length : t.regs .int [] "cur_batch_seq_len" = some (fun _ => (count p s : Int))
  prob : t.regs .int [p.block] "p_offs" = some (fun i => (probOffset p s i.1.val : Int))
  gather : t.regs .int [p.block] "v_loc_off" = some (fun i => (gatherOffset p s i.1.val : Int))
  value : t.regs .nat [1,p.width] "v_offs" = some (fun i => valueOffset p s i.2.1.val)
  accumulator : t.regs .real [p.width] "acc" = some acc

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (valid : ValidMetadata p s) :
    ∃ t, run (withSum M R) (prelude p) s = some t ∧ Ready p s t (initial M p) := by
  have hl : ((s.mem (Region.cast p.length) (s.pids 0*p.lengthStride)).read .int) = (count p s : Int) := by
    exact (Int.toNat_of_nonneg valid.length).symm
  have hs : ((s.mem (Region.cast p.start) (s.pids 0*p.startStride)).read .int) = (startIndex p s : Int) := by
    exact (Int.toNat_of_nonneg valid.start).symm
  have hr : ((s.mem (Region.cast p.request) (s.pids 0*p.requestStride)).read .int) = (requestIndex p s : Int) := by
    exact (Int.toNat_of_nonneg valid.request).symm
  simp only [Region.cast] at hl hs hr
  simp [prelude, kernel, sourceKernel, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, numeric, bop, integralFloorDiv,
    withSum, Region.cast]
  constructor <;> try simp (config := {unfoldPartialApp := Bool.true}) [State.setReg, hl, hs, hr, probOffset,
    gatherOffset, valueOffset, initial, withSum, bop, numeric, integralFloorDiv,
    Int.natCast_add, Int.natCast_mul]
  all_goals first | exact valid | rfl


theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (n : Nat) (a : TileIndex [p.width] → α)
    (h : Ready p s t a) :
    ∃ u, run (withSum M R) (body p) (t.setReg "start_n" .int [] (fun _ => (n : Int))) = some u ∧
      Ready p s u (update M R p s n a) := by
  simp [body, kernel, sourceKernel, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, withSum,
    h.mem, h.pids, h.undef, h.batch, h.head, h.offsN, h.offsD,
    h.length, h.prob, h.gather, h.value, h.accumulator]
  constructor <;> try simp [State.setReg, h.mem, h.pids, h.undef, h.batch, h.head,
    h.offsN, h.offsD, h.length, h.prob, h.gather, h.value, update, products,
    probability, value, gather, padding, bop, numeric, withSum, ofFloat, toFloat]
  all_goals try exact h.valid
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
  dsimp only [Broadcast.leftIndex, Broadcast.rightIndex]
  have mask : ((n : Int) + (j.val : Int) < (count p s : Int)) ↔ n+j.val<count p s := by omega
  simp only [mask]
  by_cases live : n+j.val<count p s
  · have shifted : gatherOffset p s j.val+n*p.requestTokenStride=gatherOffset p s (n+j.val) := by
      unfold gatherOffset; ring
    have nonnegative := h.valid.tokens (n+j.val) live (by have := j.isLt; omega)
    rw [←shifted] at nonnegative
    have word := Int.toNat_of_nonneg nonnegative
    simp only [Region.cast] at word
    simp only [products,probability,value,gather,padding,Region.cast,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,mask,live,if_true,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    rw [←word]
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  · simp [products,probability,value,gather,padding,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,mask,live] <;> norm_num


theorem range_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (start stop : Nat) (a : TileIndex [p.width] → α)
    (h : Ready p s t a) :
    ∃ u, FP.Structural.rangeInt (withSum M R) "start_n" start stop p.block (body p) t = some u ∧
      Ready p s u (foldRange (update M R p s) start stop p.block a) := by
  apply range_int_values (withSum M R) "start_n" (body p) (update M R p s)
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
