import bench.tritonbench_g.token_attn_mistral.Proofs.Model
namespace VeriTile.Bench.Repairs.TokenAttnMistral
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnMistral.Range (foldRange range_values)
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
  valid : ValidMetadata p s
  length : t.regs .int [] "cur_att_seq_len" = some (fun _ => (count p s : Int))
  sequence : t.regs .int [] "cur_batch_seq_len" = some (fun _ => (sequenceLength p s : Int))
  window : t.regs .int [] "cur_batch_start_index" = some (fun _ => (windowStart p s : Int))
  prob : t.regs .int [p.block] "p_offs" = some (fun i => (probOffset p s i.1.val : Int))
  gather : t.regs .int [p.block] "v_loc_off" = some (fun i => (gatherOffset p s i.1.val : Int))
  value : t.regs .nat [1,p.width] "v_offs" = some (fun i => valueOffset p s i.2.1.val)
  accumulator : t.regs .real [p.width] "acc" = some acc

theorem prelude_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) (valid : ValidMetadata p s) :
    ∃ t, run (withSum M R) (prelude p) s = some t ∧ Ready p s t (initial M p) := by
  have hs : (s.mem (Region.cast p.length) (s.pids 0*p.sequenceStride)).read .int=(sequenceLength p s : Int) :=
    (Int.toNat_of_nonneg valid.sequence).symm
  have ha : (s.mem (Region.cast p.attentionLength) (s.pids 0*p.attentionLengthStride)).read .int=(count p s : Int) :=
    (Int.toNat_of_nonneg valid.attention).symm
  have ht : (s.mem (Region.cast p.start) (s.pids 0*p.attentionStartStride)).read .int=(startIndex p s : Int) :=
    (Int.toNat_of_nonneg valid.start).symm
  have hr : (s.mem (Region.cast p.request) (s.pids 0*p.requestStride)).read .int=(requestIndex p s : Int) :=
    (Int.toNat_of_nonneg valid.request).symm
  have hw : (if p.window<sequenceLength p s then
      (sequenceLength p s : Int)-p.window else 0)=(windowStart p s : Int) := by
    unfold windowStart
    split <;> omega
  simp only [Region.cast] at hs ha ht hr
  simp [prelude,kernel,sourceKernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,integralFloorDiv,
    withSum,Region.cast,natLt,numericLt,hs,ha,ht,hr,hw]
  constructor <;> try simp (config := {unfoldPartialApp := Bool.true}) [State.setReg,
    probOffset,gatherOffset,valueOffset,initial,withSum,bop,numeric,
    integralFloorDiv,Int.natCast_add,Int.natCast_mul,hw]
  all_goals first | exact valid | rfl


theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (n : Nat) (a : TileIndex [p.width] → α)
    (h : Ready p s t a) (hn : n<count p s) (aligned : n%p.block=0) :
    ∃ u, run (withSum M R) (body p) (t.setReg "start_n" .int [] (fun _ => (n : Int))) = some u ∧
      Ready p s u (update M R p s n a) := by
  simp [body, kernel, sourceKernel, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
    FloatDType.toTileDType, FloatDType.numericDType, ofFloat, toFloat,
    numeric, bop, numericLt, natLt, Region.cast, withSum,
    h.mem, h.pids, h.undef, h.batch, h.head, h.offsN, h.offsD,
    h.length, h.sequence, h.window, h.prob, h.gather, h.value, h.accumulator]
  constructor <;> try simp [State.setReg, h.mem, h.pids, h.undef, h.batch, h.head,
    h.offsN, h.offsD, h.length, h.sequence, h.window, h.prob, h.gather, h.value, update, products,
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
  dsimp only [Broadcast.leftIndex,Broadcast.rightIndex]
  have mask : ((n : Int)+(j.val : Int)<(count p s : Int)) ↔ n+j.val<count p s := by omega
  have vm : ((n : Int)+(j.val : Int)+(windowStart p s : Int)<(sequenceLength p s : Int)) ↔
      n+j.val+windowStart p s<sequenceLength p s := by omega
  by_cases inWindow : n+j.val+windowStart p s<sequenceLength p s
  · have shifted : gatherOffset p s j.val+n*p.requestTokenStride=gatherOffset p s (n+j.val) := by
      unfold gatherOffset; ring
    have nonnegative := h.valid.tokens n hn aligned j inWindow
    rw [←shifted] at nonnegative
    simp only [Region.cast] at nonnegative
    have word := Int.toNat_of_nonneg nonnegative
    simp only [Region.cast] at word
    by_cases live : n+j.val<count p s
    all_goals simp only [products,probability,value,gather,padding,Region.cast,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,mask,vm,live,inWindow,if_true,if_false,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    all_goals simp [mask,vm,live,inWindow,max_eq_left nonnegative]
    all_goals rw [←word]
    all_goals simp only [←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    all_goals norm_num only
  · by_cases live : n+j.val<count p s
    all_goals simp only [products,probability,value,gather,padding,Region.cast,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,mask,vm,live,inWindow,
      ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    all_goals simp [mask,vm,live,inWindow] <;> norm_num


theorem range_run {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s t : State α) (start stop : Nat) (a : TileIndex [p.width] → α)
    (h : Ready p s t a) (bound : stop≤count p s) (aligned : start%p.block=0) :
    ∃ u, FP.Structural.rangeInt (withSum M R) "start_n" start stop p.block (body p) t = some u ∧
      Ready p s u (foldRange (update M R p s) start stop p.block a) := by
  apply Range.range_int_values (withSum M R) "start_n" (body p) (update M R p s)
    (fun n a t => Ready p s t a ∧ n%p.block=0) (fun a t => Ready p s t a) start stop p.block a t
  · intro n a u hu live
    obtain ⟨next,ran,ready⟩ := body_run M R p s u n a hu.1 (lt_of_lt_of_le live bound) hu.2
    exact ⟨next,ran,ready,by simp [Nat.add_mod,hu.2]⟩
  · intro _ _ _ hu
    exact hu.1
  · exact ⟨h,aligned⟩

#axiomsClean prelude_run
#axiomsClean body_run
#axiomsClean range_run
end
end VeriTile.Bench.Repairs.TokenAttnMistral
