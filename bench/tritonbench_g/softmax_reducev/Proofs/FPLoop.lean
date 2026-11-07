import bench.tritonbench_g.softmax_reducev.Proofs.FPPrefix
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Range (foldRange range_values)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start : Nat) (a : Accumulator α p.dimensions)
    (h : Ready p s t a) (hN : 0 < p.tokens) :
    ∃ final, run M (loopBody p) (t.setReg "start_n" .int [] (fun _ => (start : Int))) = some final ∧
      Ready p s final (update M p s start a) := by
  simp [loopBody,source,typedKernel,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
    h.mem,h.pids,h.undef,h.batch,h.head,h.seqlen,h.start,h.lanes,h.dimensions,h.locbase,h.valueptr,
    h.maximum,h.normalizer,h.output,TileShape.axisDim,hN,sequenceLength]
  constructor <;> try simp (config := {unfoldPartialApp := Bool.true}) [State.setReg,h.mem,h.pids,h.undef,h.batch,h.head,h.seqlen,h.start,h.lanes,
    h.dimensions,h.locbase,h.valueptr,h.maximum,h.normalizer,h.output,
    update,logits,values,logicOffset,valueOffset,location,active,valuePointer,
    bop,ofFloat,toFloat,Region.cast,sequenceLength,Int.natCast_mul,Int.natCast_add]
  all_goals rfl

theorem loop_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (first : Nat) (last : Int) (stride : Nat) (a : Accumulator α p.dimensions)
    (h : Ready p s t a) (hN : 0<p.tokens) :
    ∃ final,FP.Structural.rangeInt M "start_n" first last stride (loopBody p) t=some final ∧
      Ready p s final (foldRange (update M p s) first last.toNat stride a) := by
  apply Range.range_int_values M "start_n" (loopBody p) (update M p s)
    (fun _ a u=>Ready p s u a) (fun a u=>Ready p s u a) first last stride a t
  · intro n a u hu _
    exact body_run M p s u n a hu hN
  · intro _ _ _ hu
    exact hu
  · exact h
#axiomsClean body_run
#axiomsClean loop_run
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
