import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.FPPrefix
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Range (foldRange range_values)
noncomputable section
set_option maxHeartbeats 15000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false

theorem body_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (start : Nat) (a : Accumulator α p.dimensions)
    (h : Ready p s t a) (hN : 0 < p.tokens) :
    ∃ final, run M (loopBody p) (t.setReg "start_n" .nat [] (fun _ => start)) = some final ∧
      Ready p s final (update M p s start a) := by
  simp [loopBody,source,typed,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,natLe,numericLt,numericLe,
    integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
    h.mem,h.pids,h.undef,h.batch,h.head,h.seqlen,h.start,h.lanes,h.dimensions,h.locbase,h.valueptr,
    h.maximum,h.normalizer,h.output,TileShape.axisDim,hN]
  constructor <;> simp [State.setReg,h.mem,h.pids,h.undef,h.batch,h.head,h.seqlen,h.start,h.lanes,
    h.dimensions,h.locbase,h.valueptr,h.maximum,h.normalizer,h.output,
    update,logits,values,logicOffset,valueOffset,location,active,valuePointer,
    bop,ofFloat,toFloat,Region.cast]
  all_goals rfl

theorem loop_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s t : State α) (first last stride : Nat) (a : Accumulator α p.dimensions)
    (h : Ready p s t a) (hN : 0 < p.tokens) :
    ∃ final, FP.Structural.range M "start_n" first last stride (loopBody p) t = some final ∧
      Ready p s final (foldRange (update M p s) first last stride a) := by
  apply range_values M "start_n" (loopBody p) (update M p s)
    (fun _ a u => Ready p s u a) (fun a u => Ready p s u a) first last stride a t
  · intro n a u hu _
    exact body_run M p s u n a hu hN
  · intro n a u hu
    exact hu
  · exact h
#axiomsClean body_run
#axiomsClean loop_run
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
