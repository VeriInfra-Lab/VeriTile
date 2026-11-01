import bench.tritonbench_g.softmax_reducev.Proofs.FPPrefix

namespace VeriTile.Tests.SoftmaxReduceVSignedLoads
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- Actual source preserves signed sequence metadata and the complete signed
page origin even for lengths greater than maximum; no host domain is assumed. -/
theorem signed_page_origin {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    (run M (prelude p) s).map (fun t=>t.regs .int [] "off_b_loc")=
      some (some (fun _=>(s.pids 0*p.locationBatchStride : Nat)+
        ((p.maxLength : Int)-(s.mem (Region.cast p.lengths) (s.pids 0)).read .int)*p.locationSequenceStride)) := by
  obtain ⟨t,ht,ready⟩ := prefix_run M p s
  rw [ht]
  simpa only [Option.map_some,locationBase,sequenceWord] using congrArg some ready.locbase

/-- Input load rounding is retained even on masked padding, independently of
casts and arithmetic in the opaque backend. -/
theorem padded_loads {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) (start : Nat) (i : Fin p.tokens)
    (d : Fin p.dimensions) (inactive : ¬active p s start i) :
    logits M p s start (i,())=M.fp32Load M.negInf ∧
    values M p s start (i,d,())=M.fp32Load (M.literal (some .fp32) .real 0.0) := by
  simp [logits,values,inactive]

#axiomsClean signed_page_origin
#axiomsClean padded_loads
end
end VeriTile.Tests.SoftmaxReduceVSignedLoads
