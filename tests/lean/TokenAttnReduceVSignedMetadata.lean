import bench.tritonbench_g.token_attn_reduceV.Proofs.Execution

namespace VeriTile.Tests.TokenAttnReduceVSignedMetadata
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000

/-- The actual prelude retains all signed metadata words, including negative
ones. Nonnegativity belongs to the mathematical input domain, not the parser
or memory reader. -/
theorem signed_prelude {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (s : State α) :
    (run (withSum M R) (prelude p) s).map
      (fun t => (t.regs .int [] "cur_batch_seq_len",
        t.regs .int [] "cur_batch_in_all_start_index", t.regs .int [] "cur_batch_req_idx")) =
      some (some (fun _ => (s.mem (Region.cast p.length) (s.pids 0*p.lengthStride)).read .int),
        some (fun _ => (s.mem (Region.cast p.start) (s.pids 0*p.startStride)).read .int),
        some (fun _ => (s.mem (Region.cast p.request) (s.pids 0*p.requestStride)).read .int)) := by
  simp [prelude,kernel,sourceKernel,ComputeKernel.surfaceBody,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,integralFloorDiv,
    withSum,Region.cast,State.setReg]

#axiomsClean signed_prelude
end
end VeriTile.Tests.TokenAttnReduceVSignedMetadata
