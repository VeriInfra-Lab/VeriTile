import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Kernels
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Contract
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.NativeFull
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.Direct
import bench.repairs.tritonbench_g.dequantize_rowwise.Proofs.FP32Prior.Proofs.FP
namespace VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior
open VeriTile.Triton FP.Structural Bytes Direct
noncomputable section

/-- Unconditional full returned FP16 bytes, caller-memory frame and metadata for
the actual two-multiply/cast source graph, with arbitrary scalar operations. -/
theorem dequantize_rowwise_source_fp32 (c : Request) (out : Allocation c)
    (ops : Memory.Numerics) (m : Bytes.Memory) :
    ∃ ret,run c out ops m=.ok ret ∧ Result c out ops m ret := run_correct c out ops m

/-- Local scalar hardware conformance remains explicit. All source control,
indices, byte addresses, masks, grid coverage, allocations and frames are proved. -/
theorem dequantize_rowwise_fp32_if_local (c : Request) (out : Allocation c)
    (ops : Memory.Numerics) (m : Bytes.Memory)
    (ports : ∀ q : Coordinates c,Floating.LocalNumerics ops
      (logical m c.scale q.1.val) (logical m c.x (linear q))) :
    ∃ ret,run c out ops m=.ok ret ∧ Result c out Floating.software m ret := by
  obtain ⟨ret,hr,ho,hm,shape,dtype,device,read,frame⟩:=run_correct c out ops m
  refine ⟨ret,hr,ho,hm,shape,dtype,device,?_,frame⟩
  intro q
  rw [read q,Floating.compute_reference ops _ _ (ports q)]

#axiomsClean dequantize_rowwise_source_fp32
#axiomsClean dequantize_rowwise_fp32_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeRowwise.FP32Prior
