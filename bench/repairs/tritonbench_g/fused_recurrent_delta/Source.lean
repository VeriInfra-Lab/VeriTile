import bench.repairs.tritonbench_g.fused_recurrent_delta.Kernels
/-! The complete authoritative original and load-scheduled candidate are the
ordinary Runtime.forward, Runtime.backward and Runtime.forwardDirectOutput
quotations in Kernels.lean. All use tl.param scale : tl.float32.
The prior literal source lives only in Proofs/LiteralSource.lean. -/
