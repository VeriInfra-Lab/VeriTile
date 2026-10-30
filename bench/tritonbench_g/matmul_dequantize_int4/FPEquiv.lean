import bench.tritonbench_g.matmul_dequantize_int4.Proofs.GroupExtent

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

/-- Original complete native source graph: prelude, every original K block's
FP16 FMA and live-carry MMA phases, then the FP16 output epilogue. -/
def original (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :=
  ProgramIO.Implementation.original (Candidate.program actual c i) (fun _ => True)

/-- Actual group-extent specialization in the full native graph; ordinary
complete original/compiled candidate quotations are in Kernels.lean. -/
def groupExtent (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :=
  ProgramIO.Implementation.ofVariant (Candidate.variant actual c i) (fun _ => True)

/-- Complete FP16 input/output and FP32 accumulation for every original tuning,
all floating payloads, strided outputs, full grid, safety and exterior frame.
The original four local primitive model bindings are retained unchanged. -/
specification group_extent_equiv (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    original actual c i ≡[([] : Spec.Assumptions _)] groupExtent actual c i :=
  ProgramIO.Implementation.of_equivalent _ (GroupExtent.equivalent actual mma ports c valid i)

#axiomsClean group_extent_equiv
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.FPEquiv
