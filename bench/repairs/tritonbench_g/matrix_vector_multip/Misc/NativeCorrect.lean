import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.NativeResult

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section

def precision : FloatDType := .fp32

/-- The selected typed source with its ordered native FMA/add graph.
All input words are retained; the output observer reads the completed word. -/
abbrev matrixVectorIO (native : Scalars) (N M BN BM pid : Nat)
    (g : Graph BN BM) : ProgramIO (BitVec 32) Payload where
  inputs := [⟨.fp32, [N, M], "A"⟩, ⟨.fp32, [M], "B"⟩]
  outputs := [⟨.fp32, [BN], "C"⟩]
  scratch := []
  buffers := ["A", "B", "C"]
  read := Fin.cases (fun _ j => j.1.val * M + j.2.1.val)
    (Fin.cases (fun _ j => j.1.val) (fun i => i.elim0))
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ i => pid * BN + i.1.val) (fun i => i.elim0)
  outputMask := Fin.cases (fun _ i => pid * BN + i.1.val < N) (fun i => i.elim0)
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => [{
    kernel := Kernels.source N M BN BM
    backend := algebra native g
    pids := fun ax => if ax = 0 then pid else 0 }]
  encode := Payload.word
  observe := fun _ _ => Payload.bits

/-- Source-to-native observation with actual access safety, full-loop success
and the complete frame. Primitive instructions and compiler graph selection
remain explicit model parameters; no floating rewrite law is assumed. -/
theorem mv_fp32_kernel_equiv (native : Scalars) (N M BN BM pid : Nat)
    (g : Graph BN BM) (width : 0 < BM) :
    (matrixVectorIO native N M BN BM pid g).Implements (fun values =>
      IO.outputValues (KernelFP.formula native N M BN BM g
        (IO.matrixValues values) (IO.vectorValues values) pid)) :=
  KernelNativeResult.io_correct native N M BN BM pid g width

#stmtSurfaceSubset mv_fp32_kernel_equiv ⊆
  [Scalars, Graph, Payload, instInhabitedPayload,
    matrixVectorIO, ProgramIO.Implements, ProgramIO.Inputs, IO.outputValues,
    KernelFP.formula, IO.matrixValues, IO.vectorValues]
#specNonCircular KernelFP.formula avoiding [Kernels.source, FP.Structural.exec]
#axiomsClean mv_fp32_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv
