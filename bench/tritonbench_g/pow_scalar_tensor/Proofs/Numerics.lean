import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeScalarABI

/-! Optional model-conformance interfaces retained from the original profile.
Ordinary scalar ABI conversion and the unchanged libdevice primitive are model
boundaries, not task-specific numerical experiments. The index-commutation
equivalence requires neither interface and has no numerical assumptions. -/
namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole.Numerics
open VeriTile.Triton FP.Structural
noncomputable section

def ExactExtraction (abi : ScalarABI) (native_python_float_extractFP32 : Float64Bits → Float32Bits) : Prop :=
  ∀ hostWord : Float64Bits,abi.extractFP32 hostWord=native_python_float_extractFP32 hostWord

def ExactPower (M : Algebra Float32Bits) (native_libdevice_powf : Float32Bits → Float32Bits → Float32Bits) : Prop :=
  ∀ baseWord exponentWord : Float32Bits,
    M.binary (some .fp32) .real .pow baseWord exponentWord=native_libdevice_powf baseWord exponentWord

/-- Optional refinement of the extracted scalar and unchanged power primitive.
This supplies no source, control, address or memory fact and is not a pending
numerical obligation for integer-index commutation. -/
theorem local_result (M : Algebra Float32Bits) (abi : ScalarABI)
    (extract : Float64Bits → Float32Bits) (powf : Float32Bits → Float32Bits → Float32Bits)
    (extraction : ExactExtraction abi extract) (power : ExactPower M powf)
    (host : PythonFloat) (exponent : Float32Bits) :
    M.binary (some .fp32) .real .pow (abi.extractFP32 host.payload) exponent=
      powf (extract host.payload) exponent := by
  rw [power,extraction]

#print ExactExtraction
#print ExactPower
#specNonCircular ExactExtraction avoiding [runPythonHost,WholeSource.executeArguments]
#specNonCircular ExactPower avoiding [runPythonHost,WholeSource.executeArguments]
#axiomsClean local_result
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.PowScalarTensor.Whole.Numerics
