import bench.repairs.tritonbench_g.int_scaled_matmul.Correct
import bench.repairs.tritonbench_g.int_scaled_matmul.Misc.NativeCorrect

/-! Optional Python admission and actual Tensor pointer/storage helpers. -/
namespace VeriTile.Bench.Repairs.IntScaledMatmul.TensorBridge
open VeriTile.Triton FP.Structural VeriTile.Torch
open Revision001 Contract TensorBinding InputAdapter FiniteExecution
noncomputable section

/-- Complete shared kernel source and independent whole-K signed32 result.
The scaled API retains its actual FP32 promotion, multiplication and output
conversion. All admitted input words, overflow, tails and empty grids remain
included; the result contains no wrapper execution or returned-object claim. -/
theorem int_scaled_matmul_correct (rest : Algebra Word) (scaled : Bool)
    (undefined : Word → BitVec 32) (read : Read) (a b c scale : TensorBinding.Tensor)
    (la : Laws .int8 a) (lb : Laws .int8 b) (ls : scaled=Bool.true → Laws .fp32 scale)
    (p : Contract.Config) (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (inputs : Inputs read a b scale scaled initial) :
    ∃ final,KernelExecution.source rest scaled undefined a b p initial=some final ∧
      KernelExecution.Post a b initial final (KernelExecution.expected scaled undefined read a b scale la lb) :=
  KernelExecution.source_correct rest scaled undefined read a b c scale la lb ls p h initial inputs

/-- Full source/native-instruction equivalence for both current kernels,
including successful source execution, independent output values, actual
Tensor word/byte observations and the exterior memory frame. The concrete
primitive profile has no numerical-transformation assumption. -/
theorem int_scaled_matmul_equiv (rest : Algebra Word) (scaled : Bool)
    (undefined : Word → BitVec 32) (read : Read) (a b c scale : TensorBinding.Tensor)
    (la : Laws .int8 a) (lb : Laws .int8 b) (lc : Laws .int32 c)
    (ls : scaled=Bool.true → Laws .fp32 scale) (p : Contract.Config)
    (h : InputContract scaled a.metadata b.metadata c.metadata scale.metadata p)
    (initial : State Word) (inputs : Inputs read a b scale scaled initial)
    (output : OutputBytes.Bound read c initial) :
    ∃ original lowered bytes,
      KernelExecution.source rest scaled undefined a b p initial=some original ∧
      KernelNative.run scaled undefined a b p initial=some lowered ∧
      KernelNative.bytesRun scaled undefined a b c p (initial,read)=some (lowered,bytes) ∧
      original.mem=lowered.mem ∧
      KernelExecution.Post a b initial original (KernelExecution.expected scaled undefined read a b scale la lb) ∧
      ByteHost.Post read bytes a b c (KernelExecution.expected scaled undefined read a b scale la lb) :=
    by
  simpa only [KernelNative.run_eq,KernelNative.bytesRun_eq] using
    KernelBytes.complete rest scaled undefined read a b c scale la lb lc ls p h initial inputs output


/-- Previous raw interfaces retain their complete original scope. -/
alias raw_int_correct := RawCorrect.int_matmul_correct
alias raw_scaled_correct := RawCorrect.int_scaled_matmul_correct
alias raw_int_equiv := RawFP.int_matmul_equiv
alias raw_scaled_equiv := RawFP.int_scaled_matmul_equiv

/-- Exact Python assertions; this theorem does not execute a host wrapper. -/
theorem python_checks_iff_preconditions (scaled : Bool) (a b c scale : ArgumentMetadata)
    (p : ConfigArgument) :
    checkInputs scaled a b c scale p=.ok () ↔ AdmittedArguments scaled a b c scale p :=
  checkInputs_iff scaled a b c scale p

/- Actual Tensor pointer/storage-offset binding, aligned UINT64 output
addresses, and physical input preservation from the checked disjointness. -/
export KernelBytes (output_addresses input_frame pointer_index direct_int8 direct_fp32)
#axiomsClean int_scaled_matmul_correct
#axiomsClean int_scaled_matmul_equiv
#axiomsClean raw_int_correct
#axiomsClean raw_scaled_correct
#axiomsClean raw_int_equiv
#axiomsClean raw_scaled_equiv
#axiomsClean python_checks_iff_preconditions
#axiomsClean output_addresses
#axiomsClean input_frame
#axiomsClean pointer_index
#axiomsClean direct_int8
#axiomsClean direct_fp32
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IntScaledMatmul.TensorBridge
