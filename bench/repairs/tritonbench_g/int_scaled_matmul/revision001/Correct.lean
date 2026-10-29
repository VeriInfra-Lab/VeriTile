import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Whole
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.HostBindings
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.ByteHost

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Correct
open VeriTile.Triton FP.Structural VeriTile.Torch
open Contract TensorBinding InputAdapter FiniteExecution
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Exact admission, including the source-specific finite-index and byte-window
guards. Non-Tensor and malformed Config arguments are rejected. -/
specification python_checks_iff_preconditions (scaled : Bool) (a b c scale : ArgumentMetadata)
    (p : ConfigArgument) :
    checkInputs scaled a b c scale p=.ok () ↔ AdmittedArguments scaled a b c scale p :=
  checkInputs_iff scaled a b c scale p

/-- Independent mathematical integer result of the complete guarded source.
The full signed8 product sum is reduced modulo 2^32 and read as signed32,
exactly as the source's INT32 dot accumulator requires. Empty outputs return
the same supplied C without launching; every exterior logical cell is framed. -/
theorem int_matmul_logical (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (read : Read) (a b c : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (p : Contract.Config) (h : InputContract Bool.false a.metadata b.metadata c.metadata a.metadata p)
    (initial : State Word) (bound : Inputs read a b a Bool.false initial) :
    ∃ final,CheckedHost.call Bool.false cfg u a b c a p initial=some (c,final) ∧
      CheckedHost.OutputPost a b c initial final
        (fun i => LogicalInputs.integerExpected read a b la lb i.1.val i.2.val) := by
  obtain ⟨final,he,metadata,values,frame⟩ := CheckedHost.correct rest Bool.false cfg u a b c a p h initial
  refine ⟨final,he,metadata,?_,frame⟩
  intro i
  simpa only [CheckedHost.reference,Bool.false_eq_true,↓reduceIte,
    LogicalInputs.integer_reference read Bool.false a b c a la lb p h initial bound i] using values i

/-- Complete instruction and byte execution returning the same original C
object. The mathematical result is read through C's actual storage offset;
each output byte and the entire exterior byte frame are determined. -/
specification int_matmul_checked (rest : Algebra Word) (cfg : FP.Config) (u : Nat → Word → Int)
    (read : Read) (a b c : TensorBinding.Tensor) (la : Laws .int8 a) (lb : Laws .int8 b)
    (lc : Laws .int32 c) (p : Contract.Config)
    (h : InputContract Bool.false a.metadata b.metadata c.metadata a.metadata p)
    (initial : State Word) (bound : Inputs read a b a Bool.false initial)
    (outputBound : OutputBytes.Bound read c initial) :
    ∃ final bytes,ByteHost.call Bool.false cfg u a b c a p (initial,read)=some (c,(final,bytes)) ∧
      CheckedHost.OutputPost a b c initial final
        (fun i => LogicalInputs.integerExpected read a b la lb i.1.val i.2.val) ∧
      ByteHost.Post read bytes a b c
        (fun i => LogicalInputs.integerExpected read a b la lb i.1.val i.2.val) := by
  obtain ⟨final,he,post⟩ := int_matmul_logical rest cfg u read a b c la lb p h initial bound
  refine ⟨final,OutputBytes.render read c final,?_,post,?_⟩
  · rw [ByteHost.call_render Bool.false cfg u a b c a p h read initial outputBound,he]
    rfl
  · apply ByteHost.post Bool.false a b c a lc p h read initial final _ post
    intro i
    unfold LogicalInputs.integerExpected
    rw [BitVec.ofInt_toInt]

#axiomsClean python_checks_iff_preconditions
#axiomsClean int_matmul_checked
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.Correct
