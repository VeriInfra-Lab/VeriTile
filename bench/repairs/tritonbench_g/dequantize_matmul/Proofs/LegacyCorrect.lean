import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.DirectComplete
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Algorithm

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
/-- Whole mathematical returned result. The plan is selected from metadata
before any matrix values are inspected; completeness covers every K term.
The exact Python output-selection and allocation branches are in prepare. -/
specification matmul_dequantize_int8_correct (allocate : Allocator ℝ) (s : Shape)
    (host : HostInputs) (provided : Option Descriptor)
    (validOut : ∀ out,provided=some out → OutGuard s host out)
    (implementation : LibraryImplementation) (metadata : LibraryMetadata)
    (config : LaunchConfig) (configAccepted : config∈launchConfigs)
    (before : Memory (Cell ℝ)) (input : Input ℝ s)
    (rep : Represents input (prepare allocate s host provided validOut before).buffers before) :
    OutputPost s provided (prepare allocate s host provided validOut before).allocation.caller before
      (allocatedHostCall realArithmetic allocate s host provided validOut
        (config.tile configAccepted) (implementation.select s metadata) before) (mathematical input) := by
  have reference := host_word_post realArithmetic (config.tile configAccepted)
    (implementation.select s metadata) (prepare allocate s host provided validOut before).allocation input rep
  have values : wordReference realArithmetic input (implementation.select s metadata)=mathematical input := by
    funext r n
    exact reference_real input (implementation.select s metadata) r n
  rw [values] at reference
  exact reference

#axiomsClean host_word_post
#axiomsClean matmul_dequantize_int8_correct
end
end VeriTile.Bench.Repairs.DequantizeMatmul

namespace VeriTile.Bench.Repairs.DequantizeMatmul.Specification
open Direct
noncomputable section

theorem python_checks_iff_preconditions (a b scale : Argument) (out : OutputArgument) :
    checkInputs a b scale out=.ok () ↔ AdmittedArguments a b scale out := checkInputs_iff a b scale out

/-- Complete mathematical source/grid/wrapper result for the checked direct
contract, including provided-output identity, original Tensor inputs, and frame.
The external library implementation selects its complete contraction plan from
metadata; realArithmetic is its explicit mathematical interpretation only. -/
theorem dequantize_matmul_wrapper_correct (c : Request) (storageBase : Nat → Nat)
    (la : TensorLaws storageBase c.a) (lb : TensorLaws storageBase c.b) (ls : TensorLaws storageBase c.scale)
    (allocate : Allocator ℝ) (before : Memory (Cell ℝ)) (typed : InputStorage c before)
    (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key)
    (implementation : LibraryImplementation) (metadata : LibraryMetadata)
    (config : LaunchConfig) (member : config∈launchConfigs) :
    ∃ result,checkedRun c realArithmetic allocate caller alive blive slive (config.tile member)
      (implementation.select (requestShape c) metadata) before=.ok result ∧
      result.tensor.device=c.a.info.metadata.device.index ∧
      OutputPost (requestShape c) (provided c) caller before result (mathematical (logicalInput c 0 before)) ∧
      IndexPost (requestShape c) (config.tile member)
        (canonical2 (columns c.a.info) (columns c.b.info)) (canonical2 (columns c.a.info) (columns c.b.info))
        (canonical1 (columns c.b.info)) := by
  obtain ⟨result,success,device,post⟩:=checked_word_result c storageBase la lb ls realArithmetic allocate before typed
    caller alive blive slive (config.tile member) (implementation.select (requestShape c) metadata)
  refine ⟨result,success,device,?_,index_post c config member⟩
  have values : wordReference realArithmetic (logicalInput c 0 before) (implementation.select (requestShape c) metadata)=
      mathematical (logicalInput c 0 before) := by
    funext r n;exact reference_real _ _ r n
  change OutputPost (requestShape c) (provided c) caller before result
    (wordReference realArithmetic (logicalInput c 0 before) (implementation.select (requestShape c) metadata)) at post
  rw [values] at post
  exact post

#axiomsClean dequantize_matmul_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeMatmul.Specification
