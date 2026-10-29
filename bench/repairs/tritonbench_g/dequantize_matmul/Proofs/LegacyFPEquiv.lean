import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.DirectComplete
import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.Machine
namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 4000000

/-- Selected-plan FP reference, with explicit conversions, rounding, seeds,
all K terms, grouping, output identity and caller frame. Fixed ISA lane
profiles are explicit parameters, not a complete-dot or matrix oracle. -/
specification matmul_dequantize_int8_fp_correct (instruction : FixedInstructionProfile)
    (allocate : Allocator Word) (s : Shape) (host : HostInputs) (provided : Option Descriptor)
    (validOut : ∀ out,provided=some out → OutGuard s host out)
    (implementation : LibraryImplementation) (metadata : LibraryMetadata)
    (config : LaunchConfig) (configAccepted : config∈launchConfigs)
    (before : Memory (Cell Word)) (input : Input Word s)
    (rep : Represents input (prepare allocate s host provided validOut before).buffers before) :
    OutputPost s provided (prepare allocate s host provided validOut before).allocation.caller before
      (allocatedHostCall (fpArithmetic instruction) allocate s host provided validOut
        (config.tile configAccepted) (implementation.select s metadata) before)
      (wordReference (fpArithmetic instruction) input (implementation.select s metadata)) :=
  host_word_post _ _ _ (prepare allocate s host provided validOut before).allocation input rep

/-- The actual machine is driven by the installed implementation's local event
steps. This call contains no whole-run/final-memory correctness callback. -/
def nativeHostCall (abi : LocalEventABI Word σ native) (s : Shape) (b : Buffers)
    (provided : Option Descriptor) (fresh : Descriptor) (plan : LibraryPlan s)
    (allocatedState : σ) : HostReturn Word :=
  ⟨selectOutput provided fresh,abi.observe ((pipelineCode s b plan).native abi allocatedState).2⟩

specification matmul_dequantize_int8_native_correct
    (primitive : PrimitiveOps) (instruction : FixedInstructionProfile)
    (numeric : PrimitiveLeaves primitive instruction)
    (abi : LocalEventABI Word σ primitive.arithmetic)
    (allocate : Allocator Word) (s : Shape) (host : HostInputs) (provided : Option Descriptor)
    (validOut : ∀ out,provided=some out → OutGuard s host out)
    (implementation : LibraryImplementation) (metadata : LibraryMetadata)
    (config : LaunchConfig) (configAccepted : config∈launchConfigs)
    (db df : Dense2 s.K s.N) (ds : Dense1 s.N)
    (index64 : IndexABI s (config.tile configAccepted) db df ds)
    (before : Memory (Cell Word)) (input : Input Word s)
    (rep : Represents input (prepare allocate s host provided validOut before).buffers before)
    (allocatedState : σ)
    (initialBinding : abi.observe allocatedState=(prepare allocate s host provided validOut before).memory) :
    let prepared := prepare allocate s host provided validOut before
    IndexPost s (config.tile configAccepted) db df ds ∧
    OutputPost s provided prepared.allocation.caller before
      (nativeHostCall abi s prepared.buffers provided prepared.fresh
        (implementation.select s metadata) allocatedState)
      (wordReference (fpArithmetic instruction) input (implementation.select s metadata)) := by
  refine ⟨index64.exact, ?_⟩
  let prepared := prepare allocate s host provided validOut before
  have executed := native_pipeline abi s prepared.buffers (config.tile configAccepted)
    (implementation.select s metadata) prepared.allocation.separate allocatedState
  rw [initialBinding] at executed
  have same : nativeHostCall abi s prepared.buffers provided prepared.fresh
      (implementation.select s metadata) allocatedState=
      hostRun primitive.arithmetic s prepared.buffers (config.tile configAccepted)
        (implementation.select s metadata) provided prepared.fresh prepared.memory := by
    unfold nativeHostCall hostRun
    rw [executed]
  change OutputPost s provided prepared.allocation.caller before _ _
  rw [same]
  have reference := host_word_post primitive.arithmetic (config.tile configAccepted)
    (implementation.select s metadata) prepared.allocation input rep
  rw [numeric.numeric.correspondence.arithmetic_eq] at reference ⊢
  exact reference

#axiomsClean matmul_dequantize_int8_fp_correct
#axiomsClean matmul_dequantize_int8_native_correct
end
end VeriTile.Bench.Repairs.DequantizeMatmul

namespace VeriTile.Bench.Repairs.DequantizeMatmul.FloatingSpecification
open Direct
noncomputable section

/-- Source FP32/int8 precision and the external library's selected instruction
plan are retained. This is a software precision interpretation; native primitive
and fixed instruction-lane conformance are separate local obligations. -/
theorem dequantize_matmul_wrapper_fp32 (c : Request) (storageBase : Nat → Nat)
    (la : TensorLaws storageBase c.a) (lb : TensorLaws storageBase c.b) (ls : TensorLaws storageBase c.scale)
    (instruction : FixedInstructionProfile) (allocate : Allocator Word)
    (before : Memory (Cell Word)) (typed : InputStorage c before)
    (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key)
    (implementation : LibraryImplementation) (metadata : LibraryMetadata)
    (config : LaunchConfig) (member : config∈launchConfigs) :
    ∃ result,checkedRun c (fpArithmetic instruction) allocate caller alive blive slive (config.tile member)
      (implementation.select (requestShape c) metadata) before=.ok result ∧
      result.tensor.device=c.a.info.metadata.device.index ∧
      OutputPost (requestShape c) (provided c) caller before result
        (wordReference (fpArithmetic instruction) (logicalInput c (fpArithmetic instruction).zero before)
          (implementation.select (requestShape c) metadata)) ∧
      IndexPost (requestShape c) (config.tile member)
        (canonical2 (columns c.a.info) (columns c.b.info)) (canonical2 (columns c.a.info) (columns c.b.info))
        (canonical1 (columns c.b.info)) := by
  obtain ⟨result,success,device,post⟩:=checked_word_result c storageBase la lb ls (fpArithmetic instruction) allocate
    before typed caller alive blive slive (config.tile member) (implementation.select (requestShape c) metadata)
  exact ⟨result,success,device,post,index_post c config member⟩

/-- Native full-event result follows from local load/store/primitive steps and
an actual metadata-selected library plan. No whole-mm output equality is assumed.
Index bounds are derived from admission, including empty launch branches. -/
theorem dequantize_matmul_native_if_local (c : Request) (storageBase : Nat → Nat)
    (la : TensorLaws storageBase c.a) (lb : TensorLaws storageBase c.b) (ls : TensorLaws storageBase c.scale)
    (primitive : PrimitiveOps) (instruction : FixedInstructionProfile) (numeric : PrimitiveLeaves primitive instruction)
    (abi : LocalEventABI Word σ primitive.arithmetic) (allocate : Allocator Word)
    (before : Memory (Cell Word)) (typed : InputStorage c before)
    (caller : Nat → Prop) (alive : caller c.a.key) (blive : caller c.b.key) (slive : caller c.scale.key)
    (implementation : LibraryImplementation) (metadata : LibraryMetadata)
    (config : LaunchConfig) (member : config∈launchConfigs) (allocatedState : σ)
    (binding : abi.observe allocatedState=(prepare allocate (requestShape c) (host c caller alive blive slive)
      (provided c) (output_guard c caller alive blive slive) before).memory) :
    let prepared:=prepare allocate (requestShape c) (host c caller alive blive slive)
      (provided c) (output_guard c caller alive blive slive) before
    let result:=nativeHostCall abi (requestShape c) prepared.buffers (provided c) prepared.fresh
      (implementation.select (requestShape c) metadata) allocatedState
    checkTensors c.a.info c.b.info c.scale.info (c.out.map TensorView.info)=.ok () ∧
      result.tensor.device=c.a.info.metadata.device.index ∧
      OutputPost (requestShape c) (provided c) caller before result
        (wordReference (fpArithmetic instruction) (logicalInput c primitive.arithmetic.zero before)
          (implementation.select (requestShape c) metadata)) ∧
      IndexPost (requestShape c) (config.tile member)
        (canonical2 (columns c.a.info) (columns c.b.info)) (canonical2 (columns c.a.info) (columns c.b.info))
        (canonical1 (columns c.b.info)) := by
  dsimp only
  let prepared:=prepare allocate (requestShape c) (host c caller alive blive slive)
    (provided c) (output_guard c caller alive blive slive) before
  let plan:=implementation.select (requestShape c) metadata
  have execution:=native_pipeline abi (requestShape c) prepared.buffers (config.tile member) plan prepared.allocation.separate allocatedState
  rw [binding] at execution
  have same : nativeHostCall abi (requestShape c) prepared.buffers (provided c) prepared.fresh plan allocatedState=
      allocatedHostCall primitive.arithmetic allocate (requestShape c) (host c caller alive blive slive)
        (provided c) (output_guard c caller alive blive slive) (config.tile member) plan before := by
    unfold nativeHostCall allocatedHostCall hostRun
    rw [execution]
  change checkTensors _ _ _ _=.ok () ∧ _ ∧ OutputPost _ _ _ _ (nativeHostCall abi _ prepared.buffers _ prepared.fresh plan allocatedState) _ ∧ _
  rw [same]
  refine ⟨(checkTensors_iff _ _ _ _).mpr c.contract,
    returned_device c primitive.arithmetic allocate caller alive blive slive (config.tile member) plan before,?_,index_post c config member⟩
  have post:=host_word_post primitive.arithmetic (config.tile member) plan prepared.allocation
    (logicalInput c primitive.arithmetic.zero before)
    (input_represents c storageBase la lb ls primitive.arithmetic.zero before typed allocate caller alive blive slive)
  have values : wordReference primitive.arithmetic (logicalInput c primitive.arithmetic.zero before) plan=
      wordReference (fpArithmetic instruction) (logicalInput c primitive.arithmetic.zero before) plan := by
    rw [numeric.numeric.correspondence.arithmetic_eq]
  rw [values] at post
  exact post

#axiomsClean dequantize_matmul_wrapper_fp32
#axiomsClean dequantize_matmul_native_if_local
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeMatmul.FloatingSpecification
