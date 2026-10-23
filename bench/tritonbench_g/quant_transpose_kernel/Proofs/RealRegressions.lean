import bench.tritonbench_g.quant_transpose_kernel.Correct

local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace QuantTransposeReal.Regression
open QuantTransposeReal
open QuantTransposeFinite (Input Allocations Config Stage Environment parameters)
open QuantizeGlobalReal (Cell Memory Prepared readInputs readInverse emit maximumOf applyWrites store)
noncomputable section
open Classical
set_option maxHeartbeats 20000000
set_option maxRecDepth 64000

def ready : Environment := fun _ => none
def failOnly (where_ : Stage) : Environment :=
  fun stage => if stage=where_ then some "fixture failure" else none

def oneInput : Input := ⟨[1,1],[1,1],[0],true,0,0,0,1⟩
def stridedInput : Input := ⟨[2,2],[3,1],[0,1,3,4],true,0,0,0,5⟩
def alloc : Allocations := ⟨"maximum","inverse","output",fun _ => 91,0,0,0,0⟩
def singleMemory (v : ℝ) : Memory := fun r i =>
  if r="x" ∧ i=0 then some (.real v) else none
/-- Storage offset 2 is poison outside the logical view and must not be read. -/
def stridedMemory : Memory := fun r i =>
  if r="x" then some (.real (if i=0 then 1 else if i=1 then 2
    else if i=2 then ((2^63 : Int) : ℝ) else if i=3 then 3 else 4)) else none

def preparedOne (v : ℝ) : Prepared :=
  ⟨|v|,QuantizationReal.reciprocal |v|,
    allocated oneInput alloc (singleMemory v) |v| (QuantizationReal.reciprocal |v|)⟩
def preparedStrided : Prepared :=
  ⟨4,QuantizationReal.reciprocal 4,
    allocated stridedInput alloc stridedMemory 4 (QuantizationReal.reciprocal 4)⟩

theorem fixture_metadata : Metadata oneInput ∧ Metadata stridedInput ∧ alloc.Fresh "x" := by
  simp [Metadata,oneInput,stridedInput,QuantTransposeFinite.viewOffsets,Input.numel,
    Allocations.Fresh,alloc]
  decide +kernel

theorem duplicate_configs_preserved : Config.first≠Config.second ∧
    Config.first.blockM=Config.second.blockM ∧ Config.first.blockN=Config.second.blockN ∧
    Config.first.groupM=Config.second.groupM := by decide +kernel

theorem one_indices : QuantTransposeFinite.laneIndices (parameters oneInput alloc "x") 0
    (List.range (128*128))=[0] := by decide +kernel
theorem strided_indices : QuantTransposeFinite.laneIndices (parameters stridedInput alloc "x") 0
    (List.range (128*128))=[0,2,1,3] := by decide +kernel

theorem one_programs (c : Config) : QuantTransposeFinite.programs (parameters oneInput alloc "x") c=1 := by
  cases c <;> decide +kernel
theorem strided_programs (c : Config) : QuantTransposeFinite.programs (parameters stridedInput alloc "x") c=1 := by
  cases c <;> decide +kernel

theorem one_setup : setup ready oneInput=.ok () := by
  simp [setup,primitive,ready,oneInput,Input.M,Input.N,QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]
theorem strided_setup : setup ready stridedInput=.ok () := by
  simp [setup,primitive,ready,stridedInput,Input.M,Input.N,QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]

theorem prepare_one (v : ℝ) :
    prepare ready oneInput alloc (singleMemory v) "x"=.ok (preparedOne v) := by
  simp [prepare,primitive,ready,oneInput,inputView,readInputs,QuantizeGlobalReal.readInput,
    singleMemory,maximumOf,preparedOne,Input.sam,Input.san]

theorem unit_result : QuantizationReal.result 1 (QuantizationReal.reciprocal 1)=some 127 := by
  norm_num [QuantizationReal.result,QuantizationReal.reciprocal,QuantizationReal.llrint64,
    QuantizationReal.nearestEven,QuantizationReal.int64Representable]
  decide +kernel

theorem quarter_results :
    QuantizationReal.result 1 (QuantizationReal.reciprocal 4)=some 32 ∧
    QuantizationReal.result 2 (QuantizationReal.reciprocal 4)=some 64 ∧
    QuantizationReal.result 3 (QuantizationReal.reciprocal 4)=some 95 ∧
    QuantizationReal.result 4 (QuantizationReal.reciprocal 4)=some 127 := by
  norm_num [QuantizationReal.result,QuantizationReal.reciprocal,QuantizationReal.llrint64,
    QuantizationReal.nearestEven,QuantizationReal.int64Representable]
  decide +kernel

theorem zero_source_program (c : Config) :
    sourceProgramPlan oneInput alloc "x" c 0 (preparedOne 0).initial=
      .error (.conversionUndefined 0) := by
  rw [source_program_correspondence]
  unfold programPlan
  rw [one_indices]
  simp [readOutputs,readInput,QuantizeGlobalReal.readInput,readInverse,preparedOne,oneInput,
    inputView,allocated,QuantizeGlobalReal.allocated,allocations,alloc,singleMemory,
    QuantTransposeFinite.inputOffset,parameters,Input.M,Input.sam,Input.san,
    emit,QuantizationReal.zero_normalizer_has_no_scalar_value]

theorem zero_wrapper_observation (c : Config) :
    executeHost ready oneInput alloc (singleMemory 0) "x" [] c=
      .observed (.conversionUndefined 0) := by
  simp [executeHost,prepare_one,one_setup,executeTrials,executeLaunch,one_programs,
    executeGrid,zero_source_program,ready]

theorem zero_allocation_failure (c : Config) :
    executeHost (failOnly .outputAllocation) oneInput alloc (singleMemory 0) "x" [] c=
      .failed (.preparation (.primitive .outputAllocation "fixture failure")) := by
  simp [executeHost,prepare,primitive,failOnly,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf]

theorem zero_compilation_failure (c : Config) :
    executeHost (failOnly .compilation) oneInput alloc (singleMemory 0) "x" [] c=
      .failed (.setup (.primitive .compilation "fixture failure")) := by
  simp [executeHost,prepare,setup,primitive,failOnly,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf,Input.sam,Input.san]

theorem zero_launch_failure (c : Config) :
    executeHost (failOnly .launch) oneInput alloc (singleMemory 0) "x" [] c=
      .failed (.setup (.primitive .launch "fixture failure")) := by
  simp [executeHost,prepare,setup,primitive,failOnly,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf,Input.sam,Input.san,Input.M,Input.N,
    QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]

theorem strided_host_maximum :
    prepare ready stridedInput alloc stridedMemory "x"=.ok preparedStrided := by
  norm_num [prepare,primitive,ready,stridedInput,inputView,readInputs,QuantizeGlobalReal.readInput,
    stridedMemory,maximumOf,preparedStrided,Input.sam,Input.san]

/-- The actual strided input is transposed; the poison storage cell is absent
from both the logical producer and the kernel's reads. -/
theorem strided_source_program (c : Config) :
    sourceProgramPlan stridedInput alloc "x" c 0 preparedStrided.initial=
      .ok [(0,32),(2,64),(1,95),(3,127)] := by
  rw [source_program_correspondence]
  unfold programPlan
  rw [strided_indices]
  simp [readOutputs,readInput,QuantizeGlobalReal.readInput,readInverse,preparedStrided,stridedInput,
    inputView,allocated,QuantizeGlobalReal.allocated,allocations,alloc,stridedMemory,
    QuantTransposeFinite.inputOffset,parameters,Input.M,Input.sam,Input.san,
    emit,quarter_results.1,quarter_results.2.1,quarter_results.2.2.1,quarter_results.2.2.2]

theorem strided_wrapper_returns (c : Config) :
    executeHost ready stridedInput alloc stridedMemory "x" [] c=
      .returned (outputTensor stridedInput alloc,maximumTensor stridedInput alloc)
        (applyWrites alloc.output [(0,32),(2,64),(1,95),(3,127)] preparedStrided.initial) := by
  simp [executeHost,strided_host_maximum,strided_setup,executeTrials,executeLaunch,strided_programs,
    executeGrid,strided_source_program,ready]

theorem strided_returned_values :
    let final := applyWrites alloc.output [(0,32),(2,64),(1,95),(3,127)] preparedStrided.initial
    final alloc.output 0=some (.byte 32) ∧ final alloc.output 1=some (.byte 95) ∧
    final alloc.output 2=some (.byte 64) ∧ final alloc.output 3=some (.byte 127) ∧
    final "x" 2=some (.real ((2^63 : Int) : ℝ)) := by
  simp [applyWrites,store,preparedStrided,allocated,QuantizeGlobalReal.allocated,allocations,
    alloc,stridedMemory]

def cpuInput : Input := {oneInput with isCuda:=false}
def badStrideInput : Input := {oneInput with strides:=[2,2]}
def wrongRankInput : Input := {oneInput with shape:=[1],strides:=[1]}
def emptyInput : Input := ⟨[0,1],[1,1],[],true,0,0,0,0⟩

theorem zero_cpu_pointer_before_launch (c : Config) :
    executeHost (failOnly .launch) cpuInput alloc (singleMemory 0) "x" [] c=
      .failed (.setup .cpuPointer) := by
  simp [executeHost,prepare,setup,primitive,failOnly,cpuInput,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf,Input.sam,Input.san,Input.M,Input.N,
    QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]

theorem zero_stride_assertion (c : Config) :
    executeHost ready badStrideInput alloc (singleMemory 0) "x" [] c=
      .failed (.preparation .inputStrideAssertion) := by
  simp [executeHost,prepare,primitive,ready,badStrideInput,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf,Input.sam,Input.san]

theorem allocation_before_stride_assertion (c : Config) :
    executeHost (failOnly .outputAllocation) badStrideInput alloc (singleMemory 0) "x" [] c=
      .failed (.preparation (.primitive .outputAllocation "fixture failure")) := by
  simp [executeHost,prepare,primitive,failOnly,badStrideInput,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf]

theorem shape_unpack_before_allocation (c : Config) :
    executeHost (failOnly .outputAllocation) wrongRankInput alloc (singleMemory 0) "x" [] c=
      .failed (.preparation .shapeUnpack) := by
  simp [executeHost,prepare,primitive,failOnly,wrongRankInput,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf]

theorem reciprocal_before_shape_unpack (c : Config) :
    executeHost (failOnly .reciprocal) wrongRankInput alloc (singleMemory 0) "x" [] c=
      .failed (.preparation (.primitive .reciprocal "fixture failure")) := by
  simp [executeHost,prepare,primitive,failOnly,wrongRankInput,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf]

theorem empty_maximum_before_unsqueeze (c : Config) :
    executeHost (failOnly .unsqueeze) emptyInput alloc (singleMemory 0) "x" [] c=
      .failed (.preparation .emptyMaximum) := by
  simp [executeHost,prepare,primitive,failOnly,emptyInput,readInputs,maximumOf]

def hugeCpuInput : Input := {cpuInput with shape:=[2^38,1]}
theorem grid_parse_before_cpu_pointer :
    setup ready hugeCpuInput=.error (.gridLimit (2^31)) := by
  norm_num [setup,primitive,ready,hugeCpuInput,cpuInput,oneInput,Input.M,Input.N,
    QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]

theorem driver_context_before_grid_parse :
    setup (failOnly .driverContext) hugeCpuInput=
      .error (.primitive .driverContext "fixture failure") := by
  simp [setup,primitive,failOnly]

theorem partial_group_rotation :
    [QuantTransposeFinite.tileRow 11 2 16,QuantTransposeFinite.tileRow 11 2 17,
      QuantTransposeFinite.tileRow 11 2 18]=[9,10,8] := by decide +kernel

theorem strided_exit_hook_failure (c : Config) :
    executeHost (failOnly .launchExitHook) stridedInput alloc stridedMemory "x" [] c=
      .failed (.exitHook "fixture failure") := by
  have hp : prepare (failOnly .launchExitHook) stridedInput alloc stridedMemory "x"=.ok preparedStrided := by
    norm_num [prepare,primitive,failOnly,stridedInput,inputView,readInputs,QuantizeGlobalReal.readInput,
      stridedMemory,maximumOf,preparedStrided,Input.sam,Input.san]
    simp
  have hs : setup (failOnly .launchExitHook) stridedInput=.ok () := by
    simp [setup,primitive,failOnly,stridedInput,Input.M,Input.N,QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]
  simp [executeHost,hp,hs,failOnly]

/-- An undefined zero-normalizer value must not hide an actual runtime hook failure. -/
theorem zero_exit_hook_failure (c : Config) :
    executeHost (failOnly .launchExitHook) oneInput alloc (singleMemory 0) "x" [] c=
      .failed (.exitHook "fixture failure") := by
  simp [executeHost,prepare,setup,primitive,failOnly,oneInput,inputView,readInputs,
    QuantizeGlobalReal.readInput,singleMemory,maximumOf,Input.sam,Input.san,Input.M,Input.N,
    QuantTransposeFinite.gridM,QuantTransposeFinite.gridN]

#axiomsClean fixture_metadata
#axiomsClean duplicate_configs_preserved
#axiomsClean one_indices
#axiomsClean strided_indices
#axiomsClean one_programs
#axiomsClean strided_programs
#axiomsClean one_setup
#axiomsClean strided_setup
#axiomsClean prepare_one
#axiomsClean unit_result
#axiomsClean quarter_results
#axiomsClean zero_source_program
#axiomsClean zero_wrapper_observation
#axiomsClean zero_allocation_failure
#axiomsClean zero_compilation_failure
#axiomsClean zero_launch_failure
#axiomsClean strided_host_maximum
#axiomsClean strided_source_program
#axiomsClean strided_wrapper_returns
#axiomsClean strided_returned_values
#axiomsClean zero_cpu_pointer_before_launch
#axiomsClean zero_stride_assertion
#axiomsClean allocation_before_stride_assertion
#axiomsClean shape_unpack_before_allocation
#axiomsClean reciprocal_before_shape_unpack
#axiomsClean empty_maximum_before_unsqueeze
#axiomsClean grid_parse_before_cpu_pointer
#axiomsClean driver_context_before_grid_parse
#axiomsClean partial_group_rotation
#axiomsClean strided_exit_hook_failure
#axiomsClean zero_exit_hook_failure
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantTransposeReal.Regression
