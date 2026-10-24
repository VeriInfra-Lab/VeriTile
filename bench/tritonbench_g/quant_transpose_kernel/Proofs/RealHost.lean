import bench.tritonbench_g.quant_transpose_kernel.Proofs.RealExecution

local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace QuantTransposeReal
open QuantTransposeFinite (Input Allocations Config Parameters parameters Stage Environment)
open QuantizeGlobalReal (Cell Memory Write Issue Tensor PayloadKind Prepared maximumOf
  readInputs Outside outside_refl writes_outside finalCell writes_final_cell writes_frame)
noncomputable section
open Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 64000

def outputTensor (a : Input) (alloc : Allocations) : Tensor :=
  ⟨alloc.output,[a.N,a.M],[max a.M 1,1],.int8,true,alloc.currentDevice,0,a.numel⟩
def maximumTensor (a : Input) (alloc : Allocations) : Tensor :=
  ⟨alloc.maximum,[1],[1],.mathematicalReal,a.isCuda,a.deviceIndex,0,1⟩
def scalarMaximumTensor (a : Input) (alloc : Allocations) : Tensor :=
  ⟨alloc.maximum,[],[],.mathematicalReal,a.isCuda,a.deviceIndex,0,1⟩

theorem maximum_unsqueeze_alias (a : Input) (alloc : Allocations) :
    maximumTensor a alloc=QuantizeGlobalReal.unsqueezeZero (scalarMaximumTensor a alloc) := rfl

def OutputShapeOK (a : Input) (alloc : Allocations) : Prop :=
  (outputTensor a alloc).shape[0]?.getD 0=a.N ∧ (outputTensor a alloc).shape[1]?.getD 0=a.M
def OutputStrideOK (a : Input) (alloc : Allocations) : Prop :=
  (outputTensor a alloc).strides[0]?.getD 0=1 ∨ (outputTensor a alloc).strides[1]?.getD 0=1
@[simp] theorem output_shape_ok (a : Input) (alloc : Allocations) : OutputShapeOK a alloc := by
  simp [OutputShapeOK,outputTensor]
@[simp] theorem output_stride_ok (a : Input) (alloc : Allocations) : OutputStrideOK a alloc := by
  simp [OutputStrideOK,outputTensor]

theorem matrix_count (a : Input) (rank : a.shape.length=2) : a.numel=a.M*a.N := by
  rcases List.length_eq_two.mp rank with ⟨m,n,shape⟩
  simp [Input.numel,Input.M,Input.N,shape]
theorem output_numel (a : Input) (alloc : Allocations) (rank : a.shape.length=2) :
    (outputTensor a alloc).shape.prod=a.numel := by
  simp [outputTensor,matrix_count a rank,Nat.mul_comm]

def allocated (a : Input) (alloc : Allocations) (m : Memory)
    (maximum : ℝ) (inverse : Option ℝ) : Memory :=
  QuantizeGlobalReal.allocated (inputView a) (allocations alloc) m maximum inverse

inductive HostFailure where
  | primitive (stage : Stage) (detail : String)
  | inputObservation (issue : Issue)
  | emptyMaximum
  | shapeUnpack
  | outputShapeAssertion
  | inputStrideAssertion
  | outputStrideAssertion
  | gridLimit (requested : Nat)
  | cpuPointer
  deriving DecidableEq, Repr

def primitive (env : Environment) (stage : Stage) : Except HostFailure Unit :=
  match env stage with
  | none => .ok ()
  | some detail => .error (.primitive stage detail)

/-- Exact host mathematical operations and source order. Output shape/stride
assertions use the actual fresh allocation metadata. CPU is not asserted here. -/
def prepare (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) : Except HostFailure Prepared := do
  primitive env .abs
  let values ← (readInputs (inputView a) m x a.logicalOffsets).mapError HostFailure.inputObservation
  primitive env .maximum
  let maximum ← match maximumOf values with
    | none => .error .emptyMaximum
    | some maximum => .ok maximum
  primitive env .unsqueeze
  primitive env .reciprocal
  let inverse := QuantizationReal.reciprocal maximum
  if a.shape.length≠2 then throw .shapeUnpack
  primitive env .outputAllocation
  if ¬OutputShapeOK a alloc then throw .outputShapeAssertion
  if ¬(a.sam=1 ∨ a.san=1) then throw .inputStrideAssertion
  if ¬OutputStrideOK a alloc then throw .outputStrideAssertion
  pure ⟨maximum,inverse,allocated a alloc m maximum inverse⟩

def PrimitiveFailure (env : Environment) (stage : Stage) (why : HostFailure) : Prop :=
  ∃ detail, env stage=some detail ∧ why=.primitive stage detail

def PreparationFailure (env : Environment) (a : Input) (m : Memory)
    (x : String) (why : HostFailure) : Prop :=
  PrimitiveFailure env .abs why ∨
  (env .abs=none ∧
    ((∃ issue, readInputs (inputView a) m x a.logicalOffsets=.error issue ∧ why=.inputObservation issue) ∨
     ∃ values, readInputs (inputView a) m x a.logicalOffsets=.ok values ∧
      (PrimitiveFailure env .maximum why ∨
       (env .maximum=none ∧
        ((maximumOf values=none ∧ why=.emptyMaximum) ∨
         ∃ maximum, maximumOf values=some maximum ∧
          (PrimitiveFailure env .unsqueeze why ∨
           (env .unsqueeze=none ∧
            (PrimitiveFailure env .reciprocal why ∨
             (env .reciprocal=none ∧
              ((a.shape.length≠2 ∧ why=.shapeUnpack) ∨
               (a.shape.length=2 ∧
                (PrimitiveFailure env .outputAllocation why ∨
                 (env .outputAllocation=none ∧ ¬(a.sam=1 ∨ a.san=1) ∧ why=.inputStrideAssertion)))))))))))))

def PreparedSpec (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (p : Prepared) : Prop :=
  env .abs=none ∧ env .maximum=none ∧ env .unsqueeze=none ∧
  env .reciprocal=none ∧ env .outputAllocation=none ∧ a.shape.length=2 ∧ (a.sam=1 ∨ a.san=1) ∧
  ∃ values, readInputs (inputView a) m x a.logicalOffsets=.ok values ∧
    maximumOf values=some p.maximum ∧ p.inverse=QuantizationReal.reciprocal p.maximum ∧
    p.initial=allocated a alloc m p.maximum p.inverse

def PreparationSpec (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) : Except HostFailure Prepared → Prop
  | .error why => PreparationFailure env a m x why
  | .ok p => PreparedSpec env a alloc m x p

theorem prepare_all_outcomes (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) :
    PreparationSpec env a alloc m x (prepare env a alloc m x) := by
  cases ha : env .abs with
  | some detail => simp [prepare,primitive,ha,PreparationSpec,PreparationFailure,PrimitiveFailure]
  | none =>
    cases hv : readInputs (inputView a) m x a.logicalOffsets with
    | error issue => simp [prepare,primitive,ha,hv,PreparationSpec,PreparationFailure,PrimitiveFailure]
    | ok values =>
      cases hm : env .maximum with
      | some detail => simp [prepare,primitive,ha,hv,hm,PreparationSpec,PreparationFailure,PrimitiveFailure]
      | none =>
        cases hx : maximumOf values with
        | none => simp [prepare,primitive,ha,hv,hm,hx,PreparationSpec,PreparationFailure,PrimitiveFailure]
        | some maximum =>
          cases hu : env .unsqueeze with
          | some detail => simp [prepare,primitive,ha,hv,hm,hx,hu,PreparationSpec,PreparationFailure,PrimitiveFailure]
          | none =>
            cases hr : env .reciprocal with
            | some detail => simp [prepare,primitive,ha,hv,hm,hx,hu,hr,PreparationSpec,PreparationFailure,PrimitiveFailure]
            | none =>
              by_cases hz : a.shape.length=2
              · cases ho : env .outputAllocation with
                | some detail => simp [prepare,primitive,ha,hv,hm,hx,hu,hr,hz,ho,PreparationSpec,PreparationFailure,PrimitiveFailure]
                | none =>
                  by_cases stride : a.sam=1 ∨ a.san=1 <;>
                    simp [prepare,primitive,ha,hv,hm,hx,hu,hr,hz,ho,stride,
                      PreparationSpec,PreparationFailure,PrimitiveFailure,PreparedSpec]
              · simp [prepare,primitive,ha,hv,hm,hx,hu,hr,hz,PreparationSpec,PreparationFailure,PrimitiveFailure]

/-- Scalar-independent source runtime boundary. Grid parsing precedes CPU
pointer packing. These primitives are explicit environment outcomes. -/
def setup (env : Environment) (a : Input) : Except HostFailure Unit := do
  primitive env .signature
  primitive env .compilation
  primitive env .driverContext
  let requested := QuantTransposeFinite.gridM a.M*QuantTransposeFinite.gridN a.N
  if ¬requested<2^31 then throw (.gridLimit requested)
  primitive env .launchEnterHook
  primitive env .pointerPacking
  if a.isCuda=false then throw .cpuPointer
  primitive env .launch

/-- Independent chronological list of possible first failures. -/
def setupIssues (env : Environment) (a : Input) : List (Option HostFailure) :=
  [ (env .signature).map (.primitive .signature),
    (env .compilation).map (.primitive .compilation),
    (env .driverContext).map (.primitive .driverContext),
    if QuantTransposeFinite.gridM a.M*QuantTransposeFinite.gridN a.N<2^31 then none
      else some (.gridLimit (QuantTransposeFinite.gridM a.M*QuantTransposeFinite.gridN a.N)),
    (env .launchEnterHook).map (.primitive .launchEnterHook),
    (env .pointerPacking).map (.primitive .pointerPacking),
    if a.isCuda then none else some .cpuPointer,
    (env .launch).map (.primitive .launch) ]

def firstIssue : List (Option HostFailure) → Option HostFailure
  | [] => none
  | none::rest => firstIssue rest
  | some issue::_ => some issue

def SetupReady (env : Environment) (a : Input) : Prop := firstIssue (setupIssues env a)=none
def SetupFailure (env : Environment) (a : Input) (why : HostFailure) : Prop :=
  firstIssue (setupIssues env a)=some why

theorem setup_all_outcomes (env : Environment) (a : Input) :
    match setup env a with
    | .error why => SetupFailure env a why
    | .ok () => SetupReady env a := by
  cases hs : env .signature with
  | some detail => simp [setup,primitive,hs,SetupFailure,setupIssues,firstIssue]
  | none =>
    cases hc : env .compilation with
    | some detail => simp [setup,primitive,hs,hc,SetupFailure,setupIssues,firstIssue]
    | none =>
      cases hd : env .driverContext with
      | some detail => simp [setup,primitive,hs,hc,hd,SetupFailure,setupIssues,firstIssue]
      | none =>
        by_cases hg : QuantTransposeFinite.gridM a.M*QuantTransposeFinite.gridN a.N<2147483648
        · cases he : env .launchEnterHook with
          | some detail => simp [setup,primitive,hs,hc,hd,hg,he,SetupFailure,setupIssues,firstIssue]
          | none =>
            cases hp : env .pointerPacking with
            | some detail => simp [setup,primitive,hs,hc,hd,hg,he,hp,SetupFailure,setupIssues,firstIssue]
            | none =>
              cases hcuda : a.isCuda
              · simp [setup,primitive,hs,hc,hd,hg,he,hp,hcuda,SetupFailure,setupIssues,firstIssue]
              · cases hl : env .launch <;>
                  simp [setup,primitive,hs,hc,hd,hg,he,hp,hcuda,hl,
                    SetupFailure,SetupReady,setupIssues,firstIssue]
        · simp [setup,primitive,hs,hc,hd,hg,SetupFailure,setupIssues,firstIssue]

inductive Failure where
  | preparation (why : HostFailure)
  | setup (why : HostFailure)
  | exitHook (detail : String)
inductive Outcome where
  | failed (why : Failure)
  | observed (issue : Issue)
  | returned (tensors : Tensor × Tensor) (memory : Memory)

/-- Runtime launch and exit-hook classification precedes mathematical value
observations. This is not a claim that a failed launch/exit had no kernel effects;
no failed-state memory or rollback is exposed. The supplied trace is interpreted
only after that runtime boundary returns successfully. -/
def executeHost (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (trials : List Config) (last : Config) : Outcome :=
  match prepare env a alloc m x with
  | .error why => .failed (.preparation why)
  | .ok p => match setup env a with
    | .error why => .failed (.setup why)
    | .ok () => match env .launchExitHook with
      | some detail => .failed (.exitHook detail)
      | none => match executeTrials a alloc x (trials++[last]) p.initial with
        | .error issue => .observed issue
        | .ok final => .returned (outputTensor a alloc,maximumTensor a alloc) final

def Metadata (a : Input) : Prop :=
  a.strides.length=a.shape.length ∧
  a.logicalOffsets=QuantTransposeFinite.viewOffsets a.shape a.strides ∧
  a.logicalOffsets.length=a.numel

structure ReturnedSpec (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (trials : List Config) (last : Config)
    (tensors : Tensor × Tensor) (final : Memory) : Prop where
  returned : tensors=(outputTensor a alloc,maximumTensor a alloc)
  prepared : ∃ p, PreparedSpec env a alloc m x p ∧ SetupReady env a ∧ env .launchExitHook=none ∧
    ∃ writes, trialsPlan a alloc x p.initial (trials++[last])=.ok writes ∧
      (∀ r i, final r i=finalCell alloc.output writes p.initial r i) ∧
      (∀ r i, (r≠alloc.output ∨ ∀ w∈writes, w.1≠i) → final r i=p.initial r i) ∧
      (∀ w∈writes, w.1<a.numel) ∧
      (∀ i, i<a.numel → i∈writes.map Prod.fst) ∧
      Outside alloc.output final p.initial ∧
      final alloc.maximum 0=some (.real p.maximum) ∧
      final alloc.scale 0=some (.scale p.inverse) ∧
      (∀ r i, r≠alloc.maximum → r≠alloc.scale → r≠alloc.output → final r i=m r i)

def HostSpec (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (trials : List Config) (last : Config) : Outcome → Prop
  | .failed (.preparation why) => PreparationFailure env a m x why
  | .failed (.setup why) => ∃ p, PreparedSpec env a alloc m x p ∧ SetupFailure env a why
  | .failed (.exitHook detail) => ∃ p, PreparedSpec env a alloc m x p ∧ SetupReady env a ∧
      env .launchExitHook=some detail
  | .observed issue => ∃ p, PreparedSpec env a alloc m x p ∧ SetupReady env a ∧
      env .launchExitHook=none ∧ trialsPlan a alloc x p.initial (trials++[last])=.error issue
  | .returned tensors final => ReturnedSpec env a alloc m x trials last tensors final

#specNonCircular PreparationFailure avoiding [prepare,executeHost]
#specNonCircular PreparedSpec avoiding [prepare,executeHost]
#specNonCircular setupIssues avoiding [setup,executeHost]
#specNonCircular SetupReady avoiding [setup,executeHost]
#specNonCircular SetupFailure avoiding [setup,executeHost]
#specNonCircular ReturnedSpec avoiding [prepare,executeHost,executeTrials,executeLaunch,executeGrid,sourceProgramPlan,sourceReadLanes,QuantizeGlobalReal.applyWrites]
#specNonCircular HostSpec avoiding [prepare,executeHost,executeTrials,executeLaunch,executeGrid,sourceProgramPlan,sourceReadLanes,QuantizeGlobalReal.applyWrites]
#axiomsClean maximum_unsqueeze_alias
#axiomsClean output_shape_ok
#axiomsClean output_stride_ok
#axiomsClean matrix_count
#axiomsClean output_numel
#axiomsClean prepare_all_outcomes
#axiomsClean setup_all_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantTransposeReal
