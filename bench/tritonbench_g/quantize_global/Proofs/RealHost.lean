import bench.tritonbench_g.quantize_global.Proofs.RealExecution

local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace QuantizeGlobalReal
open QuantizeGlobalFinite (Input Allocations Config Stage Environment)
noncomputable section
open Classical
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000

@[simp] theorem except_map_error_ok {ε δ α : Type} (f : ε → δ) (v : α) :
    (Except.ok v : Except ε α).mapError f=Except.ok v := rfl
@[simp] theorem except_map_error_error {ε δ α : Type} (f : ε → δ) (e : ε) :
    (Except.error e : Except ε α).mapError f=Except.error (f e) := rfl
@[simp] theorem except_throw {ε α : Type} (e : ε) :
    (throw e : Except ε α)=Except.error e := rfl

/-- Exact mathematical abs and maximum of the logical view. Empty reduction
has no value. No native numerical callback supplies this result. -/
def maximumOf (values : List (Nat × ℝ)) : Option ℝ :=
  match values.map (fun item => |item.2|) with
  | [] => none
  | v::vs => some (vs.foldl max v)

def allocated (a : Input) (alloc : Allocations) (m : Memory)
    (maximum : ℝ) (inverse : Option ℝ) : Memory := fun r i =>
  if r=alloc.maximum ∧ i=0 then some (.real maximum)
  else if r=alloc.scale ∧ i=0 then some (.scale inverse)
  else if r=alloc.output ∧ i<a.numel then some (.byte (alloc.initialOutput i))
  else m r i

inductive HostFailure where
  | primitive (stage : Stage) (detail : String)
  | inputObservation (issue : Issue)
  | emptyMaximum
  | missingSize
  | cudaAssertion
  deriving DecidableEq, Repr

def primitive (env : Environment) (stage : Stage) : Except HostFailure Unit :=
  match env stage with
  | none => .ok ()
  | some detail => .error (.primitive stage detail)

structure Prepared where
  maximum : ℝ
  inverse : Option ℝ
  initial : Memory

/-- Exact source producer and first-error order. Zero reciprocal is carried
through allocation/assertion/setup as an undefined value, not raised here. -/
def prepare (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) : Except HostFailure Prepared := do
  primitive env .abs
  let values ← (readInputs a m x a.logicalOffsets).mapError HostFailure.inputObservation
  primitive env .maximum
  let maximum ← match maximumOf values with
    | none => .error .emptyMaximum
    | some maximum => .ok maximum
  primitive env .unsqueeze
  primitive env .reciprocal
  let inverse := QuantizationReal.reciprocal maximum
  if a.shape=[] then throw .missingSize
  primitive env .outputAllocation
  if a.isCuda=false then throw .cudaAssertion
  pure ⟨maximum,inverse,allocated a alloc m maximum inverse⟩

def PrimitiveFailure (env : Environment) (stage : Stage) (why : HostFailure) : Prop :=
  ∃ detail, env stage=some detail ∧ why=.primitive stage detail

/-- Declarative first-reached source failure. Later stages never override it. -/
def PreparationFailure (env : Environment) (a : Input) (m : Memory)
    (x : String) (why : HostFailure) : Prop :=
  PrimitiveFailure env .abs why ∨
  (env .abs=none ∧
    ((∃ issue, readInputs a m x a.logicalOffsets=.error issue ∧ why=.inputObservation issue) ∨
     ∃ values, readInputs a m x a.logicalOffsets=.ok values ∧
      (PrimitiveFailure env .maximum why ∨
       (env .maximum=none ∧
        ((maximumOf values=none ∧ why=.emptyMaximum) ∨
         ∃ maximum, maximumOf values=some maximum ∧
          (PrimitiveFailure env .unsqueeze why ∨
           (env .unsqueeze=none ∧
            (PrimitiveFailure env .reciprocal why ∨
             (env .reciprocal=none ∧
              ((a.shape=[] ∧ why=.missingSize) ∨
               (a.shape≠[] ∧
                (PrimitiveFailure env .outputAllocation why ∨
                 (env .outputAllocation=none ∧ a.isCuda=false ∧ why=.cudaAssertion)))))))))))))

def PreparedSpec (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (p : Prepared) : Prop :=
  env .abs=none ∧ env .maximum=none ∧ env .unsqueeze=none ∧
  env .reciprocal=none ∧ env .outputAllocation=none ∧ a.shape≠[] ∧ a.isCuda=true ∧
  ∃ values, readInputs a m x a.logicalOffsets=.ok values ∧
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
    cases hv : readInputs a m x a.logicalOffsets with
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
              by_cases hz : a.shape=[]
              · simp [prepare,primitive,ha,hv,hm,hx,hu,hr,hz,PreparationSpec,PreparationFailure,PrimitiveFailure]
              · cases ho : env .outputAllocation with
                | some detail => simp [prepare,primitive,ha,hv,hm,hx,hu,hr,hz,ho,PreparationSpec,PreparationFailure,PrimitiveFailure]
                | none =>
                  cases hc : a.isCuda <;>
                    simp [prepare,primitive,ha,hv,hm,hx,hu,hr,hz,ho,hc,
                      PreparationSpec,PreparationFailure,PrimitiveFailure,PreparedSpec]

def setup (env : Environment) : Except HostFailure Unit := do
  primitive env .signature
  primitive env .compilation
  primitive env .launch

def SetupReady (env : Environment) : Prop :=
  env .signature=none ∧ env .compilation=none ∧ env .launch=none

def SetupFailure (env : Environment) (why : HostFailure) : Prop :=
  PrimitiveFailure env .signature why ∨
  (env .signature=none ∧
    (PrimitiveFailure env .compilation why ∨
     (env .compilation=none ∧ PrimitiveFailure env .launch why)))

theorem setup_all_outcomes (env : Environment) :
    match setup env with
    | .error why => SetupFailure env why
    | .ok () => SetupReady env := by
  cases hs : env .signature with
  | some detail => simp [setup,primitive,hs,SetupFailure,PrimitiveFailure]
  | none =>
    cases hc : env .compilation with
    | some detail => simp [setup,primitive,hs,hc,SetupFailure,PrimitiveFailure]
    | none => cases hl : env .launch <;>
        simp [setup,primitive,hs,hc,hl,SetupFailure,PrimitiveFailure,SetupReady]

/-- Payload kinds are mathematical model roles, not new Python dtype names. -/
inductive PayloadKind where | mathematicalReal | int8
  deriving DecidableEq, Repr
structure Tensor where
  region : String
  shape : List Nat
  strides : List Nat
  payload : PayloadKind
  isCuda : Bool
  deviceIndex : Nat
  storageOffset : Nat
  storageLength : Nat
  deriving DecidableEq, Repr

def outputTensor (a : Input) (alloc : Allocations) : Tensor :=
  ⟨alloc.output,a.shape,QuantizeGlobalFinite.contiguousStrides a.shape,.int8,
    true,alloc.currentDevice,0,a.numel⟩
def maximumTensor (a : Input) (alloc : Allocations) : Tensor :=
  ⟨alloc.maximum,[1],[1],.mathematicalReal,a.isCuda,a.deviceIndex,0,1⟩
def scalarMaximumTensor (a : Input) (alloc : Allocations) : Tensor :=
  ⟨alloc.maximum,[],[],.mathematicalReal,a.isCuda,a.deviceIndex,0,1⟩
def unsqueezeZero (t : Tensor) : Tensor := {t with shape:=[1],strides:=[1]}

theorem maximum_unsqueeze_alias (a : Input) (alloc : Allocations) :
    maximumTensor a alloc=unsqueezeZero (scalarMaximumTensor a alloc) := rfl
theorem output_numel (a : Input) (alloc : Allocations) :
    (outputTensor a alloc).shape.prod=a.numel := rfl

inductive Failure where
  | preparation (why : HostFailure)
  | setup (why : HostFailure)
inductive Outcome where
  | failed (why : Failure)
  | observed (issue : Issue)
  | returned (tensors : Tensor × Tensor) (memory : Memory)

def executeHost (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (trials : List Config) (last : Config) : Outcome :=
  match prepare env a alloc m x with
  | .error why => .failed (.preparation why)
  | .ok p => match setup env with
    | .error why => .failed (.setup why)
    | .ok () => match executeTrials a alloc x (trials++[last]) p.initial with
      | .error issue => .observed issue
      | .ok final => .returned (outputTensor a alloc,maximumTensor a alloc) final

/-- Ordinary logical-view metadata and fresh named allocations only. There is
no contiguous-input, positive-maximum, access-safety or defined-result premise. -/
def Metadata (a : Input) : Prop :=
  a.logicalOffsets=QuantizeGlobalFinite.viewOffsets a.shape a.strides ∧
  a.logicalOffsets.length=a.numel

structure ReturnedSpec (env : Environment) (a : Input) (alloc : Allocations)
    (m : Memory) (x : String) (trials : List Config) (last : Config)
    (tensors : Tensor × Tensor) (final : Memory) : Prop where
  returned : tensors=(outputTensor a alloc,maximumTensor a alloc)
  prepared : ∃ p, PreparedSpec env a alloc m x p ∧ SetupReady env ∧
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
  | .failed (.setup why) => ∃ p, PreparedSpec env a alloc m x p ∧ SetupFailure env why
  | .observed issue => ∃ p, PreparedSpec env a alloc m x p ∧ SetupReady env ∧
      trialsPlan a alloc x p.initial (trials++[last])=.error issue
  | .returned tensors final => ReturnedSpec env a alloc m x trials last tensors final

/-- Every externally visible region outside newly produced tensors retains
its initial mathematical cell. This is not a byte encoding for Real values. -/
theorem allocation_exterior_frame (a : Input) (alloc : Allocations) (m : Memory)
    (maximum : ℝ) (inverse : Option ℝ) (r : String) (i : Nat)
    (hm : r≠alloc.maximum) (hs : r≠alloc.scale) (ho : r≠alloc.output) :
    allocated a alloc m maximum inverse r i=m r i := by
  simp [allocated,hm,hs,ho]

#specNonCircular maximumOf avoiding [prepare,executeHost]
#specNonCircular PreparationFailure avoiding [prepare,executeHost]
#specNonCircular PreparedSpec avoiding [prepare,executeHost]
#specNonCircular ReturnedSpec avoiding [prepare,executeHost,executeTrials,executeLaunch,executeGrid,sourceProgramPlan,applyWrites]
#specNonCircular HostSpec avoiding [prepare,executeHost,executeTrials,executeLaunch,executeGrid,sourceProgramPlan,applyWrites]
#axiomsClean prepare_all_outcomes
#axiomsClean setup_all_outcomes
#axiomsClean maximum_unsqueeze_alias
#axiomsClean output_numel
#axiomsClean allocation_exterior_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end QuantizeGlobalReal
