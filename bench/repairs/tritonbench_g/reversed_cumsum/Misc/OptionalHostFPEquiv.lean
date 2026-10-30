import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.reversed_cumsum.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Contract
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.NativeWrapper

/-! Exact native graph over original FP32 words, including accumulating dot.
The optional backend interfaces below are not public transformation assumptions
or numerical experiment TODOs. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.OptionalHostEquivalence
open VeriTile.Triton VeriTile.Torch FP.Structural NativeGraph
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

structure PrecisionProfile where
  input : FloatDType
  output : FloatDType
  accumulator : ComputeDType
  inputPrecision : String
  featureBlock : Nat
  configurations : List Config

def profile : PrecisionProfile := ⟨.fp32,.fp32,.fp32,"ieee",32,ReversedCumsum.configurations⟩

/-- Local reached-operand obligations; none is a whole-execution, output,
index, storage or protocol premise of the checked theorem. -/
inductive NumericalSite where
  | fp32ZeroOneLiterals | fp32LoadConversion | fp32PaddingConversion
  | fp32IEEEAccumulatingDot | fp32ConfiguredReduction | fp32CarryAdd | fp32StoreConversion
  deriving DecidableEq

def numericalSites : List NumericalSite := [.fp32ZeroOneLiterals,.fp32LoadConversion,.fp32PaddingConversion,
  .fp32IEEEAccumulatingDot,.fp32ConfiguredReduction,.fp32CarryAdd,.fp32StoreConversion]

def reference := NativeWrapper.reference

def Outcome (s : TensorMetadata) (dtype : Option DType) (rest : Algebra Expr)
    (h : Call) (p : Storage.Placement h) (api : Host.API (Cell Expr))
    (tried : List Config) (selected : Config) (positive : 0<selected.BT)
    (caller : FPStorage.Heap Expr) (words : Nat → Nat → Nat → BitVec 32) (live : Nat → Prop) : Prop :=
  match Host.guarded s dtype api (FPWrapper.grids (NativeWrapper.models rest) NativeGraph.backend h p) h tried selected caller with
  | .error e => e.stage≠.execution ∧ HostOutcome.Good h p live caller e.heap
  | .ok result => ∃ a,Host.prepare api Bool.true caller=.ok a ∧
      result.identity=a.identity ∧ result.device=s.device ∧ api.launchDevice=result.device ∧
      Host.metadata h p result ∧ HostOutcome.Good h p live caller result.heap ∧
      (∀ row : Geometry.Row h,∀ t feature,t<h.T → feature<h.S →
        result.heap (Storage.address p.backing.output (Geometry.address h row.val t feature))=
          reference rest h selected positive words row.val t feature) ∧
      (∀ addr,¬Storage.writes h p (Storage.programs h) addr → result.heap addr=a.heap addr) ∧
      (∀ row : Geometry.Row h,∀ t feature,t<h.T → feature<h.S → Materialized
        ((result.heap (Storage.address p.backing.output (Geometry.address h row.val t feature))).read .fp32))

specification reversed_cumsum_repaired_fp (rest : Algebra Expr) (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p)
    (api : Host.API (Cell Expr)) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : FPStorage.Heap Expr) (words : Nat → Nat → Nat → BitVec 32) (live : Nat → Prop)
    (laws : Host.PreparationLaws api h p (FPWrapper.readValue h) (NativeWrapper.logical words) caller live)
    (device : ∀ a, api.emptyLike caller=.ok a → a.device=s.device)
    (launchDevice : api.launchDevice=s.device) :
    (∀ nonempty : 0<h.B*h.H*(h.T*h.S),∀ c∈configurations,∀ program : Geometry.Program h,
      NativeIndex.ProgramCertificate h c p.backing (binding.pid s dtype contract h p nonempty program)) ∧
    Outcome s dtype rest h p api tried selected (Kernels.configurations_positive selected chosen).1 caller words live := by
  refine ⟨fun nonempty c hc program=>NativeIndex.certify h c hc p.backing _,?_⟩
  have general := FPWrapper.outcome (NativeWrapper.models rest) NativeGraph.backend api h binding.distinct p
    (NativeWrapper.logical words) caller live laws Bool.true tried valid selected chosen
  unfold Outcome
  rw [Host.guarded_eq s dtype contract]
  cases prepared : Host.prepare api Bool.true caller with
  | error e =>
    have ret : Host.run api (FPWrapper.grids (NativeWrapper.models rest) NativeGraph.backend h p) h Bool.true tried selected caller=.error e := by simp [Host.run,prepared]
    simpa only [ret,HostOutcome.Result] using general
  | ok a =>
    cases ret : Host.run api (FPWrapper.grids (NativeWrapper.models rest) NativeGraph.backend h p) h Bool.true tried selected caller with
    | error e => simpa only [ret,HostOutcome.Result] using general
    | ok result =>
      obtain ⟨identity,metadataOk,good,outputs,frame⟩ := FPWrapper.returned (NativeWrapper.models rest) NativeGraph.backend api
        h binding.distinct p (NativeWrapper.logical words) caller live laws Bool.true tried valid selected chosen a result prepared ret
      have allocated : api.emptyLike caller=.ok a := by
        cases eq : api.emptyLike caller <;> simp_all [Host.prepare,ReversedCumsumScalar.Host.prepare]
      have dev := (Host.returned_device api _ h Bool.true tried selected caller a result prepared ret).trans (device a allocated)
      refine ⟨a,rfl,identity,dev,launchDevice.trans dev.symm,metadataOk,good,outputs,frame,?_⟩
      intro row t feature ht hs
      rw [outputs row t feature ht hs]
      exact NativeWrapper.reference_materialized rest h binding.outputDType selected _ words row.val t feature

#axiomsClean reversed_cumsum_repaired_fp
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.OptionalHostEquivalence
