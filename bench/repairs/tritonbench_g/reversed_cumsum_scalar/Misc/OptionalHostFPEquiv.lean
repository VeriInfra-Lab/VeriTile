import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.Contract
import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPWrapper

/-! Complete source-precision wrapper result. This statement retains the actual
FP32 reduction, forward scan, descending carry, separate sub/add and store
conversion. It assumes no cancellation or reassociation law. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.OptionalHostEquivalence
open VeriTile.Triton VeriTile.Torch FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

def defaultPrecision : ComputeDType := .fp32
def inputPrecision : FloatDType := .fp32
def outputPrecision : FloatDType := .fp32

def reference (M : Config → Algebra α) (scan : Config → FPValues.ScanBackend α)
    (h : Call) (selected : Config) (positive : 0<selected.BT)
    (logical : Nat → Nat → α) (row time : Nat) : Cell α :=
  FPMemory.stored (M selected) h
    (FPExecution.expected (M selected) (scan selected) h selected positive (logical row) time)

/-- Optional native/IEEE model interfaces. They are neither assumptions of the
public kernel rewrite nor deferred numerical experiments for this task. -/
inductive NumericalSite where
  | zeroLiteral | loadConversion | reduction | carryAdd | forwardInclusiveScan
  | subtraction | outputAdd | storeConversion
  deriving DecidableEq

def numericalSites : List NumericalSite := [.zeroLiteral,.loadConversion,.reduction,
  .carryAdd,.forwardInclusiveScan,.subtraction,.outputAdd,.storeConversion]

theorem wrapper_fp32 (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) (M : Config → Algebra α)
    (scan : Config → FPValues.ScanBackend α) (api : Host.API (Cell α))
    (tried : List Config) (valid : ∀ c∈tried, c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : FPStorage.Heap α) (logical : Nat → Nat → α) (live : Nat → Prop)
    (laws : Host.PreparationLaws api h p
      (fun cell => toFloat h.inputDType (cell.read h.inputDType.toTileDType)) logical caller live)
    (a : Host.Allocated (Cell α)) (allocated : api.emptyLike caller=.ok a)
    (device : a.device=s.device) (launchDevice : api.launchDevice=s.device)
    (setups : h.B*h.H*h.T≠0 → Host.Setups api 0 (tried++[selected])) :
    ∃ result, Host.guarded s dtype api (FPWrapper.grid M scan h p) h tried selected caller=.ok result ∧
      result.identity=a.identity ∧ result.device=s.device ∧ Host.metadata h p result ∧
      (∀ row : FPStream.Row h, ∀ time, time<h.T →
        result.heap (Storage.address p.backing.output (row.val*h.T+time))=
          reference M scan h selected (Kernels.configurations_positive selected chosen).1 logical row.val time) ∧
      FPWrapper.Good h p a.heap result.heap ∧
      (∀ addr, live addr → result.heap addr=caller addr) ∧
      (∀ addr, ¬Host.preparationWrites h p addr → result.heap addr=caller addr) ∧
      api.launchDevice=result.device := by
  have layout := binding.outputLayout
  have prepared : Host.prepare api Bool.true caller=.ok a := by simp [Host.prepare,allocated]
  obtain ⟨result,returned⟩ : ∃ result,
      Host.run api (FPWrapper.grid M scan h p) h Bool.true tried selected caller=.ok result := by
    by_cases empty : h.B*h.H*h.T=0
    · exact ⟨a,Host.empty_return api _ h Bool.true tried selected caller a prepared empty⟩
    · obtain ⟨final,raw,_⟩ := FPWrapper.tuned_complete M scan h layout p tried valid selected chosen a.heap
      exact ⟨{a with heap:=final},Host.successful_return api _ h Bool.true tried selected caller
        a prepared empty final raw (setups empty)⟩
  obtain ⟨identity,metadata,output,good,liveFrame,frame⟩ :=
    FPWrapper.returned_logical M scan h layout p api Bool.true tried valid selected chosen
      caller logical live laws a prepared result returned
  have dev := (Host.returned_device api _ h Bool.true tried selected caller a result prepared returned).trans device
  refine ⟨result,(Host.guarded_eq s dtype contract api _ h tried selected caller).trans returned,
    identity,dev,metadata,?_,good,liveFrame,frame,launchDevice.trans dev.symm⟩
  intro row time ht
  rw [←binding.outputFlat row.val time row.isLt ht]
  exact output row time ht

theorem native_programs (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p) (nonempty : 0<h.B*h.H*h.T)
    (c : Config) (chosen : c∈configurations) (row : Fin (h.B*h.H)) :
    NativeIndex.ProgramCertificate h c p.backing (binding.pid s dtype contract h p nonempty row) :=
  NativeIndex.certify h c chosen p.backing _

#axiomsClean wrapper_fp32
#axiomsClean native_programs
#print axioms wrapper_fp32
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.OptionalHostEquivalence
