import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.reversed_cumsum.Kernels
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.Contract
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.RealWrapper

/-! Complete guarded wrapper and independent suffix sum over the original input. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.OptionalHostCorrectness
open VeriTile.Triton VeriTile.Torch
noncomputable section

def reference (h : Call) (logical : Nat → Nat → Nat → ℝ) (row t s : Nat) : MemCell :=
  .real (∑ k ∈ (Finset.range h.T).filter (fun k=>t≤k),logical row k s)

theorem reference_sum (h : Call) (logical : Nat → Nat → Nat → ℝ) (row t s : Nat) (ht : t<h.T) :
    RealWrapper.reference h logical row t s=reference h logical row t s := by
  unfold RealWrapper.reference reference
  rw [VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath.suffix_filter _ _ _ (Nat.le_of_lt ht)]

def Outcome (s : TensorMetadata) (dtype : Option DType) (h : Call) (p : Storage.Placement h)
    (api : Host.API MemCell) (tried : List Config) (selected : Config)
    (caller : RealStorage.Heap) (logical : Nat → Nat → Nat → ℝ) (live : Nat → Prop) : Prop :=
  match Host.guarded s dtype api (RealStream.grid h p) h tried selected caller with
  | .error e => e.stage≠.execution ∧ HostOutcome.Good h p live caller e.heap
  | .ok result => ∃ a,Host.prepare api Bool.true caller=.ok a ∧
      result.identity=a.identity ∧ result.device=s.device ∧ api.launchDevice=result.device ∧
      Host.metadata h p result ∧ HostOutcome.Good h p live caller result.heap ∧
      (∀ row : Geometry.Row h,∀ t feature,t<h.T → feature<h.S →
        result.heap (Storage.address p.backing.output (Geometry.address h row.val t feature))=
          reference h logical row.val t feature) ∧
      (∀ addr,¬Storage.writes h p (Storage.programs h) addr → result.heap addr=a.heap addr)

specification reversed_cumsum_repaired_correct (s : TensorMetadata) (dtype : Option DType)
    (contract : InputContract s dtype) (h : Call) (p : Storage.Placement h)
    (binding : TensorBinding s h p)
    (api : Host.API MemCell) (tried : List Config) (valid : ∀ c∈tried,c∈configurations)
    (selected : Config) (chosen : selected∈configurations)
    (caller : RealStorage.Heap) (logical : Nat → Nat → Nat → ℝ) (live : Nat → Prop)
    (laws : Host.PreparationLaws api h p RealStorage.read logical caller live)
    (device : ∀ a, api.emptyLike caller=.ok a → a.device=s.device)
    (launchDevice : api.launchDevice=s.device) :
    (∀ nonempty : 0<h.B*h.H*(h.T*h.S),∀ c∈configurations,∀ program : Geometry.Program h,
      NativeIndex.ProgramCertificate h c p.backing (binding.pid s dtype contract h p nonempty program)) ∧
    Outcome s dtype h p api tried selected caller logical live := by
  refine ⟨fun nonempty c hc program=>NativeIndex.certify h c hc p.backing _,?_⟩
  have general := RealWrapper.outcome api h binding.distinct p logical caller live laws Bool.true tried valid selected chosen
  unfold Outcome
  rw [Host.guarded_eq s dtype contract]
  cases prepared : Host.prepare api Bool.true caller with
  | error e =>
    have ret : Host.run api (RealStream.grid h p) h Bool.true tried selected caller=.error e := by simp [Host.run,prepared]
    simpa only [ret,HostOutcome.Result] using general
  | ok a =>
    cases ret : Host.run api (RealStream.grid h p) h Bool.true tried selected caller with
    | error e => simpa only [ret,HostOutcome.Result] using general
    | ok result =>
      obtain ⟨identity,metadataOk,good,outputs,frame⟩ := RealWrapper.returned api h binding.distinct p
        logical caller live laws Bool.true tried valid selected chosen a result prepared ret
      have allocated : api.emptyLike caller=.ok a := by
        cases eq : api.emptyLike caller <;> simp_all [Host.prepare,ReversedCumsumScalar.Host.prepare]
      have dev := (Host.returned_device api _ h Bool.true tried selected caller a result prepared ret).trans (device a allocated)
      refine ⟨a,rfl,identity,dev,launchDevice.trans dev.symm,metadataOk,good,?_,frame⟩
      intro row t feature ht hs
      exact (outputs row t feature ht hs).trans (reference_sum h logical row.val t feature ht)

#axiomsClean reversed_cumsum_repaired_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.OptionalHostCorrectness
