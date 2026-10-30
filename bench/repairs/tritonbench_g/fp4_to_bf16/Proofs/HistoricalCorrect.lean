import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.CheckedHost
import bench.repairs.tritonbench_g.fp4_to_bf16.Kernels
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Host
import bench.tritonbench_g.fp4_to_bf16.Proofs.FPReal

/-! Independent mathematical target: signed E2M1 with bias 1, magnitude
m/2 for exponent 0, otherwise (1+m/2)*2^(e-1). The complete returned tensor
has doubled last dimension and BF16 storage. Exact bits additionally retain
both zero signs. All counts follow ordinary allocation/launch outcomes;
there is no inserted user n bound or deferred numerical atom. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.HistoricalCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostSchedule RealSpec
open Bytes Connection
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

specification source_wrapper_real_correctness
    (rest : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (heap : Heap) (initial : State Word) (identity device : Nat) (jobs : List Nat)
    (fresh : h.input≠h.output) (rep : StorageRep h s heap initial)
    (order : jobs.Perm (List.range (grid h)))
    (allocationReady contiguous inputCuda outputCuda setupReady : Bool) :
    physical h s heap initial.mem=heap ∧
    ∃ outcome, observe rest h s heap initial identity device jobs
        allocationReady contiguous inputCuda outputCuda setupReady = some outcome ∧
      match outcome with
      | .error e => protocol (some h.shape) allocationReady contiguous inputCuda outputCuda setupReady = .error e
      | .ok observed => ResultSpec h s heap initial identity device observed ∧
          ∀ j, j<2*h.shape.count →
            finiteReal ⟨read observed.bytes (s.outputBase+2*j)⟩ =
              some (e2m1 (targetNibble s heap j)) := by
  refine ⟨physical_initial h s heap initial rep, ?_⟩
  obtain ⟨outcome,run,spec⟩ := observe_correct rest h s heap initial identity device jobs
    fresh rep.input order allocationReady contiguous inputCuda outputCuda setupReady
  refine ⟨outcome,run,?_⟩
  cases outcome with
  | error e => exact spec
  | ok observed =>
    refine ⟨spec,?_⟩
    intro j hj
    have bits := spec.2.2.2.2.2.2.1 j hj
    rw [bits]
    exact expected_real _

/-- All padded finite predicates agree with the reused complete source, without
assuming active lanes. Address safety and ownership are derived from ordinary
successful allocation metadata, not hidden in a numerical premise. -/
specification native_lane_refinement (h : Input) (s : Finite.Storage h.shape.count)
    (pid li lo : Nat) (hp : pid<grid h) (hi : li<512) (ho : lo<1024) :
    Kernels.inputMask h.shape.count pid li=decide (pid*512+li<h.shape.count) ∧
    Kernels.outputMask h.shape.count pid lo=decide (pid*1024+lo<2*h.shape.count) :=
  Finite.masks_exact _ _ _ _ (Finite.storage_count_bound s) hp hi ho

#axiomsClean source_wrapper_real_correctness
#axiomsClean native_lane_refinement

/-- The current checked revision. Every selected input restriction is obtained
from the executable guard; raw pointer/shape identities tie the public Tensor
to the complete source and byte-memory proof. Runtime failures remain errors. -/
specification checked_source_wrapper_real_correctness
    (rest : Algebra Word) (x : VeriTile.Torch.TensorMetadata)
    (checked : Checked.checkInputs (.tensor x) = .ok ())
    (h : Input) (s : Finite.Storage h.shape.count) (identity device : Nat)
    (binding : Checked.MetadataBinding x h s device)
    (heap : Heap) (logical : Nat → BitVec 8) (storage : Checked.LogicalStorage x heap logical)
    (initial : State Word) (jobs : List Nat)
    (fresh : h.input≠h.output) (rep : StorageRep h s heap initial)
    (order : jobs.Perm (List.range (grid h)))
    (allocationReady outputCuda setupReady : Bool) :
    Checked.InputContract x ∧ x.numel = h.shape.count ∧
    x.dataPtr.toNat = s.inputPointer ∧ physical h s heap initial.mem = heap ∧
    ∃ outcome, Checked.runChecked rest (.tensor x) h s heap initial identity device jobs
        allocationReady outputCuda setupReady = some outcome ∧
      match outcome with
      | .error (.input _) => False
      | .error (.runtime e) =>
          protocol (some h.shape) allocationReady Bool.true Bool.true outputCuda setupReady = .error e
      | .ok observed => ResultSpec h s heap initial identity device observed ∧
          ∀ j, j<2*x.numel → finiteReal ⟨read observed.bytes (s.outputBase+2*j)⟩ =
            some (e2m1 (Checked.logicalNibble logical j)) := by
  have count := Checked.numel_binding x h s _ binding
  refine ⟨(Checked.checkInputs_iff (.tensor x)).mp checked,count,binding.pointer,?_,?_⟩
  · exact physical_initial h s heap initial rep
  obtain ⟨_,outcome,run,spec⟩ := source_wrapper_real_correctness rest h s heap initial
    identity device jobs fresh rep order allocationReady Bool.true Bool.true outputCuda setupReady
  refine ⟨outcome.mapError Checked.CheckedFailure.runtime,?_,?_⟩
  · rw [Checked.checked_source rest x checked,run]
    rfl
  · cases outcome with
    | error e => exact spec
    | ok observed =>
      refine ⟨spec.1,?_⟩
      intro j hj
      have raw := spec.2 j (by simpa only [←count] using hj)
      rw [Checked.logical_nibble x h s _ binding heap logical storage checked j hj] at raw
      exact raw

#axiomsClean checked_source_wrapper_real_correctness
end
end VeriTile.Bench.Repairs.Fp4ToBf16.HistoricalCorrect
