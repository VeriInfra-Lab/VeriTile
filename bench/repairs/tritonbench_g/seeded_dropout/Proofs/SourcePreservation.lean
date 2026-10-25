import bench.repairs.tritonbench_g.seeded_dropout.Proofs.KernelExecution
import bench.repairs.tritonbench_g.seeded_dropout.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SeededDropout.SourcePreservation
/-- Complete original source equality for all seed carriers and parameters. -/
theorem legacy_kernel_ir : LegacyPastedSurface.kernel = LegacySurface.kernel := by
  funext kind
  cases kind <;> rfl

/-- Complete candidate source equality for all seed carriers and parameters. -/
theorem legacy_candidate_ir : LegacyPastedSurface.indexCommutedKernel = LegacySurface.indexCommutedKernel := by
  funext kind
  cases kind <;> rfl

#axiomsClean legacy_kernel_ir
#axiomsClean legacy_candidate_ir
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SeededDropout
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Defined interpretation of the historical proof model: the call supplies
FP32 probability, and the old extra real-to-real store self-cast is identity.
This changes that historical backend explicitly; no law is assumed for M.cast.
The actual FP program pair is checked independently under arbitrary M. -/
def legacyBackend {A : Type} (M : Algebra A) : Algebra A :=
  { M with
    literal := fun precision dtype value => M.literal (precision.or (some .fp32)) dtype value
    cast := fun precision src dst value =>
      if src = .real ∧ dst = .real then value else M.cast precision src dst value }

theorem scalar_connection {A : Type} (M : Algebra A) (lt : A → A → Bool)
    (p : ℝ) (seed : Int) (i : Nat) (x : A) :
    Whole.scalar (legacyBackend M) lt p seed i x = ActualTile.scalar M lt p seed i x := by
  simp only [Whole.scalar,ActualTile.scalar,legacyBackend,Option.or_some,Option.getD_none,Option.getD_some,ite_true,and_self]

/-- Both complete programs succeed under the unchanged seed/shape/launch
contract, and every memory cell agrees. Their extra historical registers are
not identified with actual Python registers. -/
theorem original_memory {A : Type} [Inhabited A] (M : Algebra A) (lt : A → A → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State A) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed) :
    (KernelExecution.originalExecution M lt comparison k X O n p seed pid s hn hp hs).val.mem =
      (LegacyKernelExecution.originalExecution (legacyBackend M) lt comparison
        k X O n p seed pid s hn hp hs).val.mem := by
  change (ActualTile.sourceRun M lt comparison (KernelExecution.layout X O n) k p seed
    pid s hn hp _).val.mem =
    (Whole.sourceRun (legacyBackend M) lt comparison (LegacyKernelExecution.layout X O n)
      (legacyKind k) p (legacySeed k seed) pid s hn hp _).val.mem
  rw [ActualTile.sourceRun_mem,Whole.sourceRun_mem]
  simp only [ActualTile.tileMemory,Whole.tileMemory,scalar_connection,legacy_seed_value,
    ActualTile.raw,Whole.raw,legacyBackend]
  rfl

/-- Complete original memory observation, including success, connected at
exactly the original host's admitted seed/size/grid domain. -/
theorem kernel {A : Type} [Inhabited A] (M : Algebra A) (lt : A → A → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State A) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed) :
    (FP.Structural.exec M (SeededDropout.kernel k X O n p seed) (KernelExecution.entry pid s)).map State.mem =
      (FP.Structural.exec (legacyBackend M) (LegacyPastedSurface.kernel k X O n p seed)
        (KernelExecution.entry pid s)).map State.mem := by
  have legacy_run : FP.Structural.exec (legacyBackend M) (LegacyPastedSurface.kernel k X O n p seed)
      (KernelExecution.entry pid s) = some
        (LegacyKernelExecution.originalExecution (legacyBackend M) lt comparison k X O n p seed pid s hn hp hs).val :=
    (LegacyKernelExecution.originalExecution (legacyBackend M) lt comparison k X O n p seed pid s hn hp hs).property
  rw [(KernelExecution.originalExecution M lt comparison k X O n p seed pid s hn hp hs).property,legacy_run]
  exact congrArg some (original_memory M lt comparison k X O n p seed pid s hn hp hs)

theorem indexCommutedKernel {A : Type} [Inhabited A] (M : Algebra A) (lt : A → A → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State A) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed) :
    (FP.Structural.exec M (SeededDropout.indexCommutedKernel k X O n p seed) (KernelExecution.entry pid s)).map State.mem =
      (FP.Structural.exec (legacyBackend M) (LegacyPastedSurface.indexCommutedKernel k X O n p seed)
        (KernelExecution.entry pid s)).map State.mem := by
  rw [← KernelExecution.execution_eq,← LegacyKernelExecution.execution_eq]
  exact kernel M lt comparison k X O n p seed pid s hn hp hs

#axiomsClean scalar_connection
#axiomsClean original_memory
#axiomsClean kernel
#axiomsClean indexCommutedKernel
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SeededDropout.SourcePreservation
