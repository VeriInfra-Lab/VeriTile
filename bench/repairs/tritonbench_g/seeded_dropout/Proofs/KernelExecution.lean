import bench.repairs.tritonbench_g.seeded_dropout.Proofs.ActualTile
import bench.repairs.tritonbench_g.seeded_dropout.Proofs.LegacyKernelExecution
import bench.tritonbench_g.seeded_dropout.Proofs.FP
import bench.tritonbench_g.seeded_dropout.Proofs.Safety

namespace VeriTile.Bench.Repairs.SeededDropout.KernelExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.SeededDropout
noncomputable section

def layout (X O : RegionName) (n : Nat) : ActualTile.Layout :=
  ⟨X,O,n,1024,SourceDomain.wrapper_block_legal⟩

theorem source_eq (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    kernel k X O n p seed = (layout X O n).source k p seed := by
  rw [ParametricExecutionModel.actual]
  rfl

theorem seed_valid (k : SeedKind) (seed : SeedCarrier k) (h : SeedABI k seed) :
    (legacyKind k).inRange (TypedSource.seedInt (legacyKind k) (legacySeed k seed)) = Bool.true :=
  Whole.PythonSeedABI.valid _ _ (legacy_seed_abi k seed h)

/-- Actual one-dimensional program entry. Registers, memory and unspecified
masked-load values remain arbitrary. Other program-ID axes are zero. -/
abbrev entry {α : Type} [Inhabited α] := @ActualTile.tileState α _

def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed) :
    {t : State α // FP.Structural.exec M (kernel k X O n p seed) (entry pid s) = some t} := by
  let result := ActualTile.sourceRun M lt comparison (layout X O n) k p seed
    pid s hn hp (by simpa only [legacy_seed_value] using seed_valid k seed hs)
  refine ⟨result.val, ?_⟩
  rw [source_eq]
  exact result.property

theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) (s : State α) :
    FP.Structural.exec M (kernel k X O n p seed) s =
      FP.Structural.exec M (indexCommutedKernel k X O n p seed) s := by
  change run M (kernel k X O n p seed).surfaceBody s =
    run M (indexCommutedKernel k X O n p seed).surfaceBody s
  have preRun : run M ((kernel k X O n p seed).surfaceBody.take 2) s =
      run M ((indexCommutedKernel k X O n p seed).surfaceBody.take 2) s := by
    cases k <;> simp [kernel,indexCommutedKernel,sourceI32,sourceI64,sourceU64,
      candidateI32,candidateI64,candidateU64,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,Nat.mul_comm]
    all_goals congr 1; funext i; exact Nat.mul_comm _ _
  rw [← List.take_append_drop 2 (kernel k X O n p seed).surfaceBody,
    ← List.take_append_drop 2 (indexCommutedKernel k X O n p seed).surfaceBody,
    run_append,run_append,preRun]
  cases k <;> rfl

theorem values {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed)
    (lane : Fin 1024) (active : pid*1024+lane.val < n) :
    (originalExecution M lt comparison k X O n p seed pid s hn hp hs).val.mem O (pid*1024+lane.val) =
      Cell.mk .real (ActualTile.scalar M lt p (seedValue k seed) (pid*1024+lane.val)
        ((s.mem X (pid*1024+lane.val)).read .real)) := by
  simpa only [originalExecution,legacy_seed_value] using
    ActualTile.tile_readback M lt comparison (layout X O n) k p seed
      pid s hn hp (by simpa only [legacy_seed_value] using seed_valid k seed hs) (lane,()) active

def Outside (O : RegionName) (n pid : Nat) (r : RegionName) (a : Nat) : Prop :=
  r ≠ O ∨ ∀ lane : Fin 1024, pid*1024+lane.val < n → a ≠ pid*1024+lane.val

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed)
    (r : RegionName) (a : Nat) (outside : Outside O n pid r a) :
    (originalExecution M lt comparison k X O n p seed pid s hn hp hs).val.mem r a = s.mem r a := by
  apply ActualTile.tile_frame M lt comparison (layout X O n) k p seed
    pid s hn hp (by simpa only [legacy_seed_value] using seed_valid k seed hs) r a
  rcases outside with h | h
  · exact Or.inl h
  · exact Or.inr (fun j hj => h j.1 hj)

#axiomsClean originalExecution
#axiomsClean execution_eq
#axiomsClean values
#axiomsClean frame
end
end VeriTile.Bench.Repairs.SeededDropout.KernelExecution
