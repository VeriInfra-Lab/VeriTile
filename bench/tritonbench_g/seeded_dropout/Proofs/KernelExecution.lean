import bench.tritonbench_g.seeded_dropout.Proofs.Projection
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_FP
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Safety

namespace VeriTile.Bench.Repairs.SeededDropout.KernelExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.SeededDropout
noncomputable section

def layout (X O : RegionName) (n : Nat) : Whole.Layout :=
  ⟨X,O,n,1024,SourceDomain.wrapper_block_legal⟩

theorem source_eq (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    kernel k X O n p seed = (layout X O n).source (legacyKind k) p (legacySeed k seed) := by
  cases k <;> rfl

theorem candidate_eq (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) :
    indexCommutedKernel k X O n p seed =
      TypedSource.source .fp32 (legacyKind k) Bool.true X O n p (legacySeed k seed)
        1024 SourceDomain.wrapper_block_legal := by
  cases k <;> rfl

theorem seed_valid (k : SeedKind) (seed : SeedCarrier k) (h : SeedABI k seed) :
    (legacyKind k).inRange (TypedSource.seedInt (legacyKind k) (legacySeed k seed)) = Bool.true :=
  Whole.PythonSeedABI.valid _ _ (legacy_seed_abi k seed h)

/-- Actual one-dimensional program entry. Registers, memory and unspecified
masked-load values remain arbitrary. Other program-ID axes are zero. -/
abbrev entry {α : Type} [Inhabited α] := @Whole.tileState α _

def originalExecution {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed) :
    {t : State α // FP.Structural.exec M (kernel k X O n p seed) (entry pid s) = some t} := by
  let result := Whole.sourceRun M lt comparison (layout X O n) (legacyKind k) p (legacySeed k seed)
    pid s hn hp (seed_valid k seed hs)
  refine ⟨result.val, ?_⟩
  rw [source_eq]
  exact result.property

theorem execution_eq {α : Type} [Inhabited α] (M : Algebra α)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k) (s : State α) :
    FP.Structural.exec M (kernel k X O n p seed) s =
      FP.Structural.exec M (indexCommutedKernel k X O n p seed) s := by
  rw [source_identity,candidate_eq]
  exact FPProof.whole_state .fp32 (legacyKind k) X O n p (legacySeed k seed)
    1024 SourceDomain.wrapper_block_legal M s

theorem values {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed)
    (lane : Fin 1024) (active : pid*1024+lane.val < n) :
    (originalExecution M lt comparison k X O n p seed pid s hn hp hs).val.mem O (pid*1024+lane.val) =
      Cell.mk .real (Whole.scalar M lt p (seedValue k seed) (pid*1024+lane.val)
        ((s.mem X (pid*1024+lane.val)).read .real)) := by
  simpa only [originalExecution,legacy_seed_value] using
    Whole.tile_readback M lt comparison (layout X O n) (legacyKind k) p (legacySeed k seed)
      pid s hn hp (seed_valid k seed hs) (lane,()) active

def Outside (O : RegionName) (n pid : Nat) (r : RegionName) (a : Nat) : Prop :=
  r ≠ O ∨ ∀ lane : Fin 1024, pid*1024+lane.val < n → a ≠ pid*1024+lane.val

theorem frame {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (s : State α) (hn : n < 2^31) (hp : pid < gridCount n) (hs : SeedABI k seed)
    (r : RegionName) (a : Nat) (outside : Outside O n pid r a) :
    (originalExecution M lt comparison k X O n p seed pid s hn hp hs).val.mem r a = s.mem r a := by
  apply Whole.tile_frame M lt comparison (layout X O n) (legacyKind k) p (legacySeed k seed)
    pid s hn hp (seed_valid k seed hs) r a
  rcases outside with h | h
  · exact Or.inl h
  · exact Or.inr (fun j hj => h j.1 hj)

#axiomsClean originalExecution
#axiomsClean execution_eq
#axiomsClean values
#axiomsClean frame
end
end VeriTile.Bench.Repairs.SeededDropout.KernelExecution
