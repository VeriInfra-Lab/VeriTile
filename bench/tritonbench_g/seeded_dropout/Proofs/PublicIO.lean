import bench.tritonbench_g.seeded_dropout.IO
import bench.tritonbench_g.seeded_dropout.Proofs.SourceSafety
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.SeededDropout.KernelPublic
open VeriTile.Triton FP.Structural KernelExecution
open VeriTile.Bench.TritonBenchG.SeededDropout
noncomputable section

def target {α : Type} (M : Algebra α) (k : SeedKind) (X O : RegionName)
    (n : Nat) (p : ℝ) (seed : SeedCarrier k) (pid : Nat) (f : Nat → α → α)
    (xs : (io M k X O n p seed pid).Inputs) : (io M k X O n p seed pid).Outputs :=
  fun o => Fin.cases (fun j => f (pid*1024+j.1.val) (xs 0 j)) (fun z => Fin.elim0 z) o

theorem implements {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed)
    (f : Nat → α → α)
    (scalar : ∀ i x,Whole.scalar M lt p (seedValue k seed) i x=f i x) :
    (io M k X O n p seed pid).Implements (target M k X O n p seed pid f) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb s _
    rw [show (io M k X O n p seed pid).schedule vals = [_] from rfl,scheduleSafe_singleton]
    apply KernelSafety.trace_safe M lt comparison k X O n p seed pid s hn hp hs
    · exact fun j active => hb.2.1 0 (j,()) active
    · exact fun j active => hb.2.2.1 0 (j,()) active
  · intro vals s input
    let result := originalExecution M lt comparison k X O n p seed pid s hn hp hs
    refine ⟨result.val,?_,?_,?_⟩
    · simpa only [io,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,entry,Whole.tileState] using result.property
    · intro o
      fin_cases o
      intro j active
      have read := input 0 j active
      change (s.mem X (pid*1024+j.1.val)).read .real=vals 0 j at read
      change mapCell id (result.val.mem O (pid*1024+j.1.val)) =
        Cell.mk .real (f (pid*1024+j.1.val) (vals 0 j))
      rw [show result.val.mem O (pid*1024+j.1.val)=_ from
        values M lt comparison k X O n p seed pid s hn hp hs j.1 active]
      change Cell.mk .real (Whole.scalar M lt p (seedValue k seed) (pid*1024+j.1.val) _) = _
      rw [read,scalar]
    · intro r off outside
      apply frame M lt comparison k X O n p seed pid s hn hp hs r off
      by_cases eq : r=O
      · right
        intro j active he
        exact (outside.1 0 (j,()) active).elim (fun hn => hn eq) (fun hn => hn he)
      · exact Or.inl eq

/-- Both displayed sources are safe, terminate, agree at every active cell and
preserve the exact exterior frame. This uses only the integer product identity. -/
theorem equivalent {α : Type} [Inhabited α] (M : Algebra α) (lt : α → α → Bool)
    (comparison : M.compareLt none .real = some lt)
    (k : SeedKind) (X O : RegionName) (n : Nat) (p : ℝ) (seed : SeedCarrier k)
    (pid : Nat) (hn : n<2^31) (hp : pid<gridCount n) (hs : SeedABI k seed) :
    (io M k X O n p seed pid).Equivalent (variant M k X O n p seed pid) := by
  apply ProgramIO.EquivalentOn.intro
  · intro vals _ bounds hb s _
    rw [show (io M k X O n p seed pid).schedule vals = [_] from rfl,scheduleSafe_singleton]
    apply KernelSafety.trace_safe M lt comparison k X O n p seed pid s hn hp hs
    · exact fun j active => hb.2.1 0 (j,()) active
    · exact fun j active => hb.2.2.1 0 (j,()) active
  · intro vals _ bounds hb s _
    rw [show (variant M k X O n p seed pid).schedule vals = [_] from rfl,scheduleSafe_singleton]
    apply KernelSafety.candidate_trace_safe M lt comparison k X O n p seed pid s hn hp hs
    · exact fun j active => hb.2.1 0 (j,()) active
    · exact fun j active => hb.2.2.1 0 (j,()) active
  · intro vals _ bounds hb s _
    let result := originalExecution M lt comparison k X O n p seed pid s hn hp hs
    have original : runSchedule ((io M k X O n p seed pid).schedule vals) s=some result.val := by
      simpa only [io,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,entry,Whole.tileState] using result.property
    have candidate : runSchedule ((variant M k X O n p seed pid).schedule vals) s=some result.val := by
      simpa only [variant,runSchedule_singleton,Invocation.run,Invocation.initial,
        bindRegisters,List.foldl_nil,Option.getD_none,entry,Whole.tileState] using
          (execution_eq M k X O n p seed (entry pid s)).symm.trans result.property
    have framed : (io M k X O n p seed pid).Frame vals s result.val := by
      intro r off outside
      apply frame M lt comparison k X O n p seed pid s hn hp hs r off
      by_cases eq : r=O
      · right
        intro j active he
        exact (outside.1 0 (j,()) active).elim (fun hn => hn eq) (fun hn => hn he)
      · exact Or.inl eq
    exact ⟨result.val,result.val,original,candidate,fun _ _ _ => rfl,framed,framed⟩

#axiomsClean equivalent

#axiomsClean implements
end
end VeriTile.Bench.Repairs.SeededDropout.KernelPublic
