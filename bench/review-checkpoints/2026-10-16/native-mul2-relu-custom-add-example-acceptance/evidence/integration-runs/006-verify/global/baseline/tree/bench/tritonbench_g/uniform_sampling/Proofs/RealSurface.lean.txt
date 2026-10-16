import bench.tritonbench_g.uniform_sampling.Proofs.RealIO
import VeriTile.Triton.Correctness

/-! Standard region correctness for the complete checked source, with four
streams, exact cell tags and a frame at every address outside the write map.
The separate RealIO theorem supplies successful finite flat-memory execution.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.RealSurface
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000

def writes (y : RegionName) (N B pid : Nat) : ComputeCorrect.WriteMap (Fin 4 × Fin B) :=
  ComputeCorrect.WriteMap.writeIf (fun i => SourceDomain.outputIndex pid B i.1 i.2 < N)
    (fun i => (y, SourceDomain.outputIndex pid B i.1 i.2))

def expected (seed offset : Int) (lower upper : ℝ) (B : Nat) (u : BlockState) :
    Fin 4 × Fin B → MemCell :=
  fun i => .real (Formula.expected B seed offset lower upper
    (SourceDomain.outputIndex u.pid B i.1 i.2))

theorem outside_iff (y : RegionName) (N B pid : Nat) (addr : MemCellAddr) :
    (∀ i, writes y N B pid i ≠ some addr) ↔
      ∀ f : Fin 4, ∀ j : Fin B, SourceDomain.outputIndex pid B f j < N →
        ¬ (y = addr.1 ∧ SourceDomain.outputIndex pid B f j = addr.2) := by
  constructor
  · intro h f j hj ha
    apply h (f, j)
    have he : (y, SourceDomain.outputIndex pid B f j) = addr := Prod.ext ha.1 ha.2
    simp only [writes, ComputeCorrect.WriteMap.writeIf, hj, ite_true, he]
  · intro h i he
    by_cases hj : SourceDomain.outputIndex pid B i.1 i.2 < N
    · have ha : (y, SourceDomain.outputIndex pid B i.1 i.2) = addr := by
        simpa only [writes, ComputeCorrect.WriteMap.writeIf, hj, ite_true,
          Option.some.injEq] using he
      exact h i.1 i.2 hj ⟨congrArg Prod.fst ha, congrArg Prod.snd ha⟩
    · simp only [writes, ComputeCorrect.WriteMap.writeIf, hj, ite_false] at he
      cases he

theorem projection (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    (source storage swapped y N seed offset lower upper B hB).eraseDType.toAlgorithm? =
      .ok (Algorithm.kernel swapped y N seed offset lower upper B) := by
  rw [Erasure.complete_projection, Algorithm.source_eq]

theorem realizes_frame (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (_hN : N < 2^31) (_hpid : u.pid < SourceDomain.grid N B)
    (hseed : Random.IntegerType.i64.inRange seed = Bool.true)
    (_hoffset : Random.IntegerType.i64.inRange offset = Bool.true) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
      (source storage swapped y N seed offset lower upper B hB).eraseDType u
      (writes y N B u.pid) (expected seed offset lower upper B u) := by
  have he := Execution.exec_storeBlock swapped y N seed offset lower upper B
    (SourceDomain.legalBlock_fields hB).1 u hseed
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq
      (projection storage swapped y N seed offset lower upper B hB)
    intro s₀ t he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro i hi
    exact Memory.storeBlock_active_cell u.pid (Execution.resultBase seed offset lower upper B u)
      y N B seed offset lower upper i.1 i.2 hi
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq
      (projection storage swapped y N seed offset lower upper B hB)
    intro s₀ t he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro addr hout
    exact (Memory.storeBlock_frame u.pid (Execution.resultBase seed offset lower upper B u)
      y N B seed offset lower upper addr.1 addr.2
      ((outside_iff y N B u.pid addr).mp hout)).trans
      (congrFun (congrFun (Execution.resultBase_frame seed offset lower upper B u).1 addr.1) addr.2)

#axiomsClean outside_iff
#axiomsClean projection
#axiomsClean realizes_frame

end
end VeriTile.Bench.TritonBenchG.UniformSampling.RealSurface
