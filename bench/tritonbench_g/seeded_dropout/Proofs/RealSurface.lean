import bench.tritonbench_g.seeded_dropout.Proofs.RealIO
import VeriTile.Triton.Correctness

/-! Standard region realization for the complete seeded-dropout source.
No existing KernelIO skin packages the runtime seed ABI and storage profiles.
The MemCell surface preserves exact tags and the complete frame. Successful
finite-flat execution and its access-safety premises remain a separate public
conjunct, since standard region realization alone is partial correctness. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.RealSurface
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

def writes (y : RegionName) (N B pid : Nat) : ComputeCorrect.WriteMap (Fin B) :=
  ComputeCorrect.WriteMap.writeIf (fun j => Index.offset pid B j < N)
    (fun j => (y, Index.offset pid B j))

/-- Closed form over the initial input snapshot and original runtime seed/p. -/
def expected (x : RegionName) (p : ℝ) (seed : Int) (B : Nat) (u : BlockState) :
    Fin B → MemCell :=
  fun j => .real (Formula.expected (u.readMem x) p seed (Index.offset u.pid B j))

theorem outside_iff (y : RegionName) (N B pid : Nat) (addr : MemCellAddr) :
    (∀ j, writes y N B pid j ≠ some addr) ↔
      ∀ j : Fin B, Index.offset pid B j < N →
        ¬ (y = addr.1 ∧ Index.offset pid B j = addr.2) := by
  constructor
  · intro h j hj ha
    apply h j
    have he : (y, Index.offset pid B j) = addr := Prod.ext ha.1 ha.2
    simp only [writes, ComputeCorrect.WriteMap.writeIf, hj, ite_true, he]
  · intro h j he
    by_cases hj : Index.offset pid B j < N
    · have ha : (y, Index.offset pid B j) = addr := by
        simpa only [writes, ComputeCorrect.WriteMap.writeIf, hj, ite_true,
          Option.some.injEq] using he
      exact h j hj ⟨congrArg Prod.fst ha, congrArg Prod.snd ha⟩
    · simp only [writes, ComputeCorrect.WriteMap.writeIf, hj, ite_false] at he
      cases he

/-- Successful projection is explicit; default-fallback projection is not used
as evidence for the public compute correctness surface. -/
theorem projection (storage : Storage) (kind : Random.IntegerType) (swapped : Bool)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) :
    (source storage kind swapped x y N p seed B hB).eraseDType.toAlgorithm? =
      .ok (Algorithm.kernel kind swapped x y N p seed B) := by
  cases storage <;> cases kind <;> cases swapped
  all_goals
    simp only [source, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, List.map_cons, List.map_nil,
      Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
      MemAccess.eraseDType, MaskOpt.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, Except.bind, bind]
    simp only [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
    rfl

theorem prefix_run (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    stepStmts ((Algorithm.kernel kind Bool.false x y N p seed B).body.take 4) u =
      stepStmts ((Algorithm.kernel kind Bool.true x y N p seed B).body.take 4) u := by
  cases kind
  all_goals simp [Algorithm.kernel, Algorithm.seedStmt, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem algorithm_exec (kind : Random.IntegerType) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat) (u : BlockState) :
    exec (Algorithm.kernel kind Bool.false x y N p seed B) u =
      exec (Algorithm.kernel kind Bool.true x y N p seed B) u := by
  unfold exec
  conv_lhs => rw [← List.take_append_drop 4 (Algorithm.kernel kind Bool.false x y N p seed B).body]
  conv_rhs => rw [← List.take_append_drop 4 (Algorithm.kernel kind Bool.true x y N p seed B).body]
  simp only [Execution.stepStmts_append, prefix_run, Candidate.suffix_eq]

theorem executes (kind : Random.IntegerType) (swapped : Bool) (x y : RegionName)
    (N : Nat) (p : ℝ) (seed : SeedCarrier kind) (B : Nat)
    (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) :
    exec (Algorithm.kernel kind swapped x y N p seed B) u =
      some (Memory.storeBlock u (Execution.resultBase kind x N p seed B u)
        x y N B p (seedInt kind seed)) := by
  have he := Execution.exec_storeBlock kind x y N p seed B hB u hN hpid hseed
  rw [Execution.original, Algorithm.source_eq] at he
  cases swapped
  · exact he
  · rw [← algorithm_exec]
    exact he

theorem realizes_frame (storage : Storage) (kind : Random.IntegerType) (swapped : Bool)
    (x y : RegionName) (N : Nat) (p : ℝ) (seed : SeedCarrier kind)
    (B : Nat) (hB : SourceDomain.LegalBlock B) (u : BlockState)
    (hN : N < 2^31) (hpid : u.pid < SourceDomain.grid N B)
    (hseed : kind.inRange (seedInt kind seed) = Bool.true) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
      (source storage kind swapped x y N p seed B hB).eraseDType u
      (writes y N B u.pid) (expected x p (seedInt kind seed) B u) := by
  have he := executes kind swapped x y N p seed B hB u hN hpid hseed
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq (projection storage kind swapped x y N p seed B hB)
    intro s₀ t he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro j hj
    exact Memory.storeBlock_active_cell u (Execution.resultBase kind x N p seed B u)
      x y N B p (seedInt kind seed) j hj
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq (projection storage kind swapped x y N p seed B hB)
    intro s₀ t he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro addr hout
    exact (Memory.storeBlock_frame u (Execution.resultBase kind x N p seed B u)
      x y N B p (seedInt kind seed) addr.1 addr.2
      ((outside_iff y N B u.pid addr).mp hout)).trans
        (congrFun (congrFun (Execution.resultBase_mem kind x N p seed B u) addr.1) addr.2)

end
end VeriTile.Bench.TritonBenchG.SeededDropout.RealSurface
