import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealFull
import bench.tritonbench_g.matmul_persistent_triton.Proofs.RealZeroK
import VeriTile.Triton.Correctness

/-! Standard public correctness surfaces for the complete source projection.
The region realization observes exact tagged cells and includes the full frame.
Successful finite-flat execution remains a separate conjunct of the public
contract, since the standard region surface alone is partial correctness. -/
namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSurface
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000

/-- One lane in one persistent tile assigned to the current program. -/
abbrev OutputIndex (p : Parameters) := Nat × TileIndex [p.BM,p.BN]

/-- Exactly all active source stores for this PID, including every persistent
tile and excluding masked lanes. -/
def writes (p : Parameters) (pid : Nat) : ComputeCorrect.WriteMap (OutputIndex p) :=
  @ComputeCorrect.WriteMap.writeIf (OutputIndex p)
    (fun i => RealIO.outputMask p pid i.1 i.2) (Classical.decPred _)
    (fun i => (p.C,RealIO.outputWindow p pid i.1 i.2))

/-- Independent input-only GEMM with the exact real-cell tag retained. -/
def expected (p : Parameters) (xs : RealMath.Inputs) (pid : Nat) : OutputIndex p → MemCell :=
  fun i => .real (RealIO.expected p xs pid i.1 i.2)

/-- The complement of the standard write map is exactly the already proved
whole-source raw frame, including all masked lanes, stride gaps and regions. -/
theorem outside_iff (p : Parameters) (pid : Nat) (addr : MemCellAddr) :
    (∀ i, writes p pid i ≠ some addr) ↔ RealLoop.Outside p pid addr.1 addr.2 := by
  classical
  constructor
  · intro h q hq j hj he
    apply h (q,j)
    have hm : RealIO.outputMask p pid q j := ⟨hq,hj⟩
    have ha : (p.C,RealIO.outputWindow p pid q j) = addr := Prod.ext he.1 he.2
    simp only [writes,ComputeCorrect.WriteMap.writeIf,hm,ite_true,ha]
  · intro h i he
    by_cases hm : RealIO.outputMask p pid i.1 i.2
    · have ha : (p.C,RealIO.outputWindow p pid i.1 i.2) = addr := by
        simpa only [writes,ComputeCorrect.WriteMap.writeIf,hm,ite_true,Option.some.injEq] using he
      exact h i.1 hm.1 i.2 hm.2 ⟨congrArg Prod.fst ha,congrArg Prod.snd ha⟩
    · simp only [writes,ComputeCorrect.WriteMap.writeIf,hm,ite_false] at he
      cases he

/-- Complete region-level standard realization and raw frame for both storage
profiles and variants, obtained from the actual full-source successful run.
The finite-flat termination/safety contract is retained in `RealIO.Implements`.
This erases floating precision and does not assert native GPU admission. -/
theorem realizes_frame (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (h : RealIO.Admissible p) (s : BlockState) (xs : RealMath.Inputs)
    (hl : RealIO.Launch p s) (hi : RealInput.Represents p xs s) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
      (p.source storage v).eraseDType s (writes p (s.pids 0)) (expected p xs (s.pids 0)) := by
  classical
  obtain ⟨t,he,hy,_hi,hf⟩ := RealFull.executes p v h s xs hl.pid_bound hi
  constructor
  · unfold writes
    rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgorithm_eq (RealSource.projection p storage v)
    intro s₀ t' he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro i hm
    exact hy i.1 hm.1 i.2 hm.2
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq (RealSource.projection p storage v)
    intro s₀ t' he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro addr hout
    exact hf addr.1 addr.2 ((outside_iff p (s.pids 0) addr).mp hout)

/-- No store belongs to the actual empty-K execution. The standard frame for
this map consequently states preservation of every raw memory cell. -/
def emptyWrites : ComputeCorrect.WriteMap PUnit := fun _ => none

/-- Standard region-level realization for empty K. The empty write map gives
no invented output values: the nonvacuous claim is preservation of all memory. -/
theorem zero_realizes_frame (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (s : BlockState) (hK : p.K = 0) (hBK : 0 < p.BK) :
    ComputeCorrect.RealizesFrame_without_Rounding (α := MemCell)
      (p.source storage v).eraseDType s emptyWrites (fun _ => .real 0) := by
  obtain ⟨he,hf⟩ := RealZeroK.executes p v s hK hBK
  constructor
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq (RealSource.projection p storage v)
    intro s₀ t he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro i
    exact True.intro
  · apply ComputeKernel.computeCorrect_of_toAlgorithm_eq (RealSource.projection p storage v)
    intro s₀ t he' hs
    subst s₀
    rw [he] at he'
    cases he'
    intro addr _
    exact congrFun (congrFun hf addr.1) addr.2

/-- Successful finite-flat execution with preservation of every memory cell.
It supplements the standard region surface with termination and keeps the
original allocation/undef assumptions explicit. -/
def FlatPreservesMemory (kernel : Kernel) : Prop :=
  ∀ (A : FlatAlloc) (s : BlockState), A.Disjoint →
    (∀ r, r ∉ A.regions → A.extent r = 0) → s.undef = (fun _ _ => 0) →
    ∃ t, exec (A.flattenKernel kernel) (A.flattenState s) = some t ∧
      t.mem = (A.flattenState s).mem

theorem zero_flat_preserves (p : Parameters) (storage : TypedSource.Storage) (v : Bool)
    (hK : p.K = 0) (hBK : 0 < p.BK) :
    FlatPreservesMemory (p.source storage v).eraseDType.toAlgKernel := by
  intro A s hd hc hu
  exact RealZeroK.flat_unchanged p storage v A s hd hc hu hK hBK

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSurface
