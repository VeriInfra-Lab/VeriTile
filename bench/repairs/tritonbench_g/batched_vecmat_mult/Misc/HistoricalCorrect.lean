import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.CheckedHost
import bench.repairs.tritonbench_g.batched_vecmat_mult.Kernels
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Real
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.FiniteHost

/-! Independent Real target: out[m,n] = ∑k A[m,k]*B[m,n,k]. The displayed
blocked sum enumerates each k exactly once under the original K divisibility
guard. Shapes are symbolic; zero K returns zeros and empty grids are covered.
All source accesses are unmasked, so original whole-block guards are retained.
Ordinary separate fresh output, dense float32 storage and successful launch ABI
are explicit; no i32-safe-shape restriction or numerical premise is added. -/
namespace VeriTile.Bench.Repairs.BatchedVecmatMult.HistoricalCorrectness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open scoped VeriTile.Triton.StreamMasked2DKernelIO₂
noncomputable section
local instance : Inhabited SourceGraph.Expr := ⟨SourceGraph.Expr.zero⟩
local instance : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Historical cast-free source contract, retained with its original complete
windows and flat-memory frame. The selected INT64 body's full-grid theorem is
`Correctness.batched_vecmat_grid_real`, through the actual SourceGraph source. -/
def io (A B out : RegionName) (M N BM BN BK T : Nat) : StreamMasked2DKernelIO₂ where
  kernel := (TypedSource.original A B out M N (BK*T) BM BN BK).eraseDType
  projection := RealSource.projection A B out M N (BK*T) BM BN BK Bool.false
  inp1 := A
  inp2 := B
  out := out
  T := T
  B1 := BM*BK
  B2 := BN*BM*BK
  C := BM*BN
  outDType := .real
  read1 := fun pm _ c l => (pm*BM+l.val/BK)*(BK*T)+(c.val*BK+l.val%BK)
  read2 := fun pm pn c l => (pm*BM+l.val/BK%BM)*N*(BK*T)+
    (pn*BN+l.val/BK/BM)*(BK*T)+(c.val*BK+l.val%BK)
  write := fun pm pn l => (pm*BM+l.val/BN)*N+(pn*BN+l.val%BN)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True

/-- The mathematical target reads only the two input windows. -/
def expected (BM BN BK T : Nat) (xs : Fin T → Fin (BM*BK) → ℝ)
    (ys : Fin T → Fin (BN*BM*BK) → ℝ) (l : Fin (BM*BN)) : ℝ :=
  ∑ c : Fin T, ∑ e : Fin BK,
    xs c (VeriTile.Bench.TritonBenchG.BatchedVecmatMult.aLane BM BN BK l e) *
    ys c (VeriTile.Bench.TritonBenchG.BatchedVecmatMult.bLane BM BN BK l e)

specification repaired_real_correct (A B out : RegionName) (M N BM BN BK T : Nat)
    (hBK : 0 < BK) (hBN : BN ≤ N) :
    Spec.Real (io A B out M N BM BN BK T ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l) := by
  exact VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealProofs.original_correct
    A B out M N BM BN BK T hBK hBN

/-- Independent mathematical formula on dense logical input coordinates. -/
def tensorExpected (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (mem : SourceGraph.Memory) (m n : Nat) : ℝ :=
  ∑ k : Fin p.K,
    SourceGraph.realValue decode ((mem p.A (m*p.K+k.val)).read .real) *
    SourceGraph.realValue decode ((mem p.B (m*p.N*p.K+n*p.K+k.val)).read .real)

/-- All output coordinates of the complete grid, with successful source
execution and frame; the mathematical sum is independent of execution. -/
specification repaired_grid_real_correct (decode : ValueProofs.Word → ℝ)
    (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (p : Parameters) (initial : State SourceGraph.Expr)
    (separate : SourceGraph.InputsSeparate p)
    (hBM : 0 < p.BM) (hBN : 0 < p.BN) (hM : p.BM ∣ p.M) (hN : p.BN ∣ p.N)
    (hK : p.BK ∣ p.K) :
    Spec.Real (∃ final, SourceGraph.runGrid rest plans p initial = some final ∧
      (∀ m : Fin p.M, ∀ n : Fin p.N,
        SourceGraph.realValue decode (SourceGraph.contract
          ((final.mem p.output (m.val*p.N+n.val)).read .real)) =
          tensorExpected decode p initial.mem m.val n.val) ∧
      (∀ r o, SourceGraph.GridOutside p r o → final.mem r o = initial.mem r o)) := by
  exact VeriTile.Bench.Repairs.BatchedVecmatMult.RealProofs.grid_real_correct
    decode rest plans p initial separate hBM hBN hM hN hK

#specNonCircular expected avoiding [kernel, FP.Structural.exec]
#axiomsClean repaired_real_correct
#axiomsClean repaired_grid_real_correct

/-- Independent full-K mathematical formula on the actual allocated inputs. -/
def hostExpected (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (mem : ValueProofs.Memory) (m n : Nat) : ℝ :=
  ∑ k : Fin p.K,
    decode ((mem p.A (m*p.K+k.val)).read .real) *
    decode ((mem p.B (m*p.N*p.K+n*p.K+k.val)).read .real)

/-- The complete Real host result, with original allocation/guard order,
native setup failures, symbolic finite indices, return shape/identity, every
logical output coordinate and pre-call frame. `zero` only fixes how the zero
input word is interpreted mathematically; no FP arithmetic law is used. -/
specification repaired_returned_real_correct
    (decode : ValueProofs.Word → ℝ) (zero : decode (FP.Value.zero .fp32) = 0)
    (allocate : ValueProofs.Host.Allocator) (setup : Whole.Setup)
    (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (a : ValueProofs.Host.Arguments) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : Whole.AllocationFrame allocate) (fresh : Whole.AllocationFresh liveCells allocate)
    (native : Whole.SetupLaw setup)
    (abi : ∀ l, ValueProofs.Host.prepare allocate a initial.mem = .ok l → ValueProofs.Host.TensorABI l)
    (storage : ∀ l, ValueProofs.Host.prepare allocate a initial.mem = .ok l →
      Index64.StorageLimits l.parameters) :
    Spec.Real (match ValueProofs.Host.prepare allocate a initial.mem with
    | .error e => FiniteHost.runGraph allocate setup rest plans a initial = .error (.preparation e)
    | .ok l => match setup l with
      | .error e => FiniteHost.runGraph allocate setup rest plans a initial = .error (.setup e)
      | .ok () => Whole.IntegerAgreement l.parameters ∧ ∃ result : Whole.GraphTensor,
          FiniteHost.runGraph allocate setup rest plans a initial = .ok (some result) ∧
          result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            SourceGraph.realValue decode (SourceGraph.contract
              ((result.memory result.region (m.val*l.parameters.N+n.val)).read .real)) =
                hostExpected decode l.parameters l.memory m.val n.val) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory l.memory r o) ∧
          (∀ r o, Whole.OutsidePrepared l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o) ∧
          (∀ r o, liveCells r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o)) := by
  exact FiniteHost.returned_real_correct decode zero allocate setup rest plans a initial
    liveCells frame fresh native abi storage

/-- Exact scalar admission composed with the complete finite source result. -/
specification repaired_checked_returned_real_correct
    (decode : ValueProofs.Word → ℝ) (zero : decode (FP.Value.zero .fp32) = 0)
    (allocate : ValueProofs.Host.Allocator) (setup : Whole.Setup)
    (rest : Algebra SourceGraph.Expr) (plans : SourceGraph.PlanSelector)
    (raw : Contract.Arguments) (admitted : Contract.InputContract raw) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : Whole.AllocationFrame allocate) (fresh : Whole.AllocationFresh liveCells allocate)
    (native : Whole.SetupLaw setup)
    (abi : ∀ l, ValueProofs.Host.prepare allocate (Contract.toArguments raw) initial.mem = .ok l → ValueProofs.Host.TensorABI l)
    (storage : ∀ l, ValueProofs.Host.prepare allocate (Contract.toArguments raw) initial.mem = .ok l →
      Index64.StorageLimits l.parameters) :
    Spec.Real (match ValueProofs.Host.prepare allocate (Contract.toArguments raw) initial.mem with
    | .error e => CheckedHost.runGraph allocate setup rest plans raw initial = some (.error (.preparation e))
    | .ok l => match setup l with
      | .error e => CheckedHost.runGraph allocate setup rest plans raw initial = some (.error (.setup e))
      | .ok () => Whole.IntegerAgreement l.parameters ∧ ∃ result : Whole.GraphTensor,
          CheckedHost.runGraph allocate setup rest plans raw initial = some (.ok (some result)) ∧
          result.rows = (Contract.toArguments raw).M ∧ result.cols = (Contract.toArguments raw).N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            SourceGraph.realValue decode (SourceGraph.contract
              ((result.memory result.region (m.val*l.parameters.N+n.val)).read .real)) =
                hostExpected decode l.parameters l.memory m.val n.val) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory l.memory r o) ∧
          (∀ r o, Whole.OutsidePrepared l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o) ∧
          (∀ r o, liveCells r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o)) := by
  simpa only [CheckedHost.graph_admitted _ _ _ _ _ _ admitted, Option.some.injEq] using
    repaired_returned_real_correct decode zero allocate setup rest plans
      (Contract.toArguments raw) initial liveCells frame fresh native abi storage

#axiomsClean repaired_checked_returned_real_correct


#specNonCircular hostExpected avoiding [kernel,FiniteHost.runGraph,FiniteHost.runTensor,FP.Structural.exec]
#axiomsClean repaired_returned_real_correct
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.HistoricalCorrectness
