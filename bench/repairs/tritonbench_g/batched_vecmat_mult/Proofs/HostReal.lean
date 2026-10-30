import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Host
import bench.repairs.tritonbench_g.batched_vecmat_mult.Proofs.Real

namespace VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.BatchedVecmatMult
open ValueProofs.Host
noncomputable section
local instance hostRealWordInhabited : Inhabited ValueProofs.Word := ⟨FP.Value.zero .fp32⟩

/-- Independent mathematical target on the actual allocated input words.
No program ID, reduction tree, execution, output memory or launch appears. -/
def realExpected (decode : ValueProofs.Word → ℝ) (p : Parameters)
    (mem : ValueProofs.Memory) (m n : Nat) : ℝ :=
  ∑ k : Fin p.K,
    decode ((mem p.A (m*p.K+k.val)).read .real) *
    decode ((mem p.B (m*p.N*p.K+n*p.K+k.val)).read .real)

theorem lift_read_real (decode : ValueProofs.Word → ℝ)
    (zero : decode (FP.Value.zero .fp32) = 0) (mem : ValueProofs.Memory)
    (r : RegionName) (o : Nat) :
    SourceGraph.realValue decode ((SourceGraph.Native.liftMemory mem r o).read .real) =
      decode ((mem r o).read .real) := by
  unfold SourceGraph.Native.liftMemory
  cases h : mem r o with
  | mk d v =>
    cases d <;> simp [SourceGraph.Native.mapCell,SourceGraph.Native.mapValue,
      Cell.read,defaultValue,SourceGraph.realValue] <;> exact zero.symm

theorem real_expected_lift (decode : ValueProofs.Word → ℝ)
    (zero : decode (FP.Value.zero .fp32) = 0) (p : Parameters)
    (mem : ValueProofs.Memory) (m n : Nat) :
    RealProofs.tensorExpected decode p (SourceGraph.Native.liftMemory mem) m n =
      realExpected decode p mem m n := by
  simp only [RealProofs.tensorExpected,realExpected,lift_read_real decode zero]

/-- The entire allocation/guard/setup/complete-grid/return composition under
the Real interpretation. Zero decoding fixes the mathematical interpretation
of the FP32 zero word; no floating arithmetic identity is assumed. -/
theorem returned_real_correct
    (decode : ValueProofs.Word → ℝ) (zero : decode (FP.Value.zero .fp32) = 0)
    (allocate : Allocator) (setup : Setup) (rest : Algebra SourceGraph.Expr)
    (plans : SourceGraph.PlanSelector) (a : Arguments) (initial : State ValueProofs.Word)
    (liveCells : RegionName → Nat → Prop)
    (frame : AllocationFrame allocate) (fresh : AllocationFresh liveCells allocate)
    (native : SetupLaw setup)
    (abi : ∀ l, prepare allocate a initial.mem = .ok l → TensorABI l)
    (storage : ∀ l, prepare allocate a initial.mem = .ok l → Index64.StorageLimits l.parameters) :
    match prepare allocate a initial.mem with
    | .error e => runGraph allocate setup rest plans a initial = .error (.preparation e)
    | .ok l => match setup l with
      | .error e => runGraph allocate setup rest plans a initial = .error (.setup e)
      | .ok () => IntegerAgreement l.parameters ∧ ∃ result : GraphTensor,
          runGraph allocate setup rest plans a initial = .ok (some result) ∧
          result.rows = a.M ∧ result.cols = a.N ∧ result.region = l.parameters.output ∧
          (∀ m : Fin l.parameters.M, ∀ n : Fin l.parameters.N,
            SourceGraph.realValue decode (SourceGraph.contract
              ((result.memory result.region (m.val*l.parameters.N+n.val)).read .real)) =
                realExpected decode l.parameters l.memory m.val n.val) ∧
          (∀ r o, ValueProofs.GridOutside l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory l.memory r o) ∧
          (∀ r o, OutsidePrepared l.parameters r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o) ∧
          (∀ r o, liveCells r o →
            result.memory r o = SourceGraph.Native.liftMemory initial.mem r o) := by
  cases hp : prepare allocate a initial.mem with
  | error e => simp [runGraph,hp]
  | ok l =>
    simp only []
    cases hs : setup l with
    | error e => simp [runGraph,hp,hs]
    | ok u =>
      cases u
      have prep := prepare_fields allocate a initial.mem l hp
      obtain ⟨final,he,hvalue,hframe⟩ := RealProofs.grid_real_correct decode rest plans
        l.parameters (SourceGraph.Native.liftState initial l.memory) (abi l hp)
        (prep.blocks.1 ▸ prep.positive.1) (prep.blocks.2.1 ▸ prep.positive.2.1)
        (by rw [prep.blocks.1,prep.dimensions.1]; exact Nat.dvd_of_mod_eq_zero prep.divisible.2.1)
        (by rw [prep.blocks.2.1,prep.dimensions.2.1]; exact Nat.dvd_of_mod_eq_zero prep.divisible.2.2)
        (by rw [prep.blocks.2.2,prep.dimensions.2.2]; exact Nat.dvd_of_mod_eq_zero prep.divisible.1)
      refine ⟨launch_integer_agreement l (storage l hp) (native l hs),
        ⟨l.parameters.M,l.parameters.N,l.parameters.output,final.mem⟩,?_,
        prep.dimensions.1,prep.dimensions.2.1,rfl,?_,hframe,?_,?_⟩
      · simp [runGraph,hp,hs,he]
      · intro m n
        exact (hvalue m n).trans (real_expected_lift decode zero l.parameters l.memory m.val n.val)
      · intro r o outside
        change final.mem r o = _
        rw [hframe r o (outside_grid l.parameters r o outside)]
        exact congrArg (SourceGraph.Native.mapCell (SourceGraph.Expr.input r o))
          (prepare_frame allocate a initial.mem l hp frame r o outside)
      · intro r o live
        change final.mem r o = _
        have outside := protected_outside allocate a initial.mem l hp liveCells fresh r o live
        rw [hframe r o (outside_grid l.parameters r o outside)]
        exact congrArg (SourceGraph.Native.mapCell (SourceGraph.Expr.input r o))
          (prepare_frame allocate a initial.mem l hp frame r o outside)

#specNonCircular realExpected avoiding [runGraph,runTensor,kernel,FP.Structural.exec]
#axiomsClean lift_read_real
#axiomsClean real_expected_lift
#axiomsClean returned_real_correct
end
end VeriTile.Bench.Repairs.BatchedVecmatMult.Whole
