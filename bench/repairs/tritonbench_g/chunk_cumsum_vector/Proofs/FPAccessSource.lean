import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.FPAccessPrefix
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.CandidateExecution
import VeriTile.Triton.Float.StructuralSafetyLoops
import bench.repairs.tritonbench_g.chunk_cumsum_vector.Proofs.AccessBounds

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.FPAccess
open VeriTile.Triton FP.Structural IndexProgram FPProgram FPValues AccessBounds
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem all_safe (M : Algebra α) (bounds : RegionBounds)
    (code : List ComputeStmt) (u : State α)
    (safe : ∀ st ∈ code, ∀ s, Safety.stepSafe M bounds st s) :
    Safety.traceSafeList M bounds code u := by
  induction code generalizing u with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) u)
    intro v _
    exact ih v (fun stmt member s => safe stmt (by simp [member]) s)

theorem init_safe (M : Algebra α) (bounds : RegionBounds) (h : Call) (c : Config) (u : State α) :
    Safety.traceSafeList M bounds (FPProgram.initCode h c) u := by
  apply all_safe
  intro st member s
  simp only [FPProgram.initCode, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeCompute_unfold,
      Safety.safeOp_unfold, triangularOp, startOp, activeOp, coordinateOp,
      FP.Structural.evalOp, Option.bind_some]
  apply all_safe
  intro st member s
  simp only [List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at member
  rcases member with rfl | rfl <;>
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

theorem source_range_safe (M : Algebra α) (dot : DotBackend α) (bounds : RegionBounds)
    (initial : State α) (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (positive : 0 < c.BT) (p : Geometry.Program h) (bounded : Bounds h p bounds)
    (q : Nat) (u : State α) (inv : FPExecution.Holds M dot initial h c p q u) :
    Safety.rangeIntSafe (machine M dot) bounds "i_t" (q:Int) (Geometry.chunks h c:Int) 1
      (FPProgram.loopBody h c) u := by
  let P := fun j : Int => fun s : State α =>
    0 ≤ j ∧ FPExecution.Holds M dot initial h c p j.toNat s
  refine Safety.rangeIntSafe_of_invariant (machine M dot) bounds "i_t" (Geometry.chunks h c:Int)
    1 (FPProgram.loopBody h c) P ?_ ?_ (q:Int) u ⟨by omega,by simpa using inv⟩
  · intro j hj s hs
    have inside : j.toNat < Geometry.chunks h c := by dsimp [P] at hs; omega
    have bound := chunk_bounds h c positive p bounds bounded j.toNat inside
    have pinned := FPExecution.bind_invariant M dot initial h c p j.toNat s hs.2
    have he := prefix_safe M dot bounds h c p.feature p.row j.toNat initial
      (FPExecution.carry M h c p.feature p.row initial j.toNat) _
      (by simp) pinned.batch pinned.head pinned.mask pinned.columns pinned.triangular
      pinned.carryReg pinned.reads pinned.undef
      (fun i active => (bound i active).1) (fun i active => (bound i active).2)
      _ (carry_safe _ bounds c Bool.false)
    change Safety.traceSafeList (machine M dot) bounds
      (prefixCode h c ++ [.ifThen (.ge .int .nil (.ref .int [] "i_t") (.constInt 0))
        [.assign .real [BS] "b_z" (.compute (.alg .fp32
          (if Bool.false then CandidateProgram.sumLeftCarryOp c else carryOp c))) ]]) _
    simpa only [Int.toNat_of_nonneg hs.1] using he
  · intro j hj s hs v ran
    have inside : j.toNat < Geometry.chunks h c := by dsimp [P] at hs; omega
    obtain ⟨next, executed, preserved⟩ := FPExecution.iteration M dot initial h layout c positive p j.toNat
      (s.setReg "i_t" .int [] (fun _ => (j.toNat:Int))) inside (by simp)
      (FPExecution.bind_invariant M dot initial h c p j.toNat s hs.2)
    simp only [Int.toNat_of_nonneg hs.1] at executed
    rw [executed] at ran
    cases ran
    refine ⟨by omega,?_⟩
    simpa only [show (j+(1:Nat)).toNat=j.toNat+1 by omega] using preserved

theorem source_safe (M : Algebra α) (dot : DotBackend α) (bounds : RegionBounds)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0 < c.BT)
    (p : Geometry.Program h) (bounded : Bounds h p bounds) (u : State α)
    (hf : u.pids 0 = p.feature) (hr : u.pids 1 = p.row) :
    FP.Structural.TraceSafe (machine M dot) bounds (Kernels.source h c) u := by
  unfold FP.Structural.TraceSafe
  rw [FPProgram.source_body, FPProgram.body, Safety.traceSafeList_append]
  refine ⟨init_safe _ bounds h c u, ?_⟩
  intro t executed
  rw [FPExecution.init_executes M dot h c p u hf hr] at executed
  cases executed
  rw [Safety.traceSafeList_cons]
  refine ⟨?_, fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe_unfold, Safety.safeOp_unfold, countOp, true_and,
    evalOp_unfold, numeric, bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Option.bind_some, Option.bind_eq_bind]
  exact source_range_safe M dot bounds u h layout c positive p bounded 0
    (FPExecution.initial M h c p u) (FPExecution.invariant M dot h c p u)

#axiomsClean source_safe

theorem candidate_range_safe (M : Algebra α) (dot : DotBackend α) (bounds : RegionBounds)
    (initial : State α) (h : Call) (layout : Geometry.OutputLayout h) (c : Config)
    (positive : 0 < c.BT) (p : Geometry.Program h) (bounded : Bounds h p bounds)
    (q : Nat) (u : State α) (inv : CandidateExecution.Holds M dot initial h c p q u) :
    Safety.rangeIntSafe (machine M dot) bounds "i_t" (q:Int) (Geometry.chunks h c:Int) 1
      (CandidateProgram.loopBody h c) u := by
  let P := fun j : Int => fun s : State α =>
    0 ≤ j ∧ CandidateExecution.Holds M dot initial h c p j.toNat s
  refine Safety.rangeIntSafe_of_invariant (machine M dot) bounds "i_t" (Geometry.chunks h c:Int)
    1 (CandidateProgram.loopBody h c) P ?_ ?_ (q:Int) u ⟨by omega,by simpa using inv⟩
  · intro j hj s hs
    have inside : j.toNat < Geometry.chunks h c := by dsimp [P] at hs; omega
    have bound := chunk_bounds h c positive p bounds bounded j.toNat inside
    have pinned := CandidateExecution.bind_invariant M dot initial h c p j.toNat s hs.2
    have he := prefix_safe M dot bounds h c p.feature p.row j.toNat initial
      (CandidateExecution.carry M h c p.feature p.row initial j.toNat) _
      (by simp) pinned.batch pinned.head pinned.mask pinned.columns pinned.triangular
      pinned.carryReg pinned.reads pinned.undef
      (fun i active => (bound i active).1) (fun i active => (bound i active).2)
      _ (carry_safe _ bounds c Bool.true)
    change Safety.traceSafeList (machine M dot) bounds
      (prefixCode h c ++ [.ifThen (.ge .int .nil (.ref .int [] "i_t") (.constInt 0))
        [.assign .real [BS] "b_z" (.compute (.alg .fp32
          (if Bool.true then CandidateProgram.sumLeftCarryOp c else carryOp c))) ]]) _
    simpa only [Int.toNat_of_nonneg hs.1] using he
  · intro j hj s hs v ran
    have inside : j.toNat < Geometry.chunks h c := by dsimp [P] at hs; omega
    obtain ⟨next, executed, preserved⟩ := CandidateExecution.iteration M dot initial h layout c positive p j.toNat
      (s.setReg "i_t" .int [] (fun _ => (j.toNat:Int))) inside (by simp)
      (CandidateExecution.bind_invariant M dot initial h c p j.toNat s hs.2)
    simp only [Int.toNat_of_nonneg hs.1] at executed
    rw [executed] at ran
    cases ran
    refine ⟨by omega,?_⟩
    simpa only [show (j+(1:Nat)).toNat=j.toNat+1 by omega] using preserved

theorem candidate_safe (M : Algebra α) (dot : DotBackend α) (bounds : RegionBounds)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0 < c.BT)
    (p : Geometry.Program h) (bounded : Bounds h p bounds) (u : State α)
    (hf : u.pids 0 = p.feature) (hr : u.pids 1 = p.row) :
    FP.Structural.TraceSafe (machine M dot) bounds (Kernels.sumLeft h c) u := by
  unfold FP.Structural.TraceSafe
  rw [CandidateProgram.source_body, CandidateProgram.body, Safety.traceSafeList_append]
  refine ⟨init_safe _ bounds h c u, ?_⟩
  intro t executed
  rw [CandidateExecution.init_executes M dot h c p u hf hr] at executed
  cases executed
  rw [Safety.traceSafeList_cons]
  refine ⟨?_, fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe_unfold, Safety.safeOp_unfold, countOp, true_and,
    evalOp_unfold, numeric, bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Option.bind_some, Option.bind_eq_bind]
  exact candidate_range_safe M dot bounds u h layout c positive p bounded 0
    (CandidateExecution.initial M h c p u) (CandidateExecution.invariant M dot h c p u)

#axiomsClean candidate_safe

end
end VeriTile.Bench.Repairs.ChunkCumsumVector.FPAccess
