import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealAccessPrefix
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealExecution
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.AccessBounds

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealAccess
open VeriTile.Triton IndexProgram RealProgram RealValues RealData AccessBounds
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem init_safe (bounds : RegionBounds) (h : Call) (c : Config) (u : BlockState) :
    Stmt.TraceSafeListR .triv bounds (RealProgram.initCode h c) u := by
  apply Stmt.TraceSafeListR.of_forall
  intro st member s
  simp only [RealProgram.initCode, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def, triangularOp, startOp, activeOp, coordinateOp]

theorem range_safe (bounds : RegionBounds) (initial : BlockState)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0 < c.BT)
    (p : Geometry.Program h) (a : Nat → ℝ) (bounded : Bounds h p bounds)
    (q : Nat) (u : BlockState) (inv : RealLoop.Holds initial h c p a q u) :
    Stmt.forRangeTraceSafeR .triv bounds "i_t" q (Geometry.chunks h c) 1
      (RealProgram.loopBody h c) u := by
  apply Stmt.forRangeTraceSafeR_inv .triv bounds "i_t" (Geometry.chunks h c) 1
    (RealProgram.loopBody h c) (RealLoop.Holds initial h c p a) ?_ q u inv
  intro k s inside invariant
  constructor
  · let f := value h c p.feature p.row a (initial.undef h.input)
    let acc := carry c f k
    let entry := s.setReg "i_t" .nat [] (Tile.scalar k)
    have bound := chunk_bounds h c positive p bounds bounded k inside
    change Stmt.TraceSafeListR .triv bounds
      (prefixCode h c ++ [.ifThen (.ge .nat .nil (.ref .nat [] "i_t") (.constNat 0))
        [.assign .real [BS] "b_z" (carryOp c)]]) entry
    apply prefix_safe bounds h c p.feature p.row k f acc entry
      (by simp [entry]) (by simp [entry,invariant.batch]) (by simp [entry,invariant.head])
      (by simp [entry,invariant.mask]) (by simp [entry,invariant.columns])
      (by simp [entry,invariant.triangular]) (by simpa [entry,acc,f] using invariant.accumulated)
    · intro i j
      simpa [entry,invariant.input,invariant.undef,f] using
        value_chunk h c positive p a (initial.undef h.input) k inside i j
    · exact fun i active => (bound i active).1
    · exact fun i active => (bound i active).2
    · exact carry_safe bounds c
  · obtain ⟨next, ran, preserved⟩ := RealLoop.iteration initial h c positive p a
      layout.distinct (RealExecution.layout_unique h p layout) k s inside invariant
    exact ⟨next, by simpa only [stepStmtsR_triv] using ran, preserved⟩

theorem source_safe (bounds : RegionBounds)
    (h : Call) (layout : Geometry.OutputLayout h) (c : Config) (positive : 0 < c.BT)
    (p : Geometry.Program h) (bounded : Bounds h p bounds) (u : BlockState)
    (hf : u.pids 0 = p.feature) (hr : u.pids 1 = p.row) :
    Kernel.TraceSafeR .triv bounds (Kernels.source h c).eraseDType.toAlgKernel u := by
  unfold Kernel.TraceSafeR
  rw [RealProgram.source_body, RealProgram.body]
  apply Stmt.TraceSafeListR.append_intro _ u (init_safe bounds h c u)
  intro t executed
  rw [stepStmtsR_triv, RealExecution.init_executes h c p u hf hr] at executed
  cases executed
  apply Stmt.TraceSafeListR.cons_intro
  · simp only [Stmt.TraceSafeR, countOp, Op.SafeAtR.eq_def, evalOpR_triv,
      evalOp_constNat, evalOp_div, evalOp_sub, evalOp_add, Option.bind_some,
      Option.bind_eq_bind, Tile.scalar_data, true_and]
    exact range_safe bounds u h layout c positive p (u.readMem h.input) bounded 0
      (RealExecution.initial h c p u) (RealExecution.invariant h c p u)
  · intro _ _
    exact Stmt.TraceSafeListR.nil_intro

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealAccess
