import bench.tritonbench_g.matmul_persistent_triton.Proofs.OutputCoverage

namespace VeriTile.Bench.Repairs.MatmulPersistentTriton.Real
open VeriTile.Bench.TritonBenchG.MatmulPersistentTriton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 5000000

/-- Independent matrix inputs read only from caller entry storage. -/
def inputs (b : Buffers) (entry : Memory ℝ) : RealMath.Inputs :=
  ⟨fun row k => entry (b.A.address row k),fun k column => entry (b.B.address k column)⟩

/-- Exact-real interpretation of the actual source dotAcc and explicit cast.
No statement here identifies rounded native arithmetic with real arithmetic. -/
def arithmetic : Source.Arithmetic ℝ ℝ where
  inputZero := 0
  accumulatorZero := 0
  dot width a b acc := fun row column => (∑ k : Fin width, a row k*b k column)+acc row column
  cast _ value := value

theorem block (p : Problem) (b : Buffers) (entry : Memory ℝ) (tile iteration : Nat)
    (row : Fin 128) (column : Fin 256) :
    (∑ k : Fin p.profile.blockK,
      Source.loadA arithmetic p b entry tile iteration row k*
      Source.loadB arithmetic p b entry tile iteration k column)=
    RealMath.tileSum p.K p.profile.blockK iteration (inputs b entry)
      (clamp p.M (p.row tile row.val)) (clamp p.N (p.column tile column.val)) := by
  unfold RealMath.tileSum
  apply Finset.sum_congr rfl
  intro k _
  by_cases activeK : p.k iteration k.val<p.K
  · have mask : k.val<p.K-iteration*p.profile.blockK := by unfold Problem.k at activeK; omega
    change iteration*p.profile.blockK+k.val<p.K at activeK
    simp [Source.loadA,Source.loadB,mask,clamp,inputs,Problem.k,activeK]
  · have mask : ¬k.val<p.K-iteration*p.profile.blockK := by unfold Problem.k at activeK; omega
    simp [Source.loadA,Source.loadB,activeK,mask,arithmetic]

theorem partial_value (p : Problem) (b : Buffers) (entry : Memory ℝ) (tile count : Nat)
    (row : Fin 128) (column : Fin 256) :
    Source.accumulator arithmetic p b entry tile count row column=
      RealMath.partialSum p.K p.profile.blockK count (inputs b entry)
        (clamp p.M (p.row tile row.val)) (clamp p.N (p.column tile column.val)) := by
  induction count with
  | zero => simp [Source.accumulator,arithmetic,RealMath.partial_zero]
  | succ count ih =>
    change (∑ k : Fin p.profile.blockK,
      Source.loadA arithmetic p b entry tile count row k*Source.loadB arithmetic p b entry tile count k column)+
      Source.accumulator arithmetic p b entry tile count row column=_
    rw [block,ih,RealMath.partial_succ]

/-- All K blocks, including the zero-filled final block, give exactly the
independent K-term formula in the real interpretation. -/
theorem completed (p : Problem) (b : Buffers) (entry : Memory ℝ) (tile : Nat)
    (lane : Source.Lane) (ha : Source.active p tile lane) :
    Source.accumulator arithmetic p b entry tile p.kTiles lane.1 lane.2=
      RealMath.formula p.K (inputs b entry) (p.row tile lane.1.val) (p.column tile lane.2.val) := by
  rw [partial_value]
  unfold Problem.kTiles
  rw [RealMath.partial_complete p.K p.profile.blockK (blockK_positive p.profile)]
  simp only [clamp,if_pos ha.1,if_pos ha.2]

/-- Complete positive-branch source grid, with current-memory reads and all
persistent iterations, returns each mathematical matrix coordinate. -/
theorem output (p : Problem) (b : Buffers) (layout : Layout p b)
    (hs : 0<p.SMS) (entry : Memory ℝ) (row column : Nat) (hr : row<p.M) (hc : column<p.N) :
    Source.run arithmetic p b entry (b.C.address row column)=
      RealMath.formula p.K (inputs b entry) row column := by
  obtain ⟨tile,ht,lane,ha,hrow,hcolumn⟩ := Source.output_covered p hs row column hr hc
  have address : Source.outputAddress p b tile lane=b.C.address row column := by
    rw [Source.output_address p b tile lane ha,hrow,hcolumn]
  rw [← address,Source.source_output arithmetic p b layout hs entry tile ht lane ha]
  change Source.accumulator arithmetic p b entry tile p.kTiles lane.1 lane.2=_
  rw [completed p b entry tile lane ha,hrow,hcolumn]

theorem frame (p : Problem) (b : Buffers) (layout : Layout p b) (entry : Memory ℝ)
    (point : Address) (outside : ∀ row<p.M, ∀ column<p.N, b.C.address row column≠point) :
    Source.run arithmetic p b entry point=entry point := Source.source_frame _ _ _ layout _ _ outside

#axiomsClean output
#axiomsClean frame
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MatmulPersistentTriton.Real
