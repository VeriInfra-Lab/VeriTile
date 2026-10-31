import bench.tritonbench_g.chunk_gated_attention.Sources.Mathematical_RealCumProgram

/-! Independent prefix formulas, pure scatter/readback, and local dot/store
lemmas. Complete execution is derived from the repaired source in
Proofs/CumulativeRealExecution. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumExecution
open VeriTile.Triton RealCumProgram
open ChunkCumsumVector.RealStep (pointer inputTile chunkPrefix address active)
open ChunkCumsumVector (lowerTriTile)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def tileActive (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) : Prop :=
  active p l (c*p.BT+i.1.val) i.2.1.val
instance (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) :
    Decidable (tileActive p l c i) := inferInstanceAs (Decidable (active _ _ _ _))

def offset (p : Layout) (l : Launch) (c : Nat) (i : TileIndex [p.BT,p.BS]) : Nat :=
  address p l (c*p.BT+i.1.val) i.2.1.val

def NoCollision (p : Layout) (l : Launch) (c : Nat) : Prop :=
  ∀ i j, tileActive p l c i → tileActive p l c j → offset p l c i = offset p l c j → i = j

def expected (p : Layout) (l : Launch) (c : Nat) (a : Nat → ℝ) (i : TileIndex [p.BT,p.BS]) : ℝ :=
  ∑ k ∈ Finset.univ.filter (fun k : Fin p.BT => k.val ≤ i.1.val),
    a (address p l (c*p.BT+k.val) i.2.1.val)

def outputTile (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) : Tile .real [p.BT,p.BS] :=
  Tile.mat fun i j => some (chunkPrefix p l a padding c i j)

def scatter (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BT,p.BS]).foldl (fun st i =>
    if tileActive p l c i then st.writeMem o (offset p l c i) (chunkPrefix p l a padding c i.1 i.2.1)
    else st) u

def footprint (p : Layout) (l : Launch) (c : Nat) (o : RegionName) : WriteFootprint :=
  fun cell => cell.1 = o ∧ ∃ i, tileActive p l c i ∧ cell.2 = offset p l c i

theorem dot_eval (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat) (u : BlockState)
    (hm : u.regs .real [p.BT,p.BT] "m_s" = some (lowerTriTile p.BT))
    (hb : u.regs .real [p.BT,p.BS] "b_s" = some (inputTile p l a padding c)) :
    evalOp (.dot (batch := []) (.ref .real [p.BT,p.BT] "m_s") (.ref .real [p.BT,p.BS] "b_s")
      {allowTF32 := some Bool.false}) u = some (outputTile p l a padding c) := by
  rw [evalOp_dot]
  erw [evalOp_ref,hm,evalOp_ref,hb]
  simp only [Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  exact ChunkCumsumVector.RealStep.dot_value p l a padding c i j

theorem store_eval (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState)
    (hp : u.regs .blockPtr [p.BT,p.BS] "p_o" = some (pointer p l o c))
    (hv : u.regs .real [p.BT,p.BS] "b_o" = some (outputTile p l a padding c)) :
    stepStmt (.store .real [p.BT,p.BS] (.blockPtr (.ref .blockPtr [p.BT,p.BS] "p_o") [0,1])
      (.ref .real [p.BT,p.BS] "b_o") .none) u = some (scatter p l o a padding c u) := by
  simp only [stepStmt,evalOp_ref,hp,hv,Option.bind_some,Option.bind_eq_bind,
    Bool.true_and,pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq]
  rfl

theorem prefix_active (p : Layout) (l : Launch) (a padding : Nat → ℝ) (c : Nat)
    (i : TileIndex [p.BT,p.BS]) (hi : tileActive p l c i) :
    chunkPrefix p l a padding c i.1 i.2.1 = expected p l c a i := by
  unfold chunkPrefix expected
  apply Finset.sum_congr rfl
  intro k hk
  have hk := (Finset.mem_filter.mp hk).2
  have ha : active p l (c*p.BT+k.val) i.2.1.val := ⟨by have := hi.1; omega,hi.2⟩
  simp only [ChunkCumsumVector.RealStep.value,ha,ite_true]

theorem scatter_frame (p : Layout) (l : Launch) (o : RegionName) (a padding : Nat → ℝ)
    (c : Nat) (u : BlockState) :
    BlockState.WriteWithin (footprint p l c o) u (scatter p l o a padding c u) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,i,hi,rfl⟩



#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumExecution
