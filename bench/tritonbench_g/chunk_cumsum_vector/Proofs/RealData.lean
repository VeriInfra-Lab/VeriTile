import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealValues

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.RealData
open VeriTile.Triton RealValues
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
abbrev RealMath := VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath.carry

/-- Active source cells use the logical input; inactive lanes use explicit zero.
The retained padding argument is ignored by the repaired source. -/
def value (h : Call) (c : Config) (feature row : Nat) (a padding : Nat → ℝ)
    (t : Nat) (j : Fin BS) : ℝ :=
  if t<h.T ∧ feature*BS+j.val<h.S then
    a (h.inputStrides.address h.H row t (feature*BS+j.val))
  else 0
def carry (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat) (j : Fin BS) : ℝ :=
  VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath.carry (fun t => f t j) c.BT q
/-- Independent logical prefix, with no chunk size, launch order, or padding. -/
def expected (h : Call) (feature row : Nat) (a : Nat → ℝ) (t : Nat) (j : Fin BS) : ℝ :=
  ∑ k ∈ (Finset.range h.T).filter (fun k => k ≤ t),
    a (h.inputStrides.address h.H row k (feature*BS+j.val))

theorem value_chunk (h : Call) (c : Config) (hc : 0<c.BT) (p : Geometry.Program h)
    (a padding : Nat → ℝ) (q : Nat) (hq : q<Geometry.chunks h c) (i : Fin c.BT) (j : Fin BS) :
    (if Geometry.active h c p.feature q i.val j.val then
      a (h.inputStrides.address h.H p.row (Geometry.time h c q i.val) (Geometry.column h p.feature j.val))
    else 0)=
      value h c p.feature p.row a padding (q*c.BT+i.val) j := by
  have hstart := Geometry.chunk_start h c hc q hq
  have hdiv : (q*c.BT+i.val)/c.BT=q := by rw [Nat.add_comm,Nat.add_mul_div_right _ _ hc,Nat.div_eq_of_lt i.isLt,Nat.zero_add]
  have hmod : (q*c.BT+i.val)%c.BT=i.val := by simp [Nat.add_mod,Nat.mod_eq_of_lt i.isLt]
  have iff := Geometry.active_iff h c p q i.val j.val hstart
  by_cases ha : Geometry.active h c p.feature q i.val j.val
  · have coords := Geometry.active_coordinates h c p q i.val j.val ha
    simp [ha,value,iff.mp ha,coords.1,coords.2]
  · simp [ha,value,iff.not.mp ha,hdiv,hmod]

theorem next_carry (c : Config) (f : Nat → Fin BS → ℝ) (q : Nat) :
    nextCarry c f q (carry c f q)=carry c f (q+1) := by
  funext j
  exact (VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath.carry_succ (fun t => f t j) c.BT q).symm

theorem prefix_at_chunk (h : Call) (c : Config) (p : Geometry.Program h)
    (a padding : Nat → ℝ) (q : Nat) (i : Fin c.BT) (j : Fin BS)
    (hi : q*c.BT+i.val<h.T) (hj : p.feature*BS+j.val<h.S) :
    carry c (value h c p.feature p.row a padding) q j+
      chunkPrefix c (value h c p.feature p.row a padding) q i j=
      expected h p.feature p.row a (q*c.BT+i.val) j := by
  rw [add_comm]
  refine (VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath.prefix_from_chunk
    (fun t => value h c p.feature p.row a padding t j) h.T c.BT q i hi).trans ?_
  simp only [expected,VeriTile.Bench.TritonBenchG.ChunkCumsumVector.RealMath.prefixValue]
  apply Finset.sum_congr rfl
  intro k hk
  have ht := Finset.mem_range.mp (Finset.mem_filter.mp hk).1
  simp [value,ht,hj]

#axiomsClean value_chunk
#axiomsClean prefix_at_chunk
end
end VeriTile.Bench.Repairs.ChunkCumsumVector.RealData
