import bench.repairs.tritonbench_g.bmm_chunk_fwd.Proofs.Host

/-! Dense returned-output ownership for the actual three-ax host grid.
Injectivity is derived from the strides, rather than assumed of the launch. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

abbrev PID (h : Call) (c : Config) := GridIndex (h.grid c)
def p0 {h : Call} {c : Config} (p : PID h c) : Nat := (p ⟨0, by change 0 < 3; decide⟩).val
def p1 {h : Call} {c : Config} (p : PID h c) : Nat := (p ⟨1, by change 1 < 3; decide⟩).val
def p2 {h : Call} {c : Config} (p : PID h c) : Nat := (p ⟨2, by change 2 < 3; decide⟩).val

theorem p2_lt {h : Call} {c : Config} (p : PID h c) :
    p2 p < h.shape.chunks * h.shape.groups := (p ⟨2, by change 2 < 3; decide⟩).isLt
theorem p1_lt {h : Call} {c : Config} (p : PID h c) :
    p1 p < h.shape.batch := (p ⟨1, by change 1 < 3; decide⟩).isLt

theorem pid_ext {h : Call} {c : Config} {p q : PID h c}
    (h0 : p0 p = p0 q) (h1 : p1 p = p1 q) (h2 : p2 p = p2 q) : p = q := by
  funext ax
  have ha : ax.val < 3 := ax.isLt
  rcases ax with ⟨ax, ha⟩
  interval_cases ax <;> apply Fin.ext <;> assumption

theorem mul_add_inj {n a b x y : Nat} (hx : x < n) (hy : y < n)
    (he : a * n + x = b * n + y) : a = b ∧ x = y := by
  have hn : 0 < n := Nat.zero_lt_of_lt hx
  have hm := congrArg (· % n) he
  simp [Nat.add_mod, Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] at hm
  have hd' : a = b := by nlinarith
  exact ⟨hd', hm⟩

theorem mul_add_bound {a b n x : Nat} (ha : a < b) (hx : x < n) :
    a * n + x < b * n := by
  have hab : a + 1 ≤ b := ha
  have hh := Nat.mul_le_mul_right n hab
  nlinarith

theorem output_offset (h : Call) (c : Config) (p₀ p₁ p₂ : Nat)
    (i : TileIndex [c.M, c.N]) :
    RealViews.outputOffset c.M c.N h.parameters p₀ p₁ p₂ i =
      ((p₁ * (h.shape.chunks * h.shape.groups) + p₂) * h.shape.chunk +
        RealViews.row c.M c.N h.parameters p₀ i.1) * h.shape.chunk +
        RealViews.column c.N h.parameters p₀ i.2.1 := by
  have hh : p₂ / h.shape.groups * h.shape.groups +
      (p₂ - p₂ / h.shape.groups * h.shape.groups) = p₂ :=
    Nat.add_sub_of_le (Nat.div_mul_le_self p₂ h.shape.groups)
  change (p₁ * ((h.shape.chunks * h.shape.groups) * h.shape.chunk * h.shape.chunk) +
    (p₂ / h.shape.groups) * (h.shape.groups * h.shape.chunk * h.shape.chunk) +
    (p₂ - p₂ / h.shape.groups * h.shape.groups) *
      (if h.shape.grouped then h.shape.chunk * h.shape.chunk else 0)) +
    (RealViews.row c.M c.N h.parameters p₀ i.1 * h.shape.chunk +
      RealViews.column c.N h.parameters p₀ i.2.1 * 1) = _
  cases hg : h.shape.grouped
  · have hgroup : h.shape.groups = 1 := by simp [Shape.groups, hg]
    simp only [hg, Bool.false_eq_true, ↓reduceIte, hgroup, Nat.div_one,
      Nat.mul_one, Nat.mul_zero, Nat.add_zero]
    ring
  · simp only [hg, ↓reduceIte, Nat.mul_one]
    calc
      _ = ((p₁ * (h.shape.chunks * h.shape.groups) +
        (p₂ / h.shape.groups * h.shape.groups +
          (p₂ - p₂ / h.shape.groups * h.shape.groups))) * h.shape.chunk +
          RealViews.row c.M c.N h.parameters p₀ i.1) * h.shape.chunk +
          RealViews.column c.N h.parameters p₀ i.2.1 := by ring
      _ = _ := by rw [hh]

theorem output_distinct (h : Call) (c : Config) :
    RealViews.OutputDistinct c.M c.N h.parameters := by
  intro p₀ p₁ p₂ i j hi hj he
  rw [output_offset, output_offset] at he
  obtain ⟨he, hcol⟩ := mul_add_inj hi.2.2 hj.2.2 he
  obtain ⟨_, hrow⟩ := mul_add_inj hi.2.1 hj.2.1 he
  have hri : i.1 = j.1 := Fin.ext (Nat.add_left_cancel hrow)
  have hcj : i.2.1 = j.2.1 := Fin.ext (Nat.add_left_cancel hcol)
  exact Prod.ext hri (Prod.ext hcj (Subsingleton.elim _ _))

theorem output_owner (h : Call) (c : Config) (hM : 0 < c.M) (hN : 0 < c.N)
    (p q : PID h c) (i j : TileIndex [c.M, c.N])
    (hi : (RealViews.io Bool.false c.M c.N c.K h.parameters).outputMask (p0 p) (p1 p) (p2 p) i)
    (hj : (RealViews.io Bool.false c.M c.N c.K h.parameters).outputMask (p0 q) (p1 q) (p2 q) j)
    (he : RealViews.outputOffset c.M c.N h.parameters (p0 p) (p1 p) (p2 p) i =
      RealViews.outputOffset c.M c.N h.parameters (p0 q) (p1 q) (p2 q) j) : p = q := by
  rw [output_offset, output_offset] at he
  obtain ⟨he, hcol⟩ := mul_add_inj hi.2.2 hj.2.2 he
  obtain ⟨he, hrow⟩ := mul_add_inj hi.2.1 hj.2.1 he
  obtain ⟨hb, hp⟩ := mul_add_inj (p2_lt p) (p2_lt q) he
  have hm := (mul_add_inj i.1.isLt j.1.isLt hrow).1
  have hn := (mul_add_inj i.2.1.isLt j.2.1.isLt hcol).1
  apply pid_ext _ hb hp
  have hd := Nat.div_add_mod (p0 p) (RealSetup.tilesN c.N h.parameters)
  have hd' := Nat.div_add_mod (p0 q) (RealSetup.tilesN c.N h.parameters)
  rw [hm, hn] at hd
  exact hd.symm.trans hd'

theorem output_bound (h : Call) (c : Config) (p : PID h c)
    (i : TileIndex [c.M, c.N])
    (hi : (RealViews.io Bool.false c.M c.N c.K h.parameters).outputMask (p0 p) (p1 p) (p2 p) i) :
    RealViews.outputOffset c.M c.N h.parameters (p0 p) (p1 p) (p2 p) i < h.shape.outputSize := by
  rw [output_offset]
  have hbase : p1 p * (h.shape.chunks * h.shape.groups) + p2 p <
      h.shape.batch * (h.shape.chunks * h.shape.groups) :=
    mul_add_bound (p1_lt p) (p2_lt p)
  have hrow := mul_add_bound hbase hi.2.1
  exact mul_add_bound hrow hi.2.2

theorem cdiv_covers {n b : Nat} (hb : 0 < b) : n ≤ cdiv n b * b := by
  have hd := Nat.div_add_mod (n + b - 1) b
  have hm := Nat.mod_lt (n + b - 1) hb
  unfold cdiv
  rw [Nat.mul_comm]
  omega

theorem quotient_lt_cdiv {r n b : Nat} (hb : 0 < b) (hr : r < n) : r / b < cdiv n b :=
  (Nat.div_lt_iff_lt_mul hb).2 (hr.trans_le (cdiv_covers hb))

def makePID (h : Call) (c : Config) (x y z : Nat)
    (hx : x < cdiv h.shape.chunk c.M * cdiv h.shape.chunk c.N)
    (hy : y < h.shape.batch) (hz : z < h.shape.chunks * h.shape.groups) : PID h c := by
  intro ax
  rcases ax with ⟨ax, ha⟩
  have ha' : ax < 3 := ha
  interval_cases ax
  · exact ⟨x, hx⟩
  · exact ⟨y, hy⟩
  · exact ⟨z, hz⟩

/-- Every returned logical cell is owned by a grid tile. Its causal activity is
the source's tile predicate, which is intentionally coarser than row ≥ col. -/
theorem coverage (h : Call) (c : Config) (hc : c ∈ configurations)
    (b ch g r n : Nat) (hb : b < h.shape.batch) (hch : ch < h.shape.chunks)
    (hg : g < h.shape.groups) (hr : r < h.shape.chunk) (hn : n < h.shape.chunk) :
    ∃ p : PID h c, ∃ i : TileIndex [c.M,c.N],
      p1 p = b ∧ p2 p = ch * h.shape.groups + g ∧
      RealViews.row c.M c.N h.parameters (p0 p) i.1 = r ∧
      RealViews.column c.N h.parameters (p0 p) i.2.1 = n := by
  have hm := (configuration_positive c hc).1
  have hn' := (configuration_positive c hc).2.1
  have hrq := quotient_lt_cdiv hm hr
  have hnq := quotient_lt_cdiv hn' hn
  have htiles : 0 < cdiv h.shape.chunk c.N := Nat.zero_lt_of_lt hnq
  let p := makePID h c (r / c.M * cdiv h.shape.chunk c.N + n / c.N)
    b (ch * h.shape.groups + g) (mul_add_bound hrq hnq) hb (mul_add_bound hch hg)
  refine ⟨p, (⟨r % c.M, Nat.mod_lt _ hm⟩, ⟨n % c.N, Nat.mod_lt _ hn'⟩, PUnit.unit), rfl, rfl, ?_, ?_⟩
  · change ((r / c.M * cdiv h.shape.chunk c.N + n / c.N) / cdiv h.shape.chunk c.N) * c.M + r % c.M = r
    rw [Nat.mul_comm (r / c.M), Nat.mul_add_div htiles]
    simp only [Nat.div_eq_of_lt hnq, Nat.add_zero]
    exact Nat.div_add_mod' r c.M
  · change ((r / c.M * cdiv h.shape.chunk c.N + n / c.N) % cdiv h.shape.chunk c.N) * c.N + n % c.N = n
    simp only [Nat.add_mod, Nat.mul_mod_left, Nat.zero_add, Nat.mod_eq_of_lt hnq]
    exact Nat.div_add_mod' n c.N

end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Host

#guard_msgs (drop info) in
#auditModuleAxioms
