import bench.tritonbench_g.context_attn_fwd.Proofs.HostLaunch

namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1000000

def index (a : Arguments) (b : Fin (Bounds.ceilBlocks a.maxInput (block a)))
    (bh : Fin (a.batch*a.q.heads)) : GridIndex (grid a) := fun ax => by
  rcases ax with ⟨ax, hax⟩
  have hh : ax < 3 := by simpa [grid, Grid.rank] using hax
  interval_cases ax
  · exact b
  · exact bh
  · exact ⟨0, by change 0 < 1; omega⟩

@[simp] theorem index_launch (a : Arguments)
    (b : Fin (Bounds.ceilBlocks a.maxInput (block a))) (bh : Fin (a.batch*a.q.heads)) :
    launchOf a (index a b bh) = ⟨b.val, bh.val⟩ := rfl

/-- Every requested logical output has an actual program and active lane.
No complete-grid coverage hypothesis is assumed. -/
theorem covers (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (ha : Admissible a r A s) (b h row d : Nat) (hb : b < a.batch)
    (hh : h < a.q.heads) (hr : (row : Int) < (inputs s r).sequence b-(inputs s r).prompt b)
    (hd : d < a.k.dims) :
    ∃ idx : GridIndex (grid a), ∃ i : RealIO.Lane (layout a),
      active a r s idx i ∧ batch (layout a) (launchOf a idx) = b ∧
      head (layout a) (launchOf a idx) = h ∧
      queryRow (layout a) (launchOf a idx) i.1.val = row ∧ i.2.1.val = d := by
  have hm := (ha.metadata b hb).2.2.2.1
  have hmax : row < a.maxInput := by omega
  have htile : row / block a < Bounds.ceilBlocks a.maxInput (block a) := by
    apply (Bounds.lt_ceilBlocks _ _ _ (block_positive a)).mpr
    exact lt_of_le_of_lt (Nat.div_mul_le_self _ _) hmax
  have hbh : b*a.q.heads+h < a.batch*a.q.heads := by
    have hmul := Nat.mul_le_mul_right a.q.heads (Nat.succ_le_of_lt hb)
    nlinarith only [hmul, hh]
  let idx := index a ⟨row/block a,htile⟩ ⟨b*a.q.heads+h,hbh⟩
  let i : RealIO.Lane (layout a) := (⟨row % block a, Nat.mod_lt _ (block_positive a)⟩,⟨d,hd⟩,())
  have ib : batch (layout a) (launchOf a idx) = b := by
    change (b*a.q.heads+h)/a.q.heads = b
    rw [Nat.add_comm, Nat.mul_comm b, Nat.add_mul_div_left _ _ ha.heads_positive, Nat.div_eq_of_lt hh]
    omega
  have ih : head (layout a) (launchOf a idx) = h := by
    change (b*a.q.heads+h)%a.q.heads = h
    simp [Nat.add_mod, Nat.mod_eq_of_lt hh]
  have ir : queryRow (layout a) (launchOf a idx) i.1.val = row := by
    change row / block a * block a + row % block a = row
    simpa [Nat.mul_comm] using Nat.div_add_mod row (block a)
  refine ⟨idx, i, ?_, ib, ih, ir, rfl⟩
  change (queryRow (layout a) (launchOf a idx) i.1.val : Int) <
    (inputs s r).sequence (batch (layout a) (launchOf a idx)) -
      (inputs s r).prompt (batch (layout a) (launchOf a idx))
  rw [ib,ir]
  exact hr

/-- Every input allocation is stable in the merged launch. This uses actual
finite allocation separation, not an inference from per-program alias safety. -/
theorem input_unwritten (a : Arguments) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (ha : Admissible a r A s) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (rg : RegionName) (hne : r.out ≠ rg) (off : Nat) (hb : off < A.extent rg) :
    ¬writes a r A s (A.flat,A.addr rg off) := by
  rintro ⟨idx, i, hi, he⟩
  have ho : RealIO.write (layout a) (launchOf a idx) (inputs s r) i < A.extent r.out :=
    AddressBounds.inside_toNat ((ha.bounds idx).output i.1 i.2.1 hi)
  have hr : rg ∈ A.regions := by
    by_contra hn; rw [hcov rg hn] at hb; omega
  have hout : r.out ∈ A.regions := by
    by_contra hn; rw [hcov r.out hn] at ho; omega
  have eqaddr := congrArg Prod.snd he
  have decode := congrArg A.decode eqaddr
  change A.decode (A.addr r.out (RealIO.write (layout a) (launchOf a idx) (inputs s r) i)) = A.decode (A.addr rg off) at decode
  rw [A.decode_addr hd hout ho, A.decode_addr hd hr hb] at decode
  have hp := Option.some.inj decode
  exact hne (congrArg Prod.fst hp)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host
