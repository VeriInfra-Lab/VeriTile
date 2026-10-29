import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Launch

/-! The sorted complete stream visits chunks in Python's batch/head/block
nesting. Equal chunks form one contiguous interval, so Launch.run calls setup
once per source native call, before that chunk's first program. Local PID order
within a native call is one allowed order of its disjoint programs. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.LaunchOrder
open VeriTile.Triton HostPlan
noncomputable section

def radix (c : Call) : Nat := c.batches+c.q.heads+c.blocks+2147483648
def rank (c : Call) (j : Job c) : Nat :=
  (j.1.1.val.val*radix c+j.2.1.1.val.val)*radix c+j.2.2.1.val.val

theorem digit_bounds (c : Call) (j : Job c) :
    0 < radix c ∧ j.1.1.val.val < radix c ∧ j.2.1.1.val.val < radix c ∧
    j.2.2.1.val.val < radix c ∧ j.1.2.val < radix c ∧
    j.2.1.2.val < radix c ∧ j.2.2.2.val < radix c := by
  have hb := j.1.1.val.isLt
  have hh := j.2.1.1.val.isLt
  have hq := j.2.2.1.val.isLt
  have bl := (lt_min_iff.mp j.1.2.isLt).1
  have hl := (lt_min_iff.mp j.2.1.2.isLt).1
  have ql := (lt_min_iff.mp j.2.2.2.isLt).1
  dsimp [radix]
  omega

theorem append_div (a r b : Nat) (hr : 0 < r) (hb : b < r) :
    (a*r+b)/r = a := by
  rw [Nat.add_comm,Nat.add_mul_div_right _ _ hr,Nat.div_eq_of_lt hb,Nat.zero_add]

theorem key_div (c : Call) (j : Job c) :
    orderKey c j / radix c / radix c / radix c = rank c j := by
  obtain ⟨hr,_,_,_,hb,hh,hq⟩ := digit_bounds c j
  dsimp only [orderKey]
  change (((rank c j*radix c+j.1.2.val)*radix c+j.2.1.2.val)*radix c+j.2.2.2.val) /
    radix c / radix c / radix c = rank c j
  rw [append_div _ _ _ hr hq,append_div _ _ _ hr hh,append_div _ _ _ hr hb]

theorem rank_mono {c : Call} {a b : Job c} (h : orderKey c a ≤ orderKey c b) :
    rank c a ≤ rank c b := by
  rw [← key_div c a,← key_div c b]
  exact Nat.div_le_div_right (Nat.div_le_div_right (Nat.div_le_div_right h))

theorem rank_eq_iff {c : Call} (a b : Job c) : rank c a = rank c b ↔ Launch.chunk a = Launch.chunk b := by
  obtain ⟨hr,_,ha,hab,_,_,_⟩ := digit_bounds c a
  obtain ⟨_,_,hb,hbb,_,_,_⟩ := digit_bounds c b
  constructor
  · intro he
    have hblock := congrArg (fun x => x % radix c) he
    simp only [rank,Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hab,Nat.mod_eq_of_lt hbb] at hblock
    have hhead := congrArg (fun x => x / radix c) he
    simp only [rank,append_div _ _ _ hr hab,append_div _ _ _ hr hbb] at hhead
    have hh := congrArg (fun x => x % radix c) hhead
    simp only [Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt ha,Nat.mod_eq_of_lt hb] at hh
    have hbatch := congrArg (fun x => x / radix c) hhead
    simp only [append_div _ _ _ hr ha,append_div _ _ _ hr hb] at hbatch
    exact Prod.ext hbatch (Prod.ext hh hblock)
  · intro he
    have he' : a.1.1.val.val = b.1.1.val.val ∧ a.2.1.1.val.val = b.2.1.1.val.val ∧
        a.2.2.1.val.val = b.2.2.1.val.val := by simpa only [Launch.chunk,Prod.mk.injEq] using he
    simp only [rank,he'.1,he'.2.1,he'.2.2]

/-- No different chunk can occur between two programs of one chunk. Together
with jobs_complete this proves one contiguous complete native grid per setup. -/
theorem chunk_contiguous {c : Call} (a b d : Job c)
    (hab : orderKey c a ≤ orderKey c b) (hbd : orderKey c b ≤ orderKey c d)
    (same : Launch.chunk a = Launch.chunk d) : Launch.chunk a = Launch.chunk b := by
  apply (rank_eq_iff a b).mp
  have he := (rank_eq_iff a d).mpr same
  have h1 := rank_mono hab
  have h2 := rank_mono hbd
  omega

theorem jobs_sorted (c : Call) : (jobs c).Pairwise (fun a b => orderKey c a ≤ orderKey c b) := by
  letI : Std.Total (fun a b : Job c => orderKey c a ≤ orderKey c b) := ⟨fun a b => Nat.le_total _ _⟩
  letI : IsTrans (Job c) (fun a b => orderKey c a ≤ orderKey c b) := ⟨fun _ _ _ h1 h2 => Nat.le_trans h1 h2⟩
  exact List.pairwise_mergeSort' (fun a b : Job c => orderKey c a ≤ orderKey c b) _

/-- Monotone batch, then head, then block bases: exactly source loop nesting. -/
theorem source_order {c : Call} (a b : Job c) (h : orderKey c a ≤ orderKey c b) :
    a.1.1.val.val ≤ b.1.1.val.val ∧
    (a.1.1.val.val = b.1.1.val.val → a.2.1.1.val.val ≤ b.2.1.1.val.val) ∧
    (a.1.1.val.val = b.1.1.val.val → a.2.1.1.val.val = b.2.1.1.val.val →
      a.2.2.1.val.val ≤ b.2.2.1.val.val) := by
  obtain ⟨hr,_,ha,hab,_,_,_⟩ := digit_bounds c a
  obtain ⟨_,_,hb,hbb,_,_,_⟩ := digit_bounds c b
  have hm := rank_mono h
  have hhead := Nat.div_le_div_right (c := radix c) hm
  simp only [rank,append_div _ _ _ hr hab,append_div _ _ _ hr hbb] at hhead
  have hbatch := Nat.div_le_div_right (c := radix c) hhead
  simp only [append_div _ _ _ hr ha,append_div _ _ _ hr hb] at hbatch
  refine ⟨hbatch,?_,?_⟩
  · intro he
    rw [he] at hhead
    omega
  · intro he hh
    dsimp only [rank] at hm
    rw [he,hh] at hm
    omega

#print axioms chunk_contiguous
#print axioms jobs_sorted
#print axioms source_order
end
end VeriTile.Bench.Repairs.ContextAttnNopad.LaunchOrder
