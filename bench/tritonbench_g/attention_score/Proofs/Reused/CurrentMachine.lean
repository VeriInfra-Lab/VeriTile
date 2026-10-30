import bench.tritonbench_g.attention_score.Proofs.Reused.CurrentRepresentation

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.Machine
open VeriTile Triton Geometry Physical Representation
noncomputable section
set_option maxHeartbeats 1000000

def pointers (c : Shape) (z h row key d : Nat) : List (RegionName × Nat) :=
  [("q",c.queryBase z h+row*c.D+d),
   ("k",c.keyBase z h+key*c.D+d),
   ("m",c.normalizerBase z h+row),
   ("out",c.outputBase z h+key)]

/-- Any nonnegative source subexpression of a formed displacement is bounded
by that displacement. This includes head*stride, row*D, and partial additions,
and avoids a premise assuming fixed-width address correctness. -/
def PointerExact (c : Shape) (abi : ABI c) (p : RegionName × Nat) : Prop :=
  ∀ sub, sub ≤ p.2 →
    Signed64 (sub : Int) ∧ Signed64 (4*(sub : Int)) ∧
    (BitVec.ofInt 64 (sub : Int)).toInt=(sub : Int) ∧
    (BitVec.ofInt 64 (4*(sub : Int))).toInt=4*(sub : Int) ∧
    4*((abi.alloc p.1).base+sub)<18446744073709551616

theorem pointer_exact (c : Shape) (abi : ABI c) (hc : Launch c)
    (ha : ∀ r, tracked r → 4*((abi.alloc r).base+extent c r)<18446744073709551616)
    (p : RegionName × Nat) (hr : tracked p.1) (hp : p.2<extent c p.1) : PointerExact c abi p := by
  intro sub hs
  obtain ⟨he,hb,ha⟩ := formed_address c abi hc ha p.1 hr sub (lt_of_le_of_lt hs hp)
  exact ⟨he,hb,ofInt64_exact _ he,ofInt64_exact _ hb,ha⟩

theorem source_pointers_exact (c : Shape) (abi : ABI c) (hc : Launch c)
    (ha : ∀ r, tracked r → 4*((abi.alloc r).base+extent c r)<18446744073709551616)
    (z : Fin c.Z) (h : Fin c.H) (row key : Nat)
    (hr : row<qPad c+c.B) (hk : key<kPad c) (d : Fin c.D) :
    ∀ p ∈ pointers c z.val h.val row key d.val, PointerExact c abi p := by
  obtain ⟨hq,hkey,hm,ho⟩ := source_addresses c z h row hr key hk d
  intro p hp
  simp only [pointers,List.mem_cons,List.not_mem_nil,or_false] at hp
  rcases hp with rfl|rfl|rfl|rfl
  · exact pointer_exact c abi hc ha _ (by simp [tracked]) hq
  · exact pointer_exact c abi hc ha _ (by simp [tracked]) hkey
  · exact pointer_exact c abi hc ha _ (by simp [tracked]) hm
  · exact pointer_exact c abi hc ha _ (by simp [tracked]) ho

def window64 (offset size : Int) (start base i j : Nat) (complement : Bool) : Bool :=
  if complement then decide ((BitVec.ofInt 64 size).toInt≤(distance64 i j start base offset).toInt)
  else decide (0≤(distance64 i j start base offset).toInt ∧
    (distance64 i j start base offset).toInt<(BitVec.ofInt 64 size).toInt)

/-- Covers all jobs, padded lanes and the final post-loop Q/M pointer state.
The scalar and block-coordinate results additionally justify signed32 metadata
and block-pointer narrowing; window arithmetic is source-ordered signed64. -/
def Closed (c : Shape) (abi : ABI c) (offset size : Int) (window : Bool) : Prop :=
  c.N≠0 → c.outputSize≠0 →
  (∀ x∈metadata c, (BitVec.ofInt 32 (x : Int)).toInt=(x : Int)) ∧
  (∀ (z : Fin c.Z) (h : Fin c.H) (t : Nat), t≤RealLoop.tripCount c.N c.B →
    ∀ (block : Fin c.programs) (i j : Fin c.B) (d : Fin c.D),
      Signed32 (t*c.B+i.val : Nat) ∧ Signed32 (block.val*c.B+j.val : Nat) ∧
      Signed32 (t*c.B : Nat) ∧ Signed32 (block.val*c.B : Nat) ∧
      (∀ p∈pointers c z.val h.val (t*c.B+i.val) (block.val*c.B+j.val) d.val,
        PointerExact c abi p) ∧
      (window=Bool.true → ∀ complement,
        window64 offset size (t*c.B) (block.val*c.B) i.val j.val complement=
          decide (RealFormula.windowAllows Bool.true complement offset size
            (t*c.B+i.val) (block.val*c.B+j.val))))

theorem closed (c : Shape) (abi : ABI c) (offset size : Int) (window : Bool)
    (domain : Domain c abi offset size window) : Closed c abi offset size window := by
  intro hn ho
  have hc := domain.launch hn ho
  refine ⟨metadata_exact c hc,?_⟩
  intro z h t ht block i j d
  obtain ⟨hrow,hkey,hstart,hbase⟩ := padded_coordinates c hc t ht i block j
  have hs : t*c.B≤qPad c := Nat.mul_le_mul_right c.B ht
  have hr : t*c.B+i.val<qPad c+c.B := by omega
  have hk : block.val*c.B+j.val<kPad c := digit_bound _ _ _ _ block.isLt j.isLt
  refine ⟨natural32 _ (Nat.le_of_lt hrow),natural32 _ (Nat.le_of_lt hkey),
    natural32 _ hstart,natural32 _ hbase,
    source_pointers_exact c abi hc (domain.addresses hn ho) z h _ _ hr hk d,?_⟩
  intro hw complement
  exact window_equivalence c hc offset size (domain.scalars hw).1 (domain.scalars hw).2
    _ hs block i j complement

#axiomsClean source_pointers_exact
#axiomsClean closed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.Machine
