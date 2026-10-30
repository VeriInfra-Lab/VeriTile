import VeriTile.Triton.Math.Attention
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.RealMath
open VeriTile.Triton
noncomputable section

/-- Finite valid-key blocks, with the final block truncated at logical N. -/
def blocks (xs : List α) : List (List α) :=
  if h : xs=[] then [] else xs.take 64 :: blocks (xs.drop 64)
termination_by xs.length
decreasing_by
  have hp : 0<xs.length := by cases xs <;> simp_all
  try simp only [List.length_drop]
  omega

theorem flatten_blocks (xs : List α) : (blocks xs).flatten=xs := by
  generalize hlen : xs.length=n
  induction n using Nat.strong_induction_on generalizing xs with
  | h n ih =>
    rw [blocks]
    split
    · rename_i he
      simp [he]
    · rename_i he
      have hp : 0<n := by cases xs <;> simp_all <;> omega
      have hl : (xs.drop 64).length<n := by
        rw [List.length_drop,hlen]
        omega
      rw [List.flatten_cons,ih _ hl _ rfl]
      exact List.take_append_drop 64 xs

/-- At the first nonempty source block, old m=-inf makes alpha=0, so the
source's l=1 seed disappears. All later steps are ordinary block updates. -/
def first (block : List (ℝ×ℝ)) : ℝ×ℝ×ℝ :=
  let m:=blockMax (block.headD (0,0)).1 block
  (m,(block.map (fun p => pow2 (p.1-m))).sum,
    (block.map (fun p => pow2 (p.1-m)*p.2)).sum)

def online (bs : List (List (ℝ×ℝ))) : ℝ×ℝ×ℝ :=
  match bs with
  | [] => (0,1,0)
  | b::rest => rest.foldl osBlockStep (first b)

theorem first_consistent (b : List (ℝ×ℝ)) :
    (first b).2.1=pow2 (-(first b).1)*(b.map (fun p => pow2 p.1)).sum ∧
    (first b).2.2=pow2 (-(first b).1)*(b.map (fun p => pow2 p.1*p.2)).sum := by
  constructor
  · simpa [first] using sum_map_pow2_sub (blockMax (b.headD (0,0)).1 b) b (fun _ => 1)
  · exact sum_map_pow2_sub _ b (fun p => p.2)

theorem online_closed_form (bs : List (List (ℝ×ℝ))) :
    (online bs).2.2/(online bs).2.1=
      (bs.flatten.map (fun p => pow2 p.1*p.2)).sum /
        (bs.flatten.map (fun p => pow2 p.1)).sum := by
  cases bs with
  | nil => simp [online]
  | cons b rest =>
    obtain ⟨hl,ha⟩:=osBlockStep_foldl_consistent rest (first b).1 (first b).2.1 (first b).2.2
      (b.map (fun p => pow2 p.1*p.2)).sum (b.map (fun p => pow2 p.1)).sum
      (first_consistent b).1 (first_consistent b).2
    try dsimp only at hl ha
    simp only [online,ha,hl,List.flatten_cons,List.map_append,List.sum_append]
    exact mul_div_mul_left _ _ (ne_of_gt (pow2_pos _))

/-- Independent length-N formula, using the initial logical Q/K/V cells and
per-query-block/per-key-block scales. No loop or observed output appears. -/
def score (q k : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m j : Nat) : ℝ :=
  ((Finset.range 96).sum (fun d => q m d*k j d))*qs (m/128)*ks (j/64)

def expected (N : Nat) (q k v : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m d : Nat) : ℝ :=
  ((List.range N).map (fun j => pow2 (score q k qs ks m j)*v j d)).sum /
    ((List.range N).map (fun j => pow2 (score q k qs ks m j))).sum

def inputs (N : Nat) (q k v : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m d : Nat) : List (ℝ×ℝ) :=
  (List.range N).map (fun j => (score q k qs ks m j,v j d))

theorem blocked_eq_expected (N : Nat) (q k v : Nat → Nat → ℝ) (qs ks : Nat → ℝ) (m d : Nat) :
    let st:=online (blocks (inputs N q k v qs ks m d))
    st.2.2/st.2.1=expected N q k v qs ks m d := by
  dsimp only
  rw [online_closed_form,flatten_blocks]
  simp [expected,inputs,List.map_map,Function.comp_def]

theorem pad_head (f : Nat → ℝ) :
    (Finset.range 128).sum (fun d => if d<96 then f d else 0)=(Finset.range 96).sum f := by
  have hr : 128=96+32 := rfl
  rw [hr,Finset.sum_range_add]
  have hfirst : (∑ d∈Finset.range 96, if d<96 then f d else 0)=∑ d∈Finset.range 96, f d := by
    apply Finset.sum_congr rfl
    intro d hd
    simp [Finset.mem_range.mp hd]
  have hlast : (∑ d∈Finset.range 32, if 96+d<96 then f (96+d) else 0)=0 := by simp
  simpa [hfirst,hlast]

#specNonCircular expected avoiding [online,osBlockStep]
#axiomsClean blocked_eq_expected
#axiomsClean pad_head
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.RealMath
