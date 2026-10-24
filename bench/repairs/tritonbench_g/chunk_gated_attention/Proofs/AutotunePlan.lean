import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.LaunchPlan

/-! Actual autotune host plan uses BS=16 for its outer bases; each selected
configuration computes its own active size or launches one guarded dummy. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.AutotunePlan
open LaunchPlan

def tune (total : Nat) (p : Item) : Item := ⟨p.base,max 1 (min Grid.limit (total-p.base))⟩
def plans (maximum total : Nat) := (LaunchPlan.plans maximum).map (tune total)

theorem shrink (maximum total : Nat) (ht : total≤maximum) (p : Item) (hp : p∈LaunchPlan.plans maximum) :
    (tune total p).size≤p.size := by
  have hpos := (LaunchPlan.plans_valid maximum p hp).1
  have he := LaunchPlan.member_size maximum maximum 0 p hp
  simp only [tune]
  rw [he]
  apply max_le
  · omega
  · exact min_le_min_left _ (Nat.sub_le_sub_right ht _)

theorem valid (maximum total : Nat) : ∀ p∈plans maximum total, 0<p.size ∧ p.size≤Grid.limit := by
  intro p hp
  obtain ⟨q,hq,rfl⟩ := List.mem_map.mp hp
  constructor
  · have h := le_max_left 1 (min Grid.limit (total-q.base)); change 0<max _ _; omega
  · exact max_le (by norm_num [Grid.limit]) (min_le_left _ _)

theorem ordered_map (total : Nat) (ps : List Item)
    (hs : ∀p∈ps, (tune total p).size≤p.size)
    (ho : ps.Pairwise (fun p q => p.base+p.size≤q.base)) :
    (ps.map (tune total)).Pairwise (fun p q => p.base+p.size≤q.base) := by
  induction ps with
  | nil => simp
  | cons p ps ih =>
    obtain ⟨hb,ht⟩ := List.pairwise_cons.mp ho
    simp only [List.map_cons,List.pairwise_cons]
    constructor
    · intro q hq
      obtain ⟨q',hq',rfl⟩ := List.mem_map.mp hq
      exact le_trans (Nat.add_le_add_left (hs p List.mem_cons_self) p.base) (hb q' hq')
    · exact ih (fun q hq => hs q (List.mem_cons_of_mem _ hq)) ht

theorem ordered (maximum total : Nat) (ht : total≤maximum) :
    (plans maximum total).Pairwise (fun p q => p.base+p.size≤q.base) :=
  ordered_map total _ (shrink maximum total ht) (LaunchPlan.plans_ordered maximum)

theorem covers (maximum total n : Nat) (ht : total≤maximum) (hn : n<total) :
    ∃p, p∈plans maximum total ∧ p.base≤n ∧ n<p.base+p.size := by
  obtain ⟨p,hp,hb,he⟩ := LaunchPlan.plans_cover maximum n (lt_of_lt_of_le hn ht)
  refine ⟨tune total p,List.mem_map.mpr ⟨p,hp,rfl⟩,hb,?_⟩
  have hs := (LaunchPlan.plans_valid maximum p hp).2.1
  change n<p.base+max 1 (min Grid.limit (total-p.base))
  have hm := le_max_right 1 (min Grid.limit (total-p.base))
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGatedAttention.AutotunePlan
