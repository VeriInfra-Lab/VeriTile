import bench.repairs.tritonbench_g.rotary_emb.Proofs.Memory

namespace VeriTile.Bench.Repairs.RotaryEmb

abbrev Program := Nat × Nat
abbrev Lane := Bool × (Nat × (Nat × Nat))
abbrev LaunchLane := Program × Lane

def programs (s : Shape) : List Program :=
  (List.range s.headBlocks) ×ˢ (List.range s.seqBlocks)

/-- Source branch order Q then K; within a branch, sequence/head/pair lanes.
The vector-to-scalar correspondence is proved separately. -/
def lanes (s : Shape) : List Lane :=
  [false,true] ×ˢ ((List.range 16) ×ˢ ((List.range 4) ×ˢ (List.range s.pairBlock)))

def launches (s : Shape) : List LaunchLane := (programs s) ×ˢ (lanes s)

def workOf (l : LaunchLane) : Work :=
  ⟨l.2.1,16*l.1.2+l.2.2.1,4*l.1.1+l.2.2.2.1,l.2.2.2.2⟩

def schedule (s : Shape) : List Work :=
  ((launches s).map workOf).filter (fun w => decide (s.Active w))

theorem launches_nodup (s : Shape) : (launches s).Nodup := by
  apply List.Nodup.product
  · exact (List.nodup_range).product (List.nodup_range)
  · exact (by decide : ([false,true] : List Bool).Nodup).product
      ((List.nodup_range).product ((List.nodup_range).product (List.nodup_range)))

theorem workOf_injective_on (s : Shape) (a b : LaunchLane)
    (ha : a∈launches s) (hb : b∈launches s) (eq : workOf a=workOf b) : a=b := by
  rcases a with ⟨⟨ph,ps⟩,key,it,ih,ip⟩
  rcases b with ⟨⟨ph',ps'⟩,key',it',ih',ip'⟩
  simp only [launches,programs,lanes,List.mem_product,List.mem_range,List.mem_cons,
    List.not_mem_nil,or_false] at ha hb
  simp only [workOf,Work.mk.injEq] at eq
  have hp : ph=ph' := by omega
  have hs : ps=ps' := by omega
  have ht : it=it' := by omega
  have hh : ih=ih' := by omega
  simp_all

theorem schedule_nodup (s : Shape) : (schedule s).Nodup := by
  exact ((launches_nodup s).map_on (fun a ha b hb eq => workOf_injective_on s a b ha hb eq)).filter _

theorem schedule_active (s : Shape) (w : Work) (h : w∈schedule s) : s.Active w := by
  exact of_decide_eq_true (List.mem_filter.mp h).2

theorem schedule_covers (s : Shape) (w : Work) (h : s.Active w) : w∈schedule s := by
  apply List.mem_filter.mpr
  refine ⟨?_,by simpa using h⟩
  apply List.mem_map.mpr
  refine ⟨((w.head/4,w.token/16),(w.key,w.token%16,w.head%4,w.pair)),?_,?_⟩
  · have hh := heads_covered s w.key w.head h.2.1
    have ht := h.1
    have hp := lt_of_lt_of_le h.2.2 (pairs_covered s)
    have hk : w.key=false ∨ w.key=true := by cases w.key <;> simp
    simp only [launches,programs,lanes,List.mem_product,List.mem_range,List.mem_cons,
      List.not_mem_nil,or_false]
    unfold Shape.seqBlocks
    exact ⟨⟨by omega,by omega⟩,hk,Nat.mod_lt _ (by decide),Nat.mod_lt _ (by decide),hp⟩
  · cases w
    simp [workOf,Work.mk.injEq]
    omega

theorem schedule_mem_iff (s : Shape) (w : Work) : w∈schedule s ↔ s.Active w :=
  ⟨schedule_active s w,schedule_covers s w⟩

def run {α : Type} (s : Shape) (p : Ops α) (m : Memory α) : Memory α :=
  execute s p (schedule s) m

theorem run_readback {α : Type} (s : Shape) (valid : s.Valid) (p : Ops α)
    (m : Memory α) (w : Work) (h : s.Active w) (odd : Bool) :
    run s p m (dataRegion w.key) (s.cell w odd)=component (pairValues s p w m) odd :=
  execute_readback s valid p _ m (schedule_active s) (schedule_nodup s)
    w (schedule_covers s w h) odd

theorem run_frame {α : Type} (s : Shape) (p : Ops α) (m : Memory α)
    (r : Region) (a : Nat) (outside : ∀ w,s.Active w → ¬Writes s w r a) :
    run s p m r a=m r a :=
  execute_frame s p _ m r a (fun w hw => outside w (schedule_active s w hw))

theorem tail_outside (s : Shape) (valid : s.Valid) (key : Bool) (n h d : Nat)
    (hh : h<s.heads key) (hd : d<s.width) (tail : s.rotary≤d)
    (w : Work) (active : s.Active w) :
    ¬Writes s w (dataRegion key) (s.address key n h d) := by
  rintro ⟨hr,ha|ha⟩
  all_goals
    have hk := dataRegion_injective hr
    have hwh : w.head<s.heads key := by simpa [hk] using active.2.1
  · have bound := pair_dim_bound s valid w active false
    simp only [Bool.false_eq_true,↓reduceIte,Nat.add_zero] at bound
    have equal := address_injective s key n h d w.token w.head (2*w.pair) hh hwh hd
      (lt_of_lt_of_le bound valid.rotary_le) (by simpa [Shape.cell,←hk] using ha)
    omega
  · have bound := pair_dim_bound s valid w active true
    simp only [↓reduceIte] at bound
    have equal := address_injective s key n h d w.token w.head (2*w.pair+1) hh hwh hd
      (lt_of_lt_of_le bound valid.rotary_le) (by simpa [Shape.cell,←hk] using ha)
    omega

theorem run_tail {α : Type} (s : Shape) (valid : s.Valid) (p : Ops α) (m : Memory α)
    (key : Bool) (n h d : Nat) (hh : h<s.heads key) (hd : d<s.width) (tail : s.rotary≤d) :
    run s p m (dataRegion key) (s.address key n h d)=m (dataRegion key) (s.address key n h d) :=
  run_frame s p m _ _ (tail_outside s valid key n h d hh hd tail)

theorem run_coefficients {α : Type} (s : Shape) (p : Ops α) (m : Memory α)
    (sine : Bool) (a : Nat) : run s p m (if sine then .sin else .cos) a=m (if sine then .sin else .cos) a := by
  apply run_frame
  intro w _
  cases sine <;> cases hk : w.key <;> simp [Writes,dataRegion,hk]

#axiomsClean workOf_injective_on
#axiomsClean schedule_covers
#axiomsClean run_readback
#axiomsClean run_tail
#axiomsClean run_coefficients
end VeriTile.Bench.Repairs.RotaryEmb
