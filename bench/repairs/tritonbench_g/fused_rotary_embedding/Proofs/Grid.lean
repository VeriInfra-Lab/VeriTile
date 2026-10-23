import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.Memory

namespace VeriTile.Bench.Repairs.FusedRotaryEmbedding
abbrev Program := Nat × Nat
abbrev Lane := Kind × Nat
abbrev LaunchLane := Program × Lane
def programs (s : Shape) : List Program := (List.range s.headsQ) ×ˢ (List.range s.tokens)
def lanes (s : Shape) : List Lane := [Kind.q,.k,.v] ×ˢ (List.range s.halfBlock)
def launches (s : Shape) : List LaunchLane := programs s ×ˢ lanes s
def leader (s : Shape) (l : LaunchLane) : Prop := l.2.1=.q ∨ l.1.1%s.groups=0
instance (s : Shape) (l : LaunchLane) : Decidable (leader s l) := inferInstanceAs (Decidable (_∨_))
def workOf (s : Shape) (l : LaunchLane) : Work :=
  ⟨l.2.1,l.1.2,if l.2.1=.q then l.1.1 else l.1.1/s.groups,l.2.2⟩
def selectedLanes (s : Shape) := (launches s).filter (fun l => decide (leader s l))
def schedule (s : Shape) := ((selectedLanes s).map (workOf s)).filter (fun w => decide (s.Active w))

theorem groups_mul (s : Shape) (hs : s.Valid) : s.groups*s.headsK=s.headsQ :=
  Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hs.grouping)
theorem groups_positive (s : Shape) (hs : s.Valid) : 0<s.groups := by
  have := groups_mul s hs
  have := hs.hq
  nlinarith
theorem half_covered (s : Shape) : s.half≤s.halfBlock := Nat.le_pow_clog (by decide) _
theorem full_covered (s : Shape) : s.width≤s.fullBlock := Nat.le_pow_clog (by decide) _

theorem launches_nodup (s : Shape) : (launches s).Nodup :=
  ((List.nodup_range).product (List.nodup_range)).product
    ((by decide : ([Kind.q,.k,.v] : List Kind).Nodup).product (List.nodup_range))

theorem workOf_injective_on (s : Shape) (a b : LaunchLane)
    (ha : a∈selectedLanes s) (hb : b∈selectedLanes s) (eq : workOf s a=workOf s b) : a=b := by
  rcases a with ⟨⟨h,t⟩,kind,i⟩
  rcases b with ⟨⟨h',t'⟩,kind',i'⟩
  have la : kind=.q ∨ h%s.groups=0 := by simpa [selectedLanes,leader] using (List.mem_filter.mp ha).2
  have lb : kind'=.q ∨ h'%s.groups=0 := by simpa [selectedLanes,leader] using (List.mem_filter.mp hb).2
  simp only [workOf,Work.mk.injEq] at eq
  rcases eq with ⟨rfl,rfl,he,rfl⟩
  have hh : h=h' := by
    by_cases kq : kind=.q
    · simpa [kq] using he
    · have ma : h%s.groups=0 := la.resolve_left kq
      have mb : h'%s.groups=0 := lb.resolve_left kq
      have qa : h/s.groups*s.groups=h := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero ma)
      have qb : h'/s.groups*s.groups=h' := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero mb)
      simp only [kq,ite_false] at he
      rw [←qa,←qb,he]
  simp [hh]

theorem schedule_nodup (s : Shape) : (schedule s).Nodup :=
  (((launches_nodup s).filter _).map_on (fun a ha b hb eq => workOf_injective_on s a b ha hb eq)).filter _
theorem schedule_active (s : Shape) (w : Work) (h : w∈schedule s) : s.Active w :=
  of_decide_eq_true (List.mem_filter.mp h).2

theorem schedule_covers (s : Shape) (hs : s.Valid) (w : Work) (hw : s.Active w) : w∈schedule s := by
  apply List.mem_filter.mpr
  refine ⟨?_,by simpa using hw⟩
  apply List.mem_map.mpr
  let ph := if w.kind=.q then w.head else w.head*s.groups
  refine ⟨((ph,w.token),(w.kind,w.pair)),?_,?_⟩
  · apply List.mem_filter.mpr
    constructor
    · have kind : w.kind∈([Kind.q,.k,.v] : List Kind) := by cases w.kind <;> simp
      have heads : ph<s.headsQ := by
        by_cases kq : w.kind=.q
        · simpa [ph,Shape.heads,kq] using hw.2.1
        · have wh : w.head<s.headsK := by simpa [Shape.heads,kq] using hw.2.1
          have hm := Nat.mul_lt_mul_of_pos_right wh (groups_positive s hs)
          rw [Nat.mul_comm s.headsK s.groups,groups_mul s hs] at hm
          simpa [ph,kq] using hm
      simpa only [launches,programs,lanes,List.mem_product,List.mem_range] using
        And.intro (And.intro heads hw.1) (And.intro kind (lt_of_lt_of_le hw.2.2 (half_covered s)))
    · by_cases kq : w.kind=.q <;> simp [leader,ph,kq]
  · have gp := groups_positive s hs
    by_cases kq : w.kind=.q
    · cases w;simp_all [workOf,ph]
    · cases w;simp_all [workOf,ph,Nat.mul_div_cancel,ne_of_gt gp]

theorem schedule_mem_iff (s : Shape) (hs : s.Valid) (w : Work) : w∈schedule s ↔ s.Active w :=
  ⟨schedule_active s w,schedule_covers s hs w⟩
def run {α : Type} (c : Config) (p : Kind → Ops α) (m : Memory α) := execute c p (schedule c.shape) m
theorem run_readback {α : Type} (c : Config) (hc : c.Valid) (p : Kind → Ops α) (m : Memory α)
    (w : Work) (hw : c.shape.Active w) (o : Bool) :
    run c p m w.kind.output (c.output w o)=component (pairValues c.shape p w m) o :=
  execute_readback c hc p _ m (schedule_active c.shape) (schedule_nodup c.shape)
    w (schedule_covers c.shape hc.shape w hw) o
theorem run_frame {α : Type} (c : Config) (p : Kind → Ops α) (m : Memory α) (r : Region) (a : Nat)
    (outside : ∀ w,c.shape.Active w → ¬Writes c w r a) : run c p m r a=m r a :=
  execute_frame c p _ m r a (fun w hw => outside w (schedule_active c.shape w hw))

#axiomsClean schedule_covers
#axiomsClean schedule_nodup
#axiomsClean run_readback
end VeriTile.Bench.Repairs.FusedRotaryEmbedding
