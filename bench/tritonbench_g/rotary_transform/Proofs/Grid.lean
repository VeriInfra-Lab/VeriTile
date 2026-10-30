import bench.tritonbench_g.rotary_transform.Proofs.Memory
namespace VeriTile.Bench.Repairs.RotaryTransform
abbrev Program := Nat×Nat×Nat
abbrev Lane := Nat×Nat
abbrev LaunchLane := Program×Lane
def programs (c : Config) : List Program :=
  (List.range c.shape.rowBlocks) ×ˢ ((List.range c.shape.batches) ×ˢ (List.range c.shape.heads))
def lanes (c : Config) : List Lane := (List.range c.shape.blockM) ×ˢ (List.range (c.shape.blockK/2))
def launches (c : Config) : List LaunchLane := programs c ×ˢ lanes c
def workOf (c : Config) (l : LaunchLane) : Work := ⟨l.1.2.1,l.1.1*c.shape.blockM+l.2.1,l.1.2.2,l.2.2⟩
def schedule (c : Config) := ((launches c).map (workOf c)).filter (fun w => decide (c.Active w))

theorem blockM_cases (s : Shape) : s.blockM=4 ∨ s.blockM=8 := by
  unfold Shape.blockM;split_ifs <;> simp
theorem launches_nodup (c : Config) : (launches c).Nodup := by
  exact (List.nodup_range.product (List.nodup_range.product List.nodup_range)).product
    (List.nodup_range.product List.nodup_range)
theorem workOf_injective_on (c : Config) (a b : LaunchLane)
    (ha : a∈launches c) (hb : b∈launches c) (eq : workOf c a=workOf c b) : a=b := by
  rcases a with ⟨⟨pm,pb,ph⟩,r,j⟩
  rcases b with ⟨⟨pm',pb',ph'⟩,r',j'⟩
  simp only [launches,programs,lanes,List.mem_product,List.mem_range] at ha hb
  simp only [workOf,Work.mk.injEq] at eq
  rcases blockM_cases c.shape with bm|bm <;> rw [bm] at ha hb eq <;>
    have hp : pm=pm' := by omega
  all_goals have hr : r=r' := by omega
  all_goals simp_all
theorem schedule_nodup (c : Config) : (schedule c).Nodup :=
  ((launches_nodup c).map_on (fun a ha b hb h => workOf_injective_on c a b ha hb h)).filter _
theorem schedule_active (c : Config) (w : Work) (h : w∈schedule c) : c.Active w :=
  of_decide_eq_true (List.mem_filter.mp h).2
theorem schedule_covers (c : Config) (hc : c.Valid) (w : Work) (h : c.Active w) : w∈schedule c := by
  apply List.mem_filter.mpr
  refine ⟨?_,by simpa using h⟩
  apply List.mem_map.mpr
  refine ⟨((w.row/c.shape.blockM,w.batch,w.head),(w.row%c.shape.blockM,w.pair)),?_,?_⟩
  · have hm := lt_of_lt_of_le h.2.1 (hc.lengthBound w.batch h.1)
    have hp := lt_of_lt_of_le h.2.2.2 (blockK_covers c hc)
    have row : w.row/c.shape.blockM<c.shape.rowBlocks := by
      unfold Shape.rowBlocks
      rcases blockM_cases c.shape with bm|bm <;> rw [bm] <;> omega
    simp only [launches,programs,lanes,List.mem_product,List.mem_range]
    exact ⟨⟨row,h.1,h.2.2.1⟩,Nat.mod_lt _ (blockM_pos c.shape),hp⟩
  · cases w
    simp only [workOf,Work.mk.injEq,true_and,and_true]
    simpa only [Nat.mul_comm] using Nat.div_add_mod _ _
theorem schedule_mem_iff (c : Config) (hc : c.Valid) (w : Work) : w∈schedule c↔c.Active w :=
  ⟨schedule_active c w,schedule_covers c hc w⟩
def run {α : Type} (c : Config) (p : Ops α) (m : Memory α) := execute c p (schedule c) m
theorem run_readback {α : Type} (c : Config) (hc : c.Valid) (p : Ops α) (m : Memory α)
    (w : Work) (hw : c.Active w) (o : Bool) :
    run c p m c.shape.output (c.cell w o)=component (pairValues c p w m) o :=
  execute_readback c hc p _ m (schedule_active c) (schedule_nodup c) w (schedule_covers c hc w hw) o
theorem run_frame {α : Type} (c : Config) (p : Ops α) (m : Memory α) (r : Region) (a : Nat)
    (outside : ∀ w,c.Active w → ¬Writes c w r a) : run c p m r a=m r a :=
  execute_frame c p _ m r a (fun w hw => outside w (schedule_active c w hw))
theorem tail_outside (c : Config) (hc : c.Valid) (token head col : Nat)
    (hh : head<c.shape.heads) (hd : col<c.shape.width) (tail : 2*c.shape.half≤col)
    (w : Work) (hw : c.Active w) : ¬Writes c w c.shape.output (c.shape.address token head col) := by
  rintro ⟨_,eq|eq⟩
  · have bound := dim_bound c.shape w.pair hw.2.2.2 false
    have e := address_injective c.shape _ _ _ _ _ _ hh hw.2.2.1 hd (lt_of_lt_of_le bound hc.rotary) eq
    omega
  · have bound := dim_bound c.shape w.pair hw.2.2.2 true
    have e := address_injective c.shape _ _ _ _ _ _ hh hw.2.2.1 hd (lt_of_lt_of_le bound hc.rotary) eq
    omega
theorem run_tail {α : Type} (c : Config) (hc : c.Valid) (p : Ops α) (m : Memory α)
    (token head col : Nat) (hh : head<c.shape.heads) (hd : col<c.shape.width) (tail : 2*c.shape.half≤col) :
    run c p m c.shape.output (c.shape.address token head col)=m c.shape.output (c.shape.address token head col) :=
  run_frame c p m _ _ (tail_outside c hc token head col hh hd tail)
theorem run_readonly {α : Type} (c : Config) (p : Ops α) (m : Memory α)
    (r : Region) (a : Nat) (hr : r≠.input ∧ r≠.output) : run c p m r a=m r a := by
  apply run_frame
  intro w _
  cases hi : c.shape.inplace <;> simp_all [Writes,Shape.output]
#axiomsClean schedule_covers
#axiomsClean schedule_nodup
#axiomsClean run_readback
#axiomsClean run_tail
end VeriTile.Bench.Repairs.RotaryTransform
