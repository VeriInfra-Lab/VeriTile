import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Vector
namespace VeriTile.Bench.Repairs.RotaryTransformOps
def Shape.pairOf (s : Shape) (d : Nat) := if s.interleaved then d/2 else if d<s.half then d else d-s.half
def Shape.second (s : Shape) (d : Nat) : Bool := if s.interleaved then decide (d%2=1) else decide (s.half≤d)
theorem column_inverse (s : Shape) (d : Nat) (hd : d<2*s.half) :
    s.pairOf d<s.half ∧ s.dim (s.pairOf d) (s.second d)=d := by
  unfold Shape.pairOf Shape.second Shape.dim
  cases hi : s.interleaved <;> simp only [hi,↓reduceIte,Bool.false_eq_true]
  · split_ifs <;> simp_all <;> omega
  · have hm := Nat.mod_lt d (by decide : 0<2)
    split_ifs <;> simp_all <;> omega
theorem pair_inverse (s : Shape) (j : Nat) (hj : j<s.half) (o : Bool) :
    s.pairOf (s.dim j o)=j ∧ s.second (s.dim j o)=o := by
  have back := column_inverse s (s.dim j o) (dim_bound s j hj o)
  exact dim_injective s _ _ _ _ back.1 hj back.2
theorem adjacent_swap (j : Nat) (o : Bool) :
    (2*j+(if o then 1 else 0))+(((2*j+(if o then 1 else 0))+1)%2)*2-1=
      2*j+(if o then 0 else 1) := by
  cases o <;> simp <;> omega
theorem adjacent_repeat (j : Nat) (o : Bool) : (2*j+(if o then 1 else 0))/2=j := by
  cases o <;> simp <;> omega

/-- The original adjacent branch's per-column where expression. This retains
self/partner loads and the actual add/sub order. -/
def sourceColumn {α : Type} (p : Ops α) (conjugate second : Bool) (self partner co sine : α) : α :=
  let sn := if conjugate then p.neg sine else sine
  let selfCos := p.mul self co
  let partnerSin := p.mul partner sn
  if second then p.add selfCos partnerSin else p.sub selfCos partnerSin

theorem adjacent_column_pair {α : Type} (p : Ops α) (conjugate second : Bool) (a b co sn : α) :
    sourceColumn p conjugate second (if second then b else a) (if second then a else b) co sn=
      component (sourcePair p true conjugate a b co sn) second := by
  cases second <;> rfl

theorem program_early_return (c : Config) (pid : Program) (early : c.length pid.2.1≤pid.1*c.shape.blockM) :
    programWork c pid=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w hw
  obtain ⟨mapped,active⟩ := List.mem_filter.mp hw
  obtain ⟨l,hl,rfl⟩ := List.mem_map.mp mapped
  have ha : c.Active (workOf c (pid,l)) := of_decide_eq_true active
  have hr := ha.2.1
  simp only [workOf] at hr
  omega
theorem source_early_return {α : Type} (c : Config) (p : Ops α) (pid : Program)
    (early : c.length pid.2.1≤pid.1*c.shape.blockM) (m : Memory α) : sourceProgram c p pid m=m := by
  simp [sourceProgram,program_early_return c pid early,vectorBranch,vectorWrites,Stores.commit]
theorem zero_half_schedule (c : Config) (zero : c.shape.half=0) : schedule c=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w hw
  have h := (schedule_active c w hw).2.2.2
  omega
theorem empty_total_schedule (c : Config) (hc : c.Valid) (empty : c.shape.total=0) : schedule c=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w hw
  have h := schedule_active c w hw
  have bound := row_bound c hc _ _ h.1 h.2.1
  omega
#axiomsClean column_inverse
#axiomsClean adjacent_column_pair
#axiomsClean source_early_return
end VeriTile.Bench.Repairs.RotaryTransformOps

