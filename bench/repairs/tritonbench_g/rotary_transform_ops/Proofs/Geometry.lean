import bench.repairs.tritonbench_g.rotary_transform_ops.Model
namespace VeriTile.Bench.Repairs.RotaryTransformOps

theorem blockM_pos (s : Shape) : 0<s.blockM := by unfold Shape.blockM; split_ifs <;> omega
theorem blockK_covers (c : Config) (hc : c.Valid) : c.shape.half≤c.shape.blockK/2 := by
  have h := hc.rotary;have w := hc.widthLimit
  unfold Shape.blockK
  split_ifs <;> omega
theorem start_zero (c : Config) (hc : c.Valid) : c.start 0=0 := by
  simp [Config.start,hc.cumulativeZero]
theorem start_mono (c : Config) (hc : c.Valid) (a b : Nat) (hab : a≤b) (hb : b≤c.shape.batches) :
    c.start a≤c.start b := by
  unfold Config.start
  split
  · exact hc.cumulativeMonotone a b hab hb
  · exact Nat.mul_le_mul_right _ hab
theorem start_next (c : Config) (hc : c.Valid) (b : Nat) (hb : b<c.shape.batches) :
    c.start b+c.length b=c.start (b+1) := by
  have hm := hc.cumulativeMonotone b (b+1) (by omega) (by omega)
  unfold Config.start Config.length
  split
  · exact Nat.add_sub_of_le hm
  · ring
theorem start_total (c : Config) (hc : c.Valid) : c.start c.shape.batches=c.shape.total := by
  rw [hc.total];rfl
theorem row_bound (c : Config) (hc : c.Valid) (b r : Nat)
    (hb : b<c.shape.batches) (hr : r<c.length b) : c.start b+r<c.shape.total := by
  have h := start_next c hc b hb
  have m := start_mono c hc (b+1) c.shape.batches (by omega) (by omega)
  rw [start_total c hc] at m
  omega
theorem row_injective (c : Config) (hc : c.Valid) (b r b' r' : Nat)
    (hb : b<c.shape.batches) (hr : r<c.length b)
    (hb' : b'<c.shape.batches) (hr' : r'<c.length b')
    (equal : c.start b+r=c.start b'+r') : b=b' ∧ r=r' := by
  have hn := start_next c hc b hb
  have hn' := start_next c hc b' hb'
  have eq : b=b' := by
    by_contra ne
    rcases Nat.lt_or_gt_of_ne ne with lt|gt
    · have hm := start_mono c hc (b+1) b' (by omega) (by omega);omega
    · have hm := start_mono c hc (b'+1) b (by omega) (by omega);omega
  exact ⟨eq,by simpa [eq] using equal⟩

/-- Every packed row belongs to one logical sequence, including empty sequences. -/
theorem intervals_cover (c : Config) (hc : c.Valid) (n : Nat) (hn : n≤c.shape.batches)
    (r : Nat) (hr : r<c.start n) :
    ∃ b,b<n ∧ c.start b≤r ∧ r<c.start b+c.length b := by
  induction n with
  | zero => rw [start_zero c hc] at hr;omega
  | succ n ih =>
    by_cases old : r<c.start n
    · obtain ⟨b,hb,lo,hi⟩ := ih (by omega) old
      exact ⟨b,by omega,lo,hi⟩
    · exact ⟨n,by omega,by omega,by rw [start_next c hc n (by omega)];exact hr⟩

theorem dim_bound (s : Shape) (p : Nat) (hp : p<s.half) (o : Bool) :
    s.dim p o<2*s.half := by
  unfold Shape.dim
  cases s.interleaved <;> cases o <;> simp only [↓reduceIte,Bool.false_eq_true] <;> omega
theorem dim_injective (s : Shape) (p q : Nat) (op oq : Bool)
    (hp : p<s.half) (hq : q<s.half) (eq : s.dim p op=s.dim q oq) : p=q ∧ op=oq := by
  unfold Shape.dim at eq
  cases hi : s.interleaved <;> cases op <;> cases oq <;> simp_all [hi] <;> omega
theorem address_injective (s : Shape) (a h d a' h' d' : Nat)
    (hh : h<s.heads) (hh' : h'<s.heads) (hd : d<s.width) (hd' : d'<s.width)
    (eq : s.address a h d=s.address a' h' d') : a=a' ∧ h=h' ∧ d=d' := by
  have w : 0<s.width := by omega
  have heads : 0<s.heads := by omega
  unfold Shape.address at eq
  have div (k z : Nat) (hz : z<s.width) : (k*s.width+z)/s.width=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ w,Nat.div_eq_of_lt hz];simp
  have rows := congrArg (fun n => n/s.width) eq
  simp only [div _ _ hd,div _ _ hd'] at rows
  have dims := congrArg (fun n => n%s.width) eq
  simp [Nat.add_mod,Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd'] at dims
  have divh (k z : Nat) (hz : z<s.heads) : (k*s.heads+z)/s.heads=k := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ heads,Nat.div_eq_of_lt hz];simp
  have rows' := congrArg (fun n => n/s.heads) rows
  simp only [divh _ _ hh,divh _ _ hh'] at rows'
  exact ⟨rows',by simpa [rows'] using rows,dims⟩

theorem cell_injective (c : Config) (hc : c.Valid) (a b : Work) (oa ob : Bool)
    (ha : c.Active a) (hb : c.Active b) (eq : c.cell a oa=c.cell b ob) : a=b ∧ oa=ob := by
  have x := address_injective c.shape _ _ _ _ _ _ ha.2.2.1 hb.2.2.1
    (lt_of_lt_of_le (dim_bound _ _ ha.2.2.2 _) hc.rotary)
    (lt_of_lt_of_le (dim_bound _ _ hb.2.2.2 _) hc.rotary) eq
  have rows := row_injective c hc _ _ _ _ ha.1 ha.2.1 hb.1 hb.2.1 x.1
  have dims := dim_injective _ _ _ _ _ ha.2.2.2 hb.2.2.2 x.2.2
  exact ⟨by cases a;cases b;simp_all [Work.mk.injEq],dims.2⟩
theorem cells_distinct (c : Config) (w : Work) (hw : c.Active w) : c.cell w false≠c.cell w true := by
  intro eq
  have h : c.shape.dim w.pair false=c.shape.dim w.pair true := by
    unfold Config.cell Shape.address at eq
    exact Nat.add_left_cancel eq
  exact Bool.noConfusion (dim_injective _ _ _ _ _ hw.2.2.2 hw.2.2.2 h).2

#axiomsClean row_injective
#axiomsClean intervals_cover
#axiomsClean cell_injective
end VeriTile.Bench.Repairs.RotaryTransformOps
