import bench.repairs.tritonbench_g.matmul_kernel.Kernels
namespace VeriTile.Bench.Repairs.MatmulKernel
set_option maxHeartbeats 2000000
theorem ceil_cover (x b : Nat) (hb : 0<b) : x≤ceil x b*b := by
  have hq := Nat.div_add_mod (x+b-1) b
  have hr := Nat.mod_lt (x+b-1) hb
  unfold ceil
  rw [Nat.mul_comm b] at hq
  omega
theorem view_bound (v : View) (hv : v.Valid) (r n : Nat) (hr : r<v.rows) (hn : n<v.cols) :
    v.addr r n<v.capacity := by
  rcases hv.allocation with h|h|h
  · omega
  · omega
  · have ar := Nat.mul_le_mul_right v.sr (show r≤v.rows-1 by omega)
    have an := Nat.mul_le_mul_right v.sc (show n≤v.cols-1 by omega)
    unfold View.addr
    omega
theorem separated_row_injective (rows cols sr sc : Nat) (hs : 0<sc) (sep : cols*sc≤sr)
    (r n r' n' : Nat) (hn : n<cols) (hn' : n'<cols)
    (eq : r*sr+n*sc=r'*sr+n'*sc) : r=r' ∧ n=n' := by
  have nr : n*sc<sr := lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hn hs) sep
  have nr' : n'*sc<sr := lt_of_lt_of_le (Nat.mul_lt_mul_of_pos_right hn' hs) sep
  have bound (a b x y : Nat) (lt : a<b) (hy : y<sr) : a*sr+y<b*sr+x := by
    have le := Nat.mul_le_mul_right sr (show a+1≤b by omega)
    rw [Nat.add_mul,Nat.one_mul] at le
    omega
  have rr : r=r' := by
    by_contra ne
    rcases Nat.lt_or_gt_of_ne ne with h|h
    · have := bound r r' (n'*sc) (n*sc) h nr;omega
    · have := bound r' r (n*sc) (n'*sc) h nr';omega
  have nn : n=n' := by
    have : n*sc=n'*sc := by rw [rr] at eq;omega
    exact Nat.eq_of_mul_eq_mul_right hs this
  exact ⟨rr,nn⟩
theorem view_injective (v : View) (hv : v.Valid) (w z : Work)
    (hw : w.1<v.rows ∧ w.2<v.cols) (hz : z.1<v.rows ∧ z.2<v.cols)
    (eq : v.addr w.1 w.2=v.addr z.1 z.2) : w=z := by
  rcases hv.separated with h|h
  · obtain ⟨hr,hn⟩ := separated_row_injective v.rows v.cols v.sr v.sc hv.colPositive h
      w.1 w.2 z.1 z.2 hw.2 hz.2 eq
    exact Prod.ext hr hn
  · have eq' : w.2*v.sc+w.1*v.sr=z.2*v.sc+z.1*v.sr := by unfold View.addr at eq;omega
    obtain ⟨hn,hr⟩ := separated_row_injective v.cols v.rows v.sc v.sr hv.rowPositive h
      w.2 w.1 z.2 z.1 hw.1 hz.1 eq'
    exact Prod.ext hr hn

theorem c_injective (c : Config) (hc : c.Valid) (w z : Work) (hw : c.Active w) (hz : c.Active z)
    (eq : c.C.addr w.1 w.2=c.C.addr z.1 z.2) : w=z := by
  apply view_injective c.C hc.cValid w z _ _ eq
  · simpa [hc.cShape.1,hc.cShape.2] using hw
  · simpa [hc.cShape.1,hc.cShape.2] using hz

theorem program_active (c : Config) (pid : Program) (w : Work) (h : w∈programWork c pid) : c.Active w :=
  of_decide_eq_true (List.mem_filter.mp h).2
theorem schedule_active (c : Config) (w : Work) (h : w∈schedule c) : c.Active w := by
  obtain ⟨pid,_,hw⟩ := List.mem_flatMap.mp h
  exact program_active c pid w hw
theorem coordinate_owner (x b total : Nat) (hb : 0<b) (hx : x<total) :
    x/b<ceil total b ∧ x%b<b ∧ x/b*b+x%b=x := by
  have bound : x<ceil total b*b := lt_of_lt_of_le hx (ceil_cover total b hb)
  refine ⟨(Nat.div_lt_iff_lt_mul hb).mpr bound,Nat.mod_lt x hb,?_⟩
  simpa [Nat.mul_comm] using Nat.div_add_mod x b
theorem schedule_covers (c : Config) (hc : c.Valid) (w : Work) (hw : c.Active w) : w∈schedule c := by
  obtain ⟨hr,lr,er⟩ := coordinate_owner w.1 c.BM c.M hc.bm hw.1
  obtain ⟨hn,ln,en⟩ := coordinate_owner w.2 c.BN c.N hc.bn hw.2
  apply List.mem_flatMap.mpr
  refine ⟨(w.1/c.BM,w.2/c.BN),?_,?_⟩
  · simpa [programs] using And.intro hr hn
  · apply List.mem_filter.mpr
    refine ⟨?_,by simpa using hw⟩
    apply List.mem_map.mpr
    refine ⟨(w.1%c.BM,w.2%c.BN),by simpa [lanes] using And.intro lr ln,?_⟩
    exact Prod.ext er en
theorem schedule_empty (c : Config) (h : c.M=0 ∨ c.N=0) : schedule c=[] := by
  apply List.eq_nil_iff_forall_not_mem.mpr
  intro w hw
  have ha := schedule_active c w hw
  rcases h with h|h <;> simp_all [Config.Active]
#axiomsClean view_bound
#axiomsClean view_injective
#axiomsClean schedule_covers
end VeriTile.Bench.Repairs.MatmulKernel
