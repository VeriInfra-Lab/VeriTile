import bench.repairs.tritonbench_g.triton_conv2d_fwd.Kernels
namespace VeriTile.Bench.Repairs.Conv2d

theorem plane_pos (s : Shape) (v : Valid s) : 0<s.plane := Nat.mul_pos v.oh v.ow

theorem block_cover (n tile : Nat) (ht : 0<tile) : n≤blocks n tile*tile := by
  unfold blocks
  have rem := Nat.mod_lt (n+tile-1) ht
  have split := Nat.mod_add_div (n+tile-1) tile
  rw [Nat.mul_comm tile] at split
  omega

theorem block_contains (n tile x : Nat) (ht : 0<tile) (hx : x<n) :
    x/tile<blocks n tile := by
  exact (Nat.div_lt_iff_lt_mul ht).mpr (lt_of_lt_of_le hx (block_cover n tile ht))

theorem source_grid_covers (s : Shape) (v : Valid s) (x : Cell) (hx : s.Active x) :
    ∃ pid∈programs s,∃ lane∈lanes,cellOf pid lane=x := by
  have outputBound : x.feature<s.outputs := by
    have h : s.og≤s.groups*s.og := by nlinarith [v.groups]
    exact lt_of_lt_of_le hx.2.2 h
  refine ⟨(x.spatial/128,x.feature/32,x.group),?_,(x.spatial%128,x.feature%32),?_,?_⟩
  · simp only [programs,List.mem_product,List.mem_range]
    exact ⟨block_contains _ _ _ (by decide) hx.2.1,
      block_contains _ _ _ (by decide) outputBound,hx.1⟩
  · simp only [lanes,List.mem_product,List.mem_range]
    exact ⟨Nat.mod_lt _ (by decide),Nat.mod_lt _ (by decide)⟩
  · cases x
    simp only [cellOf,Cell.mk.injEq,true_and]
    constructor <;> omega

theorem source_owner_unique (p q : Program) (i j : Lane)
    (hi : i∈lanes) (hj : j∈lanes) (same : cellOf p i=cellOf q j) : p=q ∧ i=j := by
  have hi' : i.1<128 ∧ i.2<32 := by
    rcases i with ⟨a,b⟩;simpa only [lanes,List.mem_product,List.mem_range] using hi
  have hj' : j.1<128 ∧ j.2<32 := by
    rcases j with ⟨a,b⟩;simpa only [lanes,List.mem_product,List.mem_range] using hj
  have hg := congrArg Cell.group same
  have hr := congrArg Cell.spatial same
  have hf := congrArg Cell.feature same
  simp only [cellOf] at hg hr hf
  have pr : p.1=q.1 := by omega
  have pf : p.2.1=q.2.1 := by omega
  have ir : i.1=j.1 := by omega
  have jf : i.2=j.2 := by omega
  exact ⟨Prod.ext pr (Prod.ext pf hg),Prod.ext ir jf⟩

theorem batch_eq (s : Shape) (r : Nat) : batch s r=r/s.plane := by
  simp [batch,Shape.plane,Nat.div_div_eq_div_mul,Nat.mul_comm]

theorem spatial_reconstruct (s : Shape) (r : Nat) :
    batch s r*s.plane+outH s r*s.ow+outW s r=r := by
  have a := Nat.mod_add_div r s.ow
  have b := Nat.mod_add_div (r/s.ow) s.oh
  simp only [batch,Shape.plane,outH,outW]
  nlinarith

theorem spatial_remainder (s : Shape) (v : Valid s) (r : Nat) :
    outH s r*s.ow+outW s r=r%s.plane := by
  have a := spatial_reconstruct s r
  rw [batch_eq] at a
  have b := Nat.mod_add_div r s.plane
  rw [Nat.mul_comm s.plane] at b
  omega

theorem output_flat (s : Shape) (v : Valid s) (x : Cell) :
    outputPointer s x=(x.spatial/s.plane*s.outputs+x.group*s.og+x.feature)*s.plane+
      x.spatial%s.plane := by
  have h := spatial_remainder s v x.spatial
  unfold outputPointer
  rw [batch_eq]
  simp only [Shape.plane] at *
  nlinarith

theorem feature_bound (s : Shape) (x : Cell) (hx : s.Active x) :
    x.group*s.og+x.feature<s.outputs := by
  unfold Shape.outputs
  nlinarith [hx.1,hx.2.2]

theorem address_bound (s : Shape) (v : Valid s) (x : Cell) (hx : s.Active x) :
    outputPointer s x<s.size := by
  rw [output_flat s v]
  have r : x.spatial/s.plane<s.n := by
    apply (Nat.div_lt_iff_lt_mul (plane_pos s v)).mpr
    simpa [Shape.spatial] using hx.2.1
  have f := feature_bound s x hx
  have rem := Nat.mod_lt x.spatial (plane_pos s v)
  have upper : x.spatial/s.plane*s.outputs+x.group*s.og+x.feature<s.n*s.outputs := by
    nlinarith
  have scaled := Nat.mul_le_mul_right s.plane (Nat.succ_le_of_lt upper)
  unfold Shape.size
  nlinarith

theorem address_injective (s : Shape) (v : Valid s) (x y : Cell)
    (hx : s.Active x) (hy : s.Active y) (same : outputPointer s x=outputPointer s y) : x=y := by
  rw [output_flat s v,output_flat s v] at same
  have px := Nat.mod_lt x.spatial (plane_pos s v)
  have py := Nat.mod_lt y.spatial (plane_pos s v)
  have divrem (a b : Nat) (hb : b<s.plane) : (a*s.plane+b)/s.plane=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ (plane_pos s v),Nat.div_eq_of_lt hb];simp
  have upper := congrArg (fun z => z/s.plane) same
  simp only [divrem _ _ px,divrem _ _ py] at upper
  have fx := feature_bound s x hx
  have fy := feature_bound s y hy
  have op : 0<s.outputs := Nat.mul_pos v.groups v.og
  have split (b f : Nat) (hf : f<s.outputs) : (b*s.outputs+f)/s.outputs=b := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ op,Nat.div_eq_of_lt hf];simp
  have bb := congrArg (fun z => z/s.outputs) upper
  dsimp only at bb
  rw [Nat.add_assoc,Nat.add_assoc] at bb
  rw [split _ _ fx,split _ _ fy] at bb
  rw [bb] at upper
  have ff : x.group*s.og+x.feature=y.group*s.og+y.feature := by omega
  have grp (g f : Nat) (hf : f<s.og) : (g*s.og+f)/s.og=g := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ v.og,Nat.div_eq_of_lt hf];simp
  have gg := congrArg (fun z => z/s.og) ff
  simp only [grp _ _ hx.2.2,grp _ _ hy.2.2] at gg
  rw [gg] at ff
  have f : x.feature=y.feature := by omega
  rw [bb,upper] at same
  have rr : x.spatial%s.plane=y.spatial%s.plane := by omega
  have a := Nat.mod_add_div x.spatial s.plane
  have b := Nat.mod_add_div y.spatial s.plane
  rw [bb,rr] at a
  have r : x.spatial=y.spatial := by omega
  cases x;cases y;simp_all

#axiomsClean source_grid_covers
#axiomsClean source_owner_unique
#axiomsClean output_flat
#axiomsClean address_injective
#axiomsClean address_bound
end VeriTile.Bench.Repairs.Conv2d
