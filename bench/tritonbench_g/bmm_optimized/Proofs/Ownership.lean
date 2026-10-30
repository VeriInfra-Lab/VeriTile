import bench.tritonbench_g.bmm_optimized.Sources.SourceIndex
import bench.tritonbench_g.bmm_optimized.Sources.Scatter
namespace VeriTile.Bench.Repairs.BmmOptimized

theorem source_ownership (s : Shape) (c : Config) (p q : Program)
    (hp : p∈programs s c) (hq : q∈programs s c) (u v : Lane)
    (hu : u∈lanes c) (hv : v∈lanes c)
    (same : cellOf s c p u=cellOf s c q v) : p=q ∧ u=v := by
  have cb := config_bounds c
  have up := List.mem_product.mp hu
  have vp := List.mem_product.mp hv
  have ur := List.mem_range.mp up.1
  have uc := List.mem_range.mp up.2
  have vr := List.mem_range.mp vp.1
  have vc := List.mem_range.mp vp.2
  have divide (base tile lane : Nat) (ht : 0<tile) (hl : lane<tile) :
      (base*tile+lane)/tile=base := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ ht,Nat.div_eq_of_lt hl];simp
  have rem (base tile lane : Nat) (hl : lane<tile) : (base*tile+lane)%tile=lane := by
    simp [Nat.add_mod,Nat.mod_eq_of_lt hl]
  have rr := congrArg (fun x : Cell => x.row/c.tm) same
  have cc := congrArg (fun x : Cell => x.column/c.tn) same
  have ul := congrArg (fun x : Cell => x.row%c.tm) same
  have vl := congrArg (fun x : Cell => x.column%c.tn) same
  have bb := congrArg Cell.batch same
  dsimp only [cellOf] at rr cc ul vl bb
  rw [divide _ _ _ cb.1 ur,divide _ _ _ cb.1 vr] at rr
  rw [divide _ _ _ cb.2.2.1 uc,divide _ _ _ cb.2.2.1 vc] at cc
  rw [rem _ _ _ ur,rem _ _ _ vr] at ul
  rw [rem _ _ _ uc,rem _ _ _ vc] at vl
  have pp := List.mem_product.mp hp
  have qp := List.mem_product.mp hq
  have programs := swizzle_injective_nat p.1 p.2.1 q.1 q.2.1 (s.gx c) (s.gy c) c.group
    cb.2.2.2.2.1 (List.mem_range.mp pp.1) (List.mem_range.mp (List.mem_product.mp pp.2).1)
    (List.mem_range.mp qp.1) (List.mem_range.mp (List.mem_product.mp qp.2).1) rr cc
  exact ⟨Prod.ext programs.1 (Prod.ext programs.2 bb),Prod.ext ul vl⟩

theorem output_owner (s : Shape) (c : Config) (p q : Program)
    (hp : p∈programs s c) (hq : q∈programs s c) (u v : Lane)
    (hu : u∈lanes c) (hv : v∈lanes c)
    (ha : s.Active (cellOf s c p u)) (hb : s.Active (cellOf s c q v))
    (same : s.o (cellOf s c p u).batch (cellOf s c p u).row (cellOf s c p u).column=
      s.o (cellOf s c q v).batch (cellOf s c q v).row (cellOf s c q v).column) : p=q ∧ u=v :=
  source_ownership s c p q hp hq u v hu hv (address_injective s _ _ ha hb same)

/-- The readback theorem holds for every runtime CTA permutation. Current
memory is still used by every step; no initial-snapshot executor is substituted. -/
theorem any_program_order {α : Type} (s : Shape) (c : Config) (p : Scalar α)
    (ps : List Program) (order : ps.Perm (programs s c)) (m : Memory α)
    (x : Cell) (hx : s.Active x) :
    sourcePrograms s c p ps m .output (s.o x.batch x.row x.column)=ordered s p m x := by
  rw [sourcePrograms_commit,←source_ordered s p m x hx]
  have members : ∀ w,w∈ps.flatMap (fun pid => programWrites s c p pid m) ↔ w∈allWrites s c p m := by
    intro w
    simp only [allWrites,List.mem_flatMap,order.mem_iff]
  apply commit_read
  · exact ⟨_,(members _).mpr (allWrites_covers s c p m x hx),rfl⟩
  · intro w hw eq
    obtain ⟨y,hy,wy⟩ := allWrites_member s c p m w ((members w).mp hw)
    subst w
    have same := address_injective s y x hy hx eq
    simp [same]

#axiomsClean source_ownership
#axiomsClean output_owner
#axiomsClean any_program_order
end VeriTile.Bench.Repairs.BmmOptimized
