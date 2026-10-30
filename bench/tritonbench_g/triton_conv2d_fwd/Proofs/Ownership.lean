import bench.tritonbench_g.triton_conv2d_fwd.Proofs.Scatter
namespace VeriTile.Bench.Repairs.Conv2d
variable {α β : Type}

theorem output_owner (s : Shape) (v : Valid s) (x : Cell) (hx : s.Active x) :
    ∃ pid∈programs s,∃ lane∈lanes,cellOf pid lane=x ∧
      ∀ q∈programs s,∀ k∈lanes,s.Active (cellOf q k) →
        outputPointer s (cellOf q k)=outputPointer s x → q=pid ∧ k=lane := by
  obtain ⟨pid,hp,lane,hl,e⟩ := source_grid_covers s v x hx
  refine ⟨pid,hp,lane,hl,e,?_⟩
  intro q hq k hk active address
  have eq := address_injective s v (cellOf q k) x active hx address
  rw [←e] at eq
  exact source_owner_unique q pid k lane hk hl eq

theorem any_program_order (s : Shape) (v : Valid s) (p : ISA α β) (flags : Flags)
    (initial : Memory α) (ps : List Program) (order : ps.Perm (programs s))
    (x : Cell) (hx : s.Active x) :
    sourcePrograms s p flags ps initial .output (outputPointer s x)=sourceValue s p flags initial x := by
  rw [sourcePrograms_commit]
  apply commit_read
  · obtain ⟨pid,hp,lane,hl,e⟩ := source_grid_covers s v x hx
    refine ⟨(outputPointer s x,sourceValue s p flags initial x),List.mem_flatMap.mpr ⟨pid,order.mem_iff.mpr hp,?_⟩,rfl⟩
    exact (programWrites_member s v p flags pid initial _).mpr ⟨lane,hl,x,e,hx,rfl⟩
  · intro w hw address
    obtain ⟨pid,hp,hw⟩ := List.mem_flatMap.mp hw
    obtain ⟨lane,hl,y,e,hy,wy⟩ := (programWrites_member s v p flags pid initial w).mp hw
    subst w
    have eq := address_injective s v y x hy hx address
    simp [eq]

theorem address_covers (s : Shape) (v : Valid s) (a : Nat) (ha : a<s.size) :
    ∃ x : Cell,s.Active x ∧ outputPointer s x=a := by
  let o := a/s.plane%s.outputs
  let n := a/(s.outputs*s.plane)
  let x : Cell := ⟨o/s.og,n*s.plane+a%s.plane,o%s.og⟩
  have pp := plane_pos s v
  have op : 0<s.outputs := Nat.mul_pos v.groups v.og
  have ob : o<s.outputs := Nat.mod_lt _ op
  have nb : n<s.n := by
    apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos op pp)).mpr
    simpa [Shape.size,Nat.mul_assoc] using ha
  have ar := Nat.mod_lt a pp
  have group : o/s.og<s.groups := by
    apply (Nat.div_lt_iff_lt_mul v.og).mpr
    exact ob
  have rem : x.spatial%s.plane=a%s.plane := by
    dsimp [x]
    simp [Nat.add_mod]
  have div : x.spatial/s.plane=n := by
    dsimp [x]
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ pp,Nat.div_eq_of_lt ar]
    simp
  have spatial : x.spatial<s.spatial := by
    dsimp [x,Shape.spatial]
    have bound := Nat.mul_le_mul_right s.plane (Nat.succ_le_of_lt nb)
    nlinarith
  refine ⟨x,⟨group,spatial,Nat.mod_lt _ v.og⟩,?_⟩
  rw [output_flat s v,div,rem]
  have feature : x.group*s.og+x.feature=o := by
    dsimp only [x]
    simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div o s.og
  rw [Nat.add_assoc,feature]
  have split := Nat.mod_add_div (a/s.plane) s.outputs
  have quotient : a/s.plane/s.outputs=n := by
    dsimp [n]
    rw [Nat.div_div_eq_div_mul,Nat.mul_comm s.plane]
  rw [quotient] at split
  have upper : n*s.outputs+o=a/s.plane := by
    dsimp only [o]
    simpa [Nat.mul_comm,Nat.add_comm] using split
  rw [upper]
  simpa [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div a s.plane

#axiomsClean output_owner
#axiomsClean any_program_order
#axiomsClean address_covers
end VeriTile.Bench.Repairs.Conv2d
