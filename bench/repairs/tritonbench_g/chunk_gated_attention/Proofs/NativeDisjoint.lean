import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeOwnership

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 4000000

theorem inner_owns_number (c : HostFP.Inner) (p q : LaunchPlan.Item)
    (i : Fin p.size) (j : Fin q.size) (active : p.base+i.val<c.total)
    (r : RegionName) (off : Nat)
    (left : NativeGrid.innerOwns c p i r off) (right : NativeGrid.innerOwns c q j r off) :
    p.base+i.val=q.base+j.val := by
  let n := p.base+i.val
  let m := q.base+j.val
  have hx : 0<c.PV/c.BV := by
    by_contra h
    have zero := Nat.eq_zero_of_not_pos h
    simp [HostFP.Inner.total,zero] at active
  have hy : 0<c.PK/c.BK := by
    by_contra h
    have zero := Nat.eq_zero_of_not_pos h
    simp [HostFP.Inner.total,zero] at active
  dsimp only [NativeGrid.innerOwns,HostFP.Inner.owns,HostFP.Inner.NT,
    HostFP.Inner.PK,HostFP.Inner.PV] at left right
  rcases left with ⟨e,he,f,hf,lh|lf⟩
  · rcases right with ⟨e',he',f',hf',rh|rf⟩
    · obtain ⟨_,t,ht,ho⟩ := lh
      obtain ⟨_,u,hu,ho'⟩ := rh
      have equal :
          ((Grid.decodeZ (c.PV/c.BV) (c.PK/c.BK) n*c.NT+t)*c.PK+
            (Grid.decodeY (c.PV/c.BV) (c.PK/c.BK) n*c.BK+e))*c.PV+
              (Grid.decodeX (c.PV/c.BV) n*c.BV+f) =
          ((Grid.decodeZ (c.PV/c.BV) (c.PK/c.BK) m*c.NT+u)*c.PK+
            (Grid.decodeY (c.PV/c.BV) (c.PK/c.BK) m*c.BK+e'))*c.PV+
              (Grid.decodeX (c.PV/c.BV) m*c.BV+f') := by
        calc
          _=off := by rw [ho]; dsimp only [n,m]; simp only [HostFP.Inner.NT,HostFP.Inner.PK,HostFP.Inner.PV,HostFP.Cumulative.PT,HostFP.Cumulative.PS]; ring
          _=_ := by rw [ho']; dsimp only [n,m]; simp only [HostFP.Inner.NT,HostFP.Inner.PK,HostFP.Inner.PV,HostFP.Cumulative.PT,HostFP.Cumulative.PS]; ring
      have pos := tile_position c.PK c.PV c.BK c.BV n m _ _ hx hy
        ⟨e,he⟩ ⟨e',he'⟩ ⟨f,hf⟩ ⟨f',hf'⟩ equal
      have heads := Grid.batch_no_duplicate (Nat.zero_lt_of_lt ht) ht hu pos.1
      exact decode_unique _ _ n m hx hy heads.1 pos.2.1 pos.2.2
    · have clash := lh.1.symm.trans rf.2.1
      contradiction
  · rcases right with ⟨e',he',f',hf',rh|rf⟩
    · have clash := lf.2.1.symm.trans rh.1
      contradiction
    · obtain ⟨_,_,ho⟩ := lf
      obtain ⟨_,_,ho'⟩ := rf
      have equal :
          (Grid.decodeZ (c.PV/c.BV) (c.PK/c.BK) n*c.PK+
            (Grid.decodeY (c.PV/c.BV) (c.PK/c.BK) n*c.BK+e))*c.PV+
              (Grid.decodeX (c.PV/c.BV) n*c.BV+f) =
          (Grid.decodeZ (c.PV/c.BV) (c.PK/c.BK) m*c.PK+
            (Grid.decodeY (c.PV/c.BV) (c.PK/c.BK) m*c.BK+e'))*c.PV+
              (Grid.decodeX (c.PV/c.BV) m*c.BV+f') := by
        calc
          _=off := by rw [ho]; dsimp only [n,m]; simp only [HostFP.Inner.NT,HostFP.Inner.PK,HostFP.Inner.PV,HostFP.Cumulative.PT,HostFP.Cumulative.PS]; ring
          _=_ := by rw [ho']; dsimp only [n,m]; simp only [HostFP.Inner.NT,HostFP.Inner.PK,HostFP.Inner.PV,HostFP.Cumulative.PT,HostFP.Cumulative.PS]; ring
      have pos := tile_position c.PK c.PV c.BK c.BV n m _ _ hx hy
        ⟨e,he⟩ ⟨e',he'⟩ ⟨f,hf⟩ ⟨f',hf'⟩ equal
      exact decode_unique _ _ n m hx hy pos.1 pos.2.1 pos.2.2

theorem prefix_owns_number (c : HostFP.Cumulative) (p q : LaunchPlan.Item)
    (i : Fin p.size) (j : Fin q.size) (r : RegionName) (off : Nat)
    (left : NativeGrid.prefixOwns c p i r off) (right : NativeGrid.prefixOwns c q j r off) :
    p.base+i.val=q.base+j.val := by
  let n := p.base+i.val
  let m := q.base+j.val
  dsimp only [NativeGrid.prefixOwns,HostFP.Cumulative.owns,HostFP.Cumulative.PT,
    HostFP.Cumulative.PS,HostFP.Cumulative.total] at left right
  obtain ⟨active,_,e,he,f,hf,ho⟩ := left
  change p.base+i.val<c.total at active
  obtain ⟨_,_,e',he',f',hf',ho'⟩ := right
  have hx : 0<c.PS/c.BS := by
    by_contra h
    have zero := Nat.eq_zero_of_not_pos h
    simp [HostFP.Cumulative.total,zero] at active
  have hy : 0<c.PT/c.BT := by
    by_contra h
    have zero := Nat.eq_zero_of_not_pos h
    simp [HostFP.Cumulative.total,zero] at active
  have equal :
      (Grid.decodeZ (c.PS/c.BS) (c.PT/c.BT) n*c.PT+
        (Grid.decodeY (c.PS/c.BS) (c.PT/c.BT) n*c.BT+e))*c.PS+
          (Grid.decodeX (c.PS/c.BS) n*c.BS+f) =
      (Grid.decodeZ (c.PS/c.BS) (c.PT/c.BT) m*c.PT+
        (Grid.decodeY (c.PS/c.BS) (c.PT/c.BT) m*c.BT+e'))*c.PS+
          (Grid.decodeX (c.PS/c.BS) m*c.BS+f') := by
    calc
      _=off := by rw [ho]; dsimp only [n,m]; simp only [HostFP.Inner.NT,HostFP.Inner.PK,HostFP.Inner.PV,HostFP.Cumulative.PT,HostFP.Cumulative.PS]; ring
      _=_ := by rw [ho']; dsimp only [n,m]; simp only [HostFP.Inner.NT,HostFP.Inner.PK,HostFP.Inner.PV,HostFP.Cumulative.PT,HostFP.Cumulative.PS]; ring
  have pos := tile_position c.PT c.PS c.BT c.BS n m _ _ hx hy
    ⟨e,he⟩ ⟨e',he'⟩ ⟨f,hf⟩ ⟨f',hf'⟩ equal
  exact decode_unique _ _ n m hx hy pos.1 pos.2.1 pos.2.2

theorem inner_disjoint (c : HostFP.Inner) (s t : State NativeGraph.Expr)
    (i j : Job c.plans) (r : RegionName) (off : Nat)
    (left : NativeStores.Hits (innerStores c i s) r off)
    (right : NativeStores.Hits (innerStores c j t) r off) : i=j := by
  apply number_injective _ (inner_ordered c)
  exact inner_owns_number c (plan i) (plan j) i.2 j.2 (inner_job_bound c i) r off
    ((inner_owns_hits c (plan i) i.2 s r off).mpr left)
    ((inner_owns_hits c (plan j) j.2 t r off).mpr right)

theorem prefix_disjoint (c : HostFP.Cumulative) (s t : State NativeGraph.Expr)
    (i j : Job c.plans) (r : RegionName) (off : Nat)
    (left : NativeStores.Hits (prefixStores c i s) r off)
    (right : NativeStores.Hits (prefixStores c j t) r off) : i=j := by
  apply number_injective _ (prefix_ordered c)
  exact prefix_owns_number c (plan i) (plan j) i.2 j.2 r off
    ((prefix_owns_hits c (plan i) i.2 s r off).mpr left)
    ((prefix_owns_hits c (plan j) j.2 t r off).mpr right)

#axiomsClean inner_disjoint
#axiomsClean prefix_disjoint
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
