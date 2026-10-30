import bench.tritonbench_g.matmul_kernel.Proofs.Bounds
namespace VeriTile.Bench.Repairs.MatmulKernel
set_option maxHeartbeats 3000000
theorem tile_div (p l b : Nat) (hb : 0<b) (hl : l<b) : (p*b+l)/b=p := by
  simpa [Nat.add_comm,Nat.mul_comm,Nat.div_eq_of_lt hl] using Nat.add_mul_div_left l p hb
theorem tile_unique (p l q k b : Nat) (hb : 0<b) (hl : l<b) (hk : k<b)
    (h : p*b+l=q*b+k) : p=q ∧ l=k := by
  have hd := congrArg (fun x => x/b) h
  dsimp only at hd
  rw [tile_div p l b hb hl,tile_div q k b hb hk] at hd
  refine ⟨hd,?_⟩
  rw [hd] at h
  omega

theorem program_owner (c : Config) (hc : c.Valid) (pid : Program) (w : Work)
    (hw : w∈programWork c pid) : pid=(w.1/c.BM,w.2/c.BN) := by
  obtain ⟨l,hl,eq⟩ := List.mem_map.mp (List.mem_filter.mp hw).1
  rcases pid with ⟨pm,pn⟩
  rcases l with ⟨lm,ln⟩
  simp only [lanes,List.mem_product,List.mem_range] at hl
  subst w
  simp only [workOf,tile_div pm lm c.BM hc.bm hl.1,tile_div pn ln c.BN hc.bn hl.2]

theorem program_nodup (c : Config) (pid : Program) : (programWork c pid).Nodup := by
  apply List.Nodup.filter
  apply (List.nodup_range.product List.nodup_range).map
  intro a b eq
  simp only [workOf,Prod.mk.injEq] at eq
  exact Prod.ext (by omega) (by omega)
theorem schedule_nodup (c : Config) (hc : c.Valid) : (schedule c).Nodup := by
  apply List.nodup_flatMap.mpr
  refine ⟨fun pid _ => program_nodup c pid,?_⟩
  have nd : (programs c).Nodup := List.nodup_range.product List.nodup_range
  apply nd.imp
  intro a b different w ha hb
  exact different ((program_owner c hc a w ha).trans (program_owner c hc b w hb).symm)
theorem writes_unique (c : Config) (hc : c.Valid) (a b : Work)
    (ha : a∈schedule c) (hb : b∈schedule c) (eq : c.C.addr a.1 a.2=c.C.addr b.1 b.2) : a=b :=
  c_injective c hc a b (schedule_active c a ha) (schedule_active c b hb) eq

theorem features_covered (c : Config) (hc : c.Valid) (k : Nat) (hk : k<c.K) :
    ∃ q j e,q<c.steps ∧ j<c.BK/8 ∧ e<8 ∧ q*c.BK+j*8+e=k := by
  obtain ⟨hq,hl,eq⟩ := coordinate_owner k c.BK c.K hc.bk hk
  have hj : (k%c.BK)/8<c.BK/8 := by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0<8)).mpr
    rwa [hc.atoms]
  refine ⟨k/c.BK,(k%c.BK)/8,(k%c.BK)%8,hq,hj,Nat.mod_lt _ (by decide),?_⟩
  have div := Nat.div_add_mod (k%c.BK) 8
  omega
theorem feature_unique (c : Config) (hc : c.Valid) (q j e q' j' e' : Nat)
    (hj : j<c.BK/8) (he : e<8) (hj' : j'<c.BK/8) (he' : e'<8)
    (eq : q*c.BK+j*8+e=q'*c.BK+j'*8+e') : q=q' ∧ j=j' ∧ e=e' := by
  have hl : j*8+e<c.BK := by rw [←hc.atoms];exact tiled_bound j e (c.BK/8) 8 hj he
  have hl' : j'*8+e'<c.BK := by rw [←hc.atoms];exact tiled_bound j' e' (c.BK/8) 8 hj' he'
  have h := tile_unique q (j*8+e) q' (j'*8+e') c.BK hc.bk hl hl' (by omega)
  have hi := tile_unique j e j' e' 8 (by decide) he he' h.2
  exact ⟨h.1,hi⟩
theorem power_of_two_atoms (e : Nat) (he : 4≤e) : (2^e)/8*8=2^e := by
  have hp : 2^e=8*2^(e-3) := by
    conv_lhs => rw [show e=3+(e-3) by omega]
    rw [pow_add]
    norm_num
  exact Nat.div_mul_cancel ⟨2^(e-3),hp⟩
theorem execute_permutation {α} (c : Config) (hc : c.Valid) (p : Ops α)
    (xs ys : List Work) (perm : xs.Perm ys) (active : ∀ w∈xs,c.Active w) (m : Memory α) :
    execute c p xs m=execute c p ys m := by
  classical
  funext r a
  have ay : ∀ w∈ys,c.Active w := fun w hw => active w (perm.mem_iff.mpr hw)
  by_cases hit : ∃ w,w∈xs ∧ Writes c w r a
  · obtain ⟨w,hw,hr,ha⟩ := hit
    subst r
    subst a
    rw [execute_readback c hc p xs m active w hw,
      execute_readback c hc p ys m ay w (perm.mem_iff.mp hw)]
  · have outside : ∀ w∈xs,¬Writes c w r a := by
      intro w hw h;exact hit ⟨w,hw,h⟩
    rw [execute_frame c p xs m r a outside,
      execute_frame c p ys m r a (fun w hw => outside w (perm.mem_iff.mpr hw))]
#axiomsClean execute_permutation
#axiomsClean power_of_two_atoms
#axiomsClean schedule_nodup
#axiomsClean writes_unique
#axiomsClean features_covered
#axiomsClean feature_unique
end VeriTile.Bench.Repairs.MatmulKernel
