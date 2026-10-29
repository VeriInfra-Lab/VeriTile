import bench.tritonbench_g.quant_transpose_kernel.Kernels
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic.Linarith
import Mathlib.Data.Nat.ModEq

namespace QuantTransposeFinite
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem rotated_residue (a s t : Nat) (hs : 0<s) (ht : t<s) :
    let u := if a%s≤t then t-a%s else s+t-a%s
    u<s ∧ (a+u)%s=t := by
  have ha := Nat.mod_lt a hs
  dsimp only
  split_ifs with h
  · constructor
    · omega
    · rw [Nat.add_mod]
      have he : a%s+(t-a%s)%s=t := by rw [Nat.mod_eq_of_lt (show t-a%s<s by omega)]; omega
      rw [he,Nat.mod_eq_of_lt ht]
  · constructor
    · omega
    · rw [Nat.add_mod,Nat.mod_eq_of_lt (show s+t-a%s<s by omega)]
      have he : a%s+(s+t-a%s)=s+t := by omega
      rw [he,Nat.add_mod]
      simp [Nat.mod_eq_of_lt ht]

theorem inverse_tile (gm gn r c : Nat) (hr : r<gm) (hc : c<gn) :
    inversePid gm gn r c<gm*gn ∧
    tileRow gm gn (inversePid gm gn r c)=r ∧
    tileCol gm gn (inversePid gm gn r c)=c := by
  let g := r/8
  let t := r%8
  let s := min (gm-g*8) 8
  let a := g*(8*gn)
  let u := if a%s≤t then t-a%s else s+t-a%s
  have hg : g*8+t=r := by dsimp [g,t]; omega
  have ht8 : t<8 := by dsimp [t]; omega
  have hgm : g*8<gm := by omega
  have hs : 0<s := by dsimp [s]; omega
  have hs8 : s≤8 := Nat.min_le_right _ _
  have hsr : s≤gm-g*8 := Nat.min_le_left _ _
  have ht : t<s := by dsimp [s]; omega
  have hu : u<s ∧ (a+u)%s=t := rotated_residue a s t hs ht
  have hn : 0<gn := by omega
  have hw : 0<8*gn := by omega
  have hres : c*s+u<8*gn := by
    have hcs : (c+1)*s≤gn*s := Nat.mul_le_mul_right s (by omega)
    have hsg : gn*s≤gn*8 := Nat.mul_le_mul_left gn hs8
    nlinarith [hu.1]
  have hsmall : c*s+u<s*gn := by
    have hcs : (c+1)*s≤gn*s := Nat.mul_le_mul_right s (by omega)
    nlinarith [hu.1]
  have hbound : a+c*s+u<gm*gn := by
    have hp : s*gn≤(gm-g*8)*gn := Nat.mul_le_mul_right gn hsr
    have he : gm-g*8+g*8=gm := by omega
    dsimp [a]
    nlinarith
  have hdiv : (a+c*s+u)/(8*gn)=g := by
    have he : a+c*s+u=g*(8*gn)+(c*s+u) := by dsimp [a]; omega
    rw [he,Nat.mul_comm g (8*gn),Nat.mul_add_div hw]
    simp [Nat.div_eq_of_lt hres]
  have hmod : (a+c*s+u)%(8*gn)=c*s+u := by
    have he : a+c*s+u=g*(8*gn)+(c*s+u) := by dsimp [a]; omega
    rw [he,Nat.mul_add_mod']
    exact Nat.mod_eq_of_lt hres
  have hgs : groupSize gm gn (a+c*s+u)=s := by
    simp only [groupSize,groupId,groupWidth,hdiv]
    rfl
  have hrow : (a+c*s+u)%s=t := by
    have he : a+c*s+u=c*s+(a+u) := by omega
    rw [he,Nat.mul_add_mod']
    exact hu.2
  have hcol : (c*s+u)/s=c := by
    rw [Nat.mul_comm c s,Nat.mul_add_div hs]
    simp [Nat.div_eq_of_lt hu.1]
  change a+c*s+u<gm*gn ∧ _
  refine ⟨hbound,?_,?_⟩
  · simp only [tileRow,groupId,groupWidth]
    change (a+c*s+u)/(8*gn)*8+(a+c*s+u)%groupSize gm gn (a+c*s+u)=r
    rw [hdiv,hgs,hrow,hg]
  · simp only [tileCol,groupWidth]
    change (a+c*s+u)%(8*gn)/groupSize gm gn (a+c*s+u)=c
    rw [hmod,hgs,hcol]

theorem tile_bounds (gm gn pid : Nat) (hn : 0<gn) (hp : pid<gm*gn) :
    0<groupSize gm gn pid ∧ tileRow gm gn pid<gm ∧ tileCol gm gn pid<gn := by
  let g := pid/(8*gn)
  let q := pid%(8*gn)
  have hw : 0<8*gn := by omega
  have hdecomp : g*(8*gn)+q=pid := by simpa [g,q] using Nat.div_add_mod' pid (8*gn)
  have hq : q<8*gn := Nat.mod_lt _ hw
  have hg : g*8<gm := by nlinarith
  have hs : 0<min (gm-g*8) 8 := by omega
  have hr : pid%min (gm-g*8) 8<min (gm-g*8) 8 := Nat.mod_lt _ hs
  have hsle := Nat.min_le_left (gm-g*8) 8
  have hc : q<min (gm-g*8) 8*gn := by
    by_cases he : gm-g*8≤8
    · rw [Nat.min_eq_left he]
      have hsub : gm-g*8+g*8=gm := by omega
      nlinarith
    · rw [Nat.min_eq_right (by omega)]
      exact hq
  refine ⟨hs,?_,?_⟩
  · change g*8+pid%min (gm-g*8) 8<gm
    omega
  · change q/min (gm-g*8) 8<gn
    exact (Nat.div_lt_iff_lt_mul hs).2 (by simpa [Nat.mul_comm] using hc)

theorem tile_injective (gm gn p q : Nat) (hn : 0<gn)
    (hp : p<gm*gn) (hq : q<gm*gn)
    (hr : tileRow gm gn p=tileRow gm gn q)
    (hc : tileCol gm gn p=tileCol gm gn q) : p=q := by
  obtain ⟨hsp,_,_⟩ := tile_bounds gm gn p hn hp
  obtain ⟨hsq,_,_⟩ := tile_bounds gm gn q hn hq
  have hmp := Nat.mod_lt p hsp
  have hmq := Nat.mod_lt q hsq
  have h8p : groupSize gm gn p≤8 := Nat.min_le_right _ _
  have h8q : groupSize gm gn q≤8 := Nat.min_le_right _ _
  have hg : groupId gn p=groupId gn q := by unfold tileRow at hr; omega
  have hs : groupSize gm gn p=groupSize gm gn q := by simp only [groupSize,hg]
  have he : p%groupSize gm gn q=q%groupSize gm gn q := by unfold tileRow at hr; rw [hg,hs] at hr; omega
  have hdp := Nat.div_add_mod p (groupWidth gn)
  have hdq := Nat.div_add_mod q (groupWidth gn)
  change groupWidth gn*groupId gn p+p%groupWidth gn=p at hdp
  change groupWidth gn*groupId gn q+q%groupWidth gn=q at hdq
  rw [hg] at hdp
  have hcancel : (p%groupWidth gn)%groupSize gm gn q=(q%groupWidth gn)%groupSize gm gn q := by
    apply Nat.ModEq.add_left_cancel' (groupWidth gn*groupId gn q)
    change (groupWidth gn*groupId gn q+p%groupWidth gn)%groupSize gm gn q=
      (groupWidth gn*groupId gn q+q%groupWidth gn)%groupSize gm gn q
    rw [hdp,hdq]
    exact he
  unfold tileCol at hc
  rw [hs] at hc
  have hpdecomp := Nat.div_add_mod (p%groupWidth gn) (groupSize gm gn q)
  have hqdecomp := Nat.div_add_mod (q%groupWidth gn) (groupSize gm gn q)
  rw [hc,hcancel] at hpdecomp
  omega

theorem coordinate_preimage (m n r c : Nat) (hr : r<m) (hc : c<n) :
    ∃ pid<gridM m*gridN n, ∃ lane<128*128,
      row m n pid lane=r ∧ col m n pid lane=c := by
  have hrt : r/128<gridM m := by dsimp [gridM]; omega
  have hct : c/128<gridN n := by dsimp [gridN]; omega
  obtain ⟨hp,hm,hn⟩ := inverse_tile (gridM m) (gridN n) (r/128) (c/128) hrt hct
  refine ⟨inversePid (gridM m) (gridN n) (r/128) (c/128),hp,r%128*128+c%128,?_,?_,?_⟩
  · omega
  · dsimp [row]; rw [hm]; omega
  · dsimp [col]; rw [hn]; omega

theorem encode_decode (m r c : Nat) (hr : r<m) :
    (c*m+r)%m=r ∧ (c*m+r)/m=c := by
  have hm : 0<m := by omega
  constructor
  · rw [Nat.mul_add_mod',Nat.mod_eq_of_lt hr]
  · rw [Nat.mul_comm c m,Nat.mul_add_div hm,Nat.div_eq_of_lt hr]; omega

theorem decode_bounds (m n i : Nat) (hi : i<m*n) :
    i%m<m ∧ i/m<n ∧ i/m*m+i%m=i := by
  have hm : 0<m := by nlinarith
  exact ⟨Nat.mod_lt _ hm,(Nat.div_lt_iff_lt_mul hm).2 (by simpa [Nat.mul_comm] using hi),Nat.div_add_mod' i m⟩

theorem active_encode_bound (m n r c : Nat) (hr : r<m) (hc : c<n) : c*m+r<m*n := by
  have hcm : (c+1)*m≤n*m := Nat.mul_le_mul_right m (by omega)
  nlinarith

#axiomsClean inverse_tile
#axiomsClean tile_bounds
#axiomsClean tile_injective
#axiomsClean coordinate_preimage
#guard_msgs (drop info) in
#auditModuleAxioms
end QuantTransposeFinite
