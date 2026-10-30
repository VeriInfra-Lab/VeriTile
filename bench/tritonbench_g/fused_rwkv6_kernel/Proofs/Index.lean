import bench.tritonbench_g.fused_rwkv6_kernel.Proofs.Reference
namespace VeriTile.Bench.Repairs.FusedRwkv6Kernel
noncomputable section
set_option maxHeartbeats 4000000

structure Sizes (c : Shape) : Prop where
  B : 0<c.B
  H : 0<c.H
  K : 0<c.K
  V : 0<c.V
  BK : 0<c.BK
  BV : 0<c.BV

def Valid (c : Shape) (p : Program) : Prop := p.z<c.Z ∧ p.ik<c.NK ∧ p.iv<c.NV

def Output (c : Shape) (r : Region) (b : Nat) : Prop :=
  (r=.o ∧ b<c.NK*c.Z*c.T*c.V) ∨ (c.final ∧ r=.ht ∧ b<c.Z*c.K*c.V)

def PartialWrite (c : Shape) (p : Program) (n : Nat) (r : Region) (b : Nat) : Prop :=
  r=.o ∧ ∃ l<c.BV, valueIndex c p l<c.V ∧
    b=partialAddress c p.z p.ik (visit c n) (valueIndex c p l)

def FinalWrite (c : Shape) (p : Program) (r : Region) (b : Nat) : Prop :=
  c.final ∧ r=.ht ∧ ∃ l<c.BV, ∃ j<c.BK, valueIndex c p l<c.V ∧
    keyIndex c p j<c.K ∧ b=stateAddress c p.z (keyIndex c p j) (valueIndex c p l)

def Writes (c : Shape) (p : Program) (r : Region) (b : Nat) : Prop :=
  (∃ n<c.T, PartialWrite c p n r b) ∨ FinalWrite c p r b

theorem visit_lt (c : Shape) {n : Nat} (hn : n<c.T) : visit c n<c.T := by
  unfold visit
  split <;> omega

theorem visit_visit (c : Shape) {n : Nat} (hn : n<c.T) : visit c (visit c n)=n := by
  unfold visit
  split <;> omega

theorem visit_inj (c : Shape) {n m : Nat} (hn : n<c.T) (hm : m<c.T)
    (h : visit c n=visit c m) : n=m := by
  have := congrArg (visit c) h
  simpa [visit_visit c hn,visit_visit c hm] using this

theorem div_mod_pair (a d b : Nat) (hb : b<d) :
    (a*d+b)/d=a ∧ (a*d+b)%d=b := by
  have hd : 0<d := by omega
  constructor <;> simp [Nat.add_div,Nat.add_mod,Nat.div_eq_of_lt hb,Nat.mod_eq_of_lt hb,hd,hb]

theorem result_decode (c : Shape) (z t f : Nat) (ht : t<c.T) (hf : f<c.V) :
    resultAddress c z t f/c.V/c.T=z ∧ resultAddress c z t f/c.V%c.T=t ∧
    resultAddress c z t f%c.V=f := by
  have a := div_mod_pair (z*c.T+t) c.V f hf
  have b := div_mod_pair z c.T t ht
  simp only [resultAddress,a.1,a.2,b.1,b.2,and_self]

theorem partial_decode (c : Shape) (z i t f : Nat) (hz : z<c.Z) (ht : t<c.T) (hf : f<c.V) :
    partialAddress c z i t f/c.V/c.T%c.Z=z ∧
    partialAddress c z i t f/c.V/c.T/c.Z=i ∧
    partialAddress c z i t f/c.V%c.T=t ∧ partialAddress c z i t f%c.V=f := by
  have a := div_mod_pair ((i*c.Z+z)*c.T+t) c.V f hf
  have b := div_mod_pair (i*c.Z+z) c.T t ht
  have d := div_mod_pair i c.Z z hz
  simp only [partialAddress,a.1,a.2,b.1,b.2,d.1,d.2,and_self]

theorem state_decode (c : Shape) (z e f : Nat) (he : e<c.K) (hf : f<c.V) :
    stateAddress c z e f/c.V/c.K=z ∧ stateAddress c z e f/c.V%c.K=e ∧
    stateAddress c z e f%c.V=f := by
  have a := div_mod_pair (z*c.K+e) c.V f hf
  have b := div_mod_pair z c.K e he
  simp only [stateAddress,a.1,a.2,b.1,b.2,and_self]

theorem partial_lane (c : Shape) (p : Program) (n l : Nat) :
    partialAddress c p.z p.ik (visit c n) (valueIndex c p l) -
      partialAddress c p.z p.ik (visit c n) (p.iv*c.BV)=l := by
  unfold partialAddress valueIndex
  omega

theorem partial_bound (c : Shape) (p : Program) (hp : Valid c p) (n l : Nat)
    (hn : n<c.T) (hf : valueIndex c p l<c.V) :
    partialAddress c p.z p.ik (visit c n) (valueIndex c p l)<c.NK*c.Z*c.T*c.V := by
  have ht := visit_lt c hn
  have h1 : p.ik*c.Z+p.z<c.NK*c.Z := by nlinarith [hp.1,hp.2.1]
  have h2 : (p.ik*c.Z+p.z)*c.T+visit c n<c.NK*c.Z*c.T := by nlinarith
  dsimp [partialAddress]
  nlinarith

theorem state_bound (c : Shape) (p : Program) (hp : Valid c p) (e f : Nat)
    (he : e<c.K) (hf : f<c.V) : stateAddress c p.z e f<c.Z*c.K*c.V := by
  have h : p.z*c.K+e<c.Z*c.K := by nlinarith [hp.1]
  dsimp [stateAddress]
  nlinarith

theorem writes_output (c : Shape) (p : Program) (hp : Valid c p) {r : Region} {b : Nat}
    (h : Writes c p r b) : Output c r b := by
  rcases h with ⟨n,hn,rfl,l,hl,hf,rfl⟩ | ⟨hs,rfl,l,hl,j,hj,hf,he,rfl⟩
  · exact Or.inl ⟨rfl,partial_bound c p hp n l hn hf⟩
  · exact Or.inr ⟨hs,rfl,state_bound c p hp _ _ he hf⟩

theorem mem_programs (c : Shape) (p : Program) : p ∈ programs c ↔ Valid c p := by
  rcases p with ⟨z,i,v⟩
  simp [programs,Valid,List.mem_flatMap,List.mem_map,Program.mk.injEq]

/-- The unique grid owner reconstructed solely from the stored address. -/
def owner (c : Shape) (r : Region) (b : Nat) : Program :=
  if r=.ht then ⟨b/c.V/c.K,(b/c.V%c.K)/c.BK,(b%c.V)/c.BV⟩
  else ⟨b/c.V/c.T%c.Z,b/c.V/c.T/c.Z,(b%c.V)/c.BV⟩

theorem writes_owner (c : Shape) (p : Program) (hp : Valid c p) {r : Region} {b : Nat}
    (h : Writes c p r b) : owner c r b=p := by
  rcases h with ⟨n,hn,rfl,l,hl,hf,rfl⟩ | ⟨hs,rfl,l,hl,j,hj,hf,he,rfl⟩
  · have dec := partial_decode c p.z p.ik (visit c n) (valueIndex c p l) hp.1 (visit_lt c hn) hf
    have v := div_mod_pair p.iv c.BV l hl
    simp only [owner,reduceCtorEq,if_false,dec.1,dec.2.1,dec.2.2.2]
    simp only [valueIndex,v.1]
  · have dec := state_decode c p.z (keyIndex c p j) (valueIndex c p l) he hf
    have v := div_mod_pair p.iv c.BV l hl
    have k := div_mod_pair p.ik c.BK j hj
    simp only [owner,if_true,dec.1,dec.2.1,dec.2.2]
    simp only [keyIndex,valueIndex,v.1,k.1]

theorem unique_owner (c : Shape) (p q : Program) (hp : Valid c p) (hq : Valid c q)
    {r : Region} {b : Nat} (a : Writes c p r b) (d : Writes c q r b) : p=q :=
  (writes_owner c p hp a).symm.trans (writes_owner c q hq d)

theorem ceil_cover (d b : Nat) (hb : 0<b) : d ≤ ((d+b-1)/b)*b := by
  have := Nat.mod_lt (d+b-1) hb
  have hx := Nat.mod_add_div (d+b-1) b
  rw [Nat.mul_comm b] at hx
  omega

theorem ceil_member (d b x : Nat) (hb : 0<b) (hx : x<d) : x/b < (d+b-1)/b := by
  have hd := ceil_cover d b hb
  exact (Nat.div_lt_iff_lt_mul hb).2 (by nlinarith)

theorem reconstruct (a d : Nat) : a/d*d+a%d=a := by
  simpa [Nat.mul_comm] using Nat.div_add_mod a d

theorem cover (c : Shape) (hc : Sizes c) {r : Region} {b : Nat} (h : Output c r b) :
    ∃ p ∈ programs c, Writes c p r b := by
  rcases h with ⟨rfl,hb⟩ | ⟨hs,rfl,hb⟩
  · have hv : b%c.V<c.V := Nat.mod_lt _ hc.V
    have ht0 : 0<c.T := by
      by_contra h
      have he : c.T=0 := by omega
      simp [he] at hb
    have hz0 : 0<c.Z := Nat.mul_pos hc.B hc.H
    have ht : b/c.V%c.T<c.T := Nat.mod_lt _ ht0
    have hz : b/c.V/c.T%c.Z<c.Z := Nat.mod_lt _ hz0
    have hi : b/c.V/c.T/c.Z<c.NK := by
      rw [Nat.div_div_eq_div_mul,Nat.div_div_eq_div_mul]
      apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos hc.V (Nat.mul_pos ht0 hz0))).2
      nlinarith only [hb]
    let p : Program := ⟨b/c.V/c.T%c.Z,b/c.V/c.T/c.Z,(b%c.V)/c.BV⟩
    refine ⟨p,(mem_programs c p).2 ⟨hz,hi,ceil_member c.V c.BV _ hc.BV hv⟩,Or.inl ?_⟩
    refine ⟨visit c (b/c.V%c.T),visit_lt c ht,rfl,(b%c.V)%c.BV,Nat.mod_lt _ hc.BV,?_,?_⟩
    · simpa [valueIndex,p,reconstruct] using hv
    · simp only [valueIndex,p,visit_visit c ht]
      simp only [reconstruct]
      unfold partialAddress
      simp only [reconstruct]
  · have hv : b%c.V<c.V := Nat.mod_lt _ hc.V
    have he : b/c.V%c.K<c.K := Nat.mod_lt _ hc.K
    have hz : b/c.V/c.K<c.Z := by
      rw [Nat.div_div_eq_div_mul]
      apply (Nat.div_lt_iff_lt_mul (Nat.mul_pos hc.V hc.K)).2
      nlinarith only [hb]
    let p : Program := ⟨b/c.V/c.K,(b/c.V%c.K)/c.BK,(b%c.V)/c.BV⟩
    refine ⟨p,(mem_programs c p).2 ⟨hz,ceil_member c.K c.BK _ hc.BK he,
      ceil_member c.V c.BV _ hc.BV hv⟩,Or.inr ?_⟩
    refine ⟨hs,rfl,(b%c.V)%c.BV,Nat.mod_lt _ hc.BV,
      (b/c.V%c.K)%c.BK,Nat.mod_lt _ hc.BK,?_,?_,?_⟩
    · simpa [valueIndex,p,reconstruct] using hv
    · simpa [keyIndex,p,reconstruct] using he
    · simp only [valueIndex,keyIndex,p,reconstruct]
      unfold stateAddress
      simp only [reconstruct]

end
end VeriTile.Bench.Repairs.FusedRwkv6Kernel
