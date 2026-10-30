import bench.tritonbench_g.attention_fwd_triton2.Proofs.Host
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Machine
open VeriTile Triton Source Whole
noncomputable section
set_option maxHeartbeats 4000000

/-- Canonical word strides. Empty returns need no launch representation. -/
def dense (p : Parameters) : Strides := ⟨p.heads*p.length*96,p.length*96,96,1⟩
def Live (p : Parameters) : Prop := 0<p.batches ∧ 0<p.heads ∧ 0<p.length
def Canonical (p : Parameters) : Prop :=
  Live p → p.q=dense p ∧ p.k=dense p ∧ p.v=dense p ∧ p.o=dense p

def words (p : Parameters) : Nat := p.batches*p.heads*p.length*96

def offset (s : Strides) (b h n d : Nat) : Nat :=
  b*s.batch+h*s.head+n*s.row+d*s.dim

/-- Each entry is an actual data_ptr and its scalar byte width. Alias among
read-only inputs is allowed. Freshness of output is an allocation condition. -/
structure Bases where
  (q k v qs ks out : Nat)

def baseWidths (b : Bases) : List (Nat × Nat) :=
  [(b.q,4),(b.k,4),(b.v,4),(b.qs,4),(b.ks,4),(b.out,2)]

structure Domain (p : Parameters) (b : Bases) : Prop where
  canonical : Canonical p
  launch : Live p → Guard p
  addresses : Live p → ∀ x∈baseWidths b,
    x.1%x.2=0 ∧ x.1+x.2*envelope p<18446744073709551616

/-- Dense geometry derives finite active access bounds; no access bound is a
numerical premise and no output equation appears in this domain. -/
theorem digit_bound (a A d D : Nat) (ha : a<A) (hd : d<D) : a*D+d<A*D := by
  have h:=Nat.mul_le_mul_right D (show a+1≤A by omega)
  rw [Nat.add_mul,Nat.one_mul] at h
  omega

theorem dense_bound (p : Parameters) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<96) :
    offset (dense p) b h n d<words p := by
  have h1:=digit_bound b p.batches h p.heads hb hh
  have h2:=digit_bound (b*p.heads+h) (p.batches*p.heads) n p.length h1 hn
  have h3:=digit_bound ((b*p.heads+h)*p.length+n) (p.batches*p.heads*p.length) d 96 h2 hd
  simpa [offset,dense,words,Nat.add_mul,Nat.mul_add,Nat.mul_assoc] using h3

theorem digits_injective (a a' d d' D : Nat) (hd : d<D) (hd' : d'<D)
    (he : a*D+d=a'*D+d') : a=a' ∧ d=d' := by
  have hp : 0<D := by omega
  have hq:=congrArg (fun n=>n/D) he
  have hm:=congrArg (fun n=>n%D) he
  have ha : a=a' := by
    simpa only [Nat.add_comm (a*D) d,Nat.add_comm (a'*D) d',
      Nat.add_mul_div_right d a hp,Nat.add_mul_div_right d' a' hp,
      Nat.div_eq_of_lt hd,Nat.div_eq_of_lt hd',Nat.zero_add] using hq
  exact ⟨ha,by simpa only [Nat.mul_add_mod_of_lt hd,Nat.mul_add_mod_of_lt hd'] using hm⟩

theorem dense_injective (p : Parameters) (b h n d b' h' n' d' : Nat)
    (hh : h<p.heads) (hn : n<p.length) (hd : d<96)
    (hh' : h'<p.heads) (hn' : n'<p.length) (hd' : d'<96)
    (he : offset (dense p) b h n d=offset (dense p) b' h' n' d') :
    b=b' ∧ h=h' ∧ n=n' ∧ d=d' := by
  have eq0 : ((b*p.heads+h)*p.length+n)*96+d=
      ((b'*p.heads+h')*p.length+n')*96+d' := by
    simpa [offset,dense,Nat.add_mul,Nat.mul_add,Nat.mul_assoc] using he
  obtain ⟨eq1,de⟩:=digits_injective _ _ _ _ 96 hd hd' eq0
  obtain ⟨eq2,ne⟩:=digits_injective _ _ _ _ p.length hn hn' eq1
  obtain ⟨be,hhe⟩:=digits_injective _ _ _ _ p.heads hh hh' eq2
  exact ⟨be,hhe,ne,de⟩

def storage (p : Parameters) (hc : Canonical p)
    (fresh : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS) : TensorStorage p where
  sizeQ:=words p
  sizeK:=words p
  sizeV:=words p
  sizeQS:=p.batches*p.heads*((p.length+127)/128)
  sizeKS:=p.batches*p.heads*((p.length+63)/64)
  sizeOut:=words p
  query:=by
    intro b hb h hh n hn d hd
    have hhcan:=(hc ⟨by omega,by omega,by omega⟩).1
    change offset p.q b h n d<words p
    rw [hhcan]
    exact dense_bound p b h n d hb hh hn hd
  key:=by
    intro b hb h hh n hn d hd
    have hhcan:=(hc ⟨by omega,by omega,by omega⟩).2.1
    change offset p.k b h n d<words p
    rw [hhcan]
    exact dense_bound p b h n d hb hh hn hd
  value:=by
    intro b hb h hh n hn d hd
    have hhcan:=(hc ⟨by omega,by omega,by omega⟩).2.2.1
    change offset p.v b h n d<words p
    rw [hhcan]
    exact dense_bound p b h n d hb hh hn hd
  scales:=⟨rfl,rfl⟩
  output:=by
    intro b hb h hh n hn d hd
    have hhcan:=(hc ⟨by omega,by omega,by omega⟩).2.2.2
    change offset p.o b h n d<words p
    rw [hhcan]
    exact dense_bound p b h n d hb hh hn hd
  outputInjective:=by
    intro b hb h hh n hn d hd b' hb' h' hh' n' hn' d' hd' he
    have hcan:=(hc ⟨by omega,by omega,by omega⟩).2.2.2
    change offset p.o b h n d=offset p.o b' h' n' d' at he
    rw [hcan] at he
    exact dense_injective p b h n d b' h' n' d' hh hn hd hh' hn' hd' he
  separated:=fresh

/-- Exact fixed-width interpretation of every nonnegative source subexpression
bounded by a formed offset. This includes the still-i32 loop and stride steps. -/
theorem repr32 (n : Nat) (h : n<2147483648) :
    (BitVec.ofInt 32 (n : Int)).toInt=(n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

theorem repr64 (n : Nat) (h : n<9223372036854775808) :
    (BitVec.ofInt 64 (n : Int)).toInt=(n : Int) := by
  apply BitVec.toInt_ofInt_eq_self (by decide) <;> norm_num <;> omega

structure Exact (n : Nat) : Prop where
  signed32 : n<2147483648
  int32 : (BitVec.ofInt 32 (n : Int)).toInt=(n : Int)
  int64 : (BitVec.ofInt 64 (n : Int)).toInt=(n : Int)
  subexpressions : ∀ x≤n,(BitVec.ofInt 32 (x : Int)).toInt=(x : Int) ∧
    (BitVec.ofInt 64 (x : Int)).toInt=(x : Int)

theorem exact_of_bound (n : Nat) (h : n<2147483648) : Exact n := by
  refine ⟨h,repr32 n h,repr64 n (by omega),?_⟩
  intro x hx
  exact ⟨repr32 x (by omega),repr64 x (by omega)⟩

theorem formed_exact (p : Parameters) (g : Guard p) (s : Strides) (hs : s∈[p.q,p.k,p.v,p.o])
    (b h n d : Nat) (hb : b≤p.batches) (hh : h≤p.heads)
    (hn : n≤p.length+128) (hd : d≤128) : Exact (offset s b h n d) :=
  exact_of_bound _ (sourceintegerbounds p g s hs b h n d hb hh hn hd).1

theorem source_coordinates (p : Parameters) (x y : Nat)
    (hx : x<(p.length+127)/128) (hy : y<p.batches*p.heads)
    (row r d : Nat) (hr : row<p.length) (hlane : r<128) (hd : d<128) :
    y/p.heads<p.batches ∧ y%p.heads<p.heads ∧
    x*128+r<p.length+128 ∧ row+r<p.length+128 ∧ row+64<p.length+128 := by
  have head:=head_decode p y hy
  have lane:=query_lane_bound p.length x r hx hlane
  have lo:=loop_lane_bound p.length row r hr hlane
  exact ⟨head.2.1,head.2.2,lane,lo.1,lo.2⟩

theorem byte_extent (p : Parameters) (b : Bases) (dom : Domain p b) (live : Live p)
    (pair : Nat × Nat) (hp : pair∈baseWidths b) (n : Nat) (hn : n<envelope p) :
    pair.2*n<9223372036854775808 ∧ pair.1+pair.2*n<18446744073709551616 ∧
    (BitVec.ofInt 64 ((pair.2*n : Nat) : Int)).toInt=((pair.2*n : Nat) : Int) := by
  have hg:=(dom.launch live).signed
  have hw : pair.2=4 ∨ pair.2=2 := by
    simp only [baseWidths,List.mem_cons,List.not_mem_nil,or_false] at hp
    rcases hp with rfl|rfl|rfl|rfl|rfl|rfl <;> simp
  have hdisp : pair.2*n<9223372036854775808 := by rcases hw with h|h <;> rw [h] <;> omega
  have physical:=(dom.addresses live pair hp).2
  have le:=Nat.mul_le_mul_left pair.2 (Nat.le_of_lt hn)
  exact ⟨hdisp,by omega,repr64 _ hdisp⟩

/-- Source formation (including inactive lanes and final advances) and every
smaller nonnegative product/addend are fixed-width exact. -/
structure Closed (p : Parameters) (b : Bases) : Prop where
  offsets : Live p → ∀ s∈[p.q,p.k,p.v,p.o],∀ z≤p.batches,∀ h≤p.heads,
    ∀ n≤p.length+128,∀ d≤128,Exact (offset s z h n d)
  scaleOffsets : Live p → ∀ hz<p.batches*p.heads,∀ block blocks,
    block≤blocks → (blocks=(p.length+127)/128 ∨ blocks=(p.length+63)/64) → Exact (hz*blocks+block)
  physical : Live p → ∀ pair∈baseWidths b,∀ n<envelope p,
    pair.2*n<9223372036854775808 ∧ pair.1+pair.2*n<18446744073709551616 ∧
    (BitVec.ofInt 64 ((pair.2*n : Nat) : Int)).toInt=((pair.2*n : Nat) : Int)

theorem closed (p : Parameters) (b : Bases) (dom : Domain p b) : Closed p b := by
  refine ⟨?_,?_,?_⟩
  · intro live s hs z hz h hh n hn d hd
    exact formed_exact p (dom.launch live) s hs z h n d hz hh hn hd
  · intro live hz hhz block blocks hb hblocks
    exact exact_of_bound _ (scale_integer_bound p (dom.launch live) hz block blocks hhz hb hblocks)
  · intro live pair hp n hn
    exact byte_extent p b dom live pair hp n hn

#axiomsClean storage
#axiomsClean closed
#specNonCircular Domain avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Machine
