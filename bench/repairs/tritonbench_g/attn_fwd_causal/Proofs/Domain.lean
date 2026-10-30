import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Engine
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Whole
open VeriTile.Triton Source

def strideEnvelope (p : Parameters) (s : Strides) : Nat :=
  p.batches*s.batch+p.heads*s.head+(p.length+128)*s.row+128*s.dim

def envelope (p : Parameters) : Nat :=
  p.batches+p.heads+p.length+256+p.batches*p.heads+
  (p.batches*p.heads+1)*((p.length+127)/128+(p.length+63)/64+2)+
  strideEnvelope p p.q+strideEnvelope p p.k+strideEnvelope p p.v+strideEnvelope p p.o

structure Guard (p : Parameters) : Prop where
  signed : 0<p.batches → 0<p.heads → 0<p.length → envelope p<2147483648
  launchY : p.batches*p.heads≤65535
  inputStrides : 0<p.batches → 0<p.heads → 0<p.length → ∀ s∈[p.q,p.k,p.v],
    0<s.batch ∧ 0<s.head ∧ 0<s.row ∧ 0<s.dim

/-- Ordinary Tensor storage facts and fresh empty_like output facts. These are
address/shape/allocation contracts only, with no numerical or execution premise. -/
structure TensorStorage (p : Parameters) where
  (sizeQ sizeK sizeV sizeQS sizeKS sizeOut : Nat)
  query : ∀ b<p.batches, ∀ h<p.heads, ∀ n<p.length, ∀ d<128,
    b*p.q.batch+h*p.q.head+n*p.q.row+d*p.q.dim<sizeQ
  key : ∀ b<p.batches, ∀ h<p.heads, ∀ n<p.length, ∀ d<128,
    b*p.k.batch+h*p.k.head+n*p.k.row+d*p.k.dim<sizeK
  value : ∀ b<p.batches, ∀ h<p.heads, ∀ n<p.length, ∀ d<128,
    b*p.v.batch+h*p.v.head+n*p.v.row+d*p.v.dim<sizeV
  scales : sizeQS=p.batches*p.heads*((p.length+127)/128) ∧
    sizeKS=p.batches*p.heads*((p.length+63)/64)
  output : ∀ b<p.batches, ∀ h<p.heads, ∀ n<p.length, ∀ d<128,
    b*p.o.batch+h*p.o.head+n*p.o.row+d*p.o.dim<sizeOut
  separated : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS

theorem two_digits (width a b a' b' : Nat) (hw : 0<width)
    (hb : b<width) (hb' : b'<width) (he : a*width+b=a'*width+b') : a=a' ∧ b=b' := by
  have left : (a*width+b)/width=a := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hw,Nat.div_eq_of_lt hb]
    omega
  have right : (a'*width+b')/width=a' := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hw,Nat.div_eq_of_lt hb']
    omega
  have quotient := congrArg (fun x=>x/width) he
  dsimp only at quotient
  rw [left,right] at quotient
  exact ⟨quotient,by rw [quotient] at he;omega⟩

/-- Canonical coordinate injectivity is derived; it is not an allocator or
user premise. Empty tensors have no active coordinate to prove. -/
theorem TensorStorage.outputInjective {p : Parameters} (_storage : TensorStorage p)
    (b : Nat) (_hb : b<p.batches) (h : Nat) (hh : h<p.heads)
    (n : Nat) (hn : n<p.length) (d : Nat) (hd : d<128)
    (b' : Nat) (_hb' : b'<p.batches) (h' : Nat) (hh' : h'<p.heads)
    (n' : Nat) (hn' : n'<p.length) (d' : Nat) (hd' : d'<128)
    (he : b*p.o.batch+h*p.o.head+n*p.o.row+d*p.o.dim=
      b'*p.o.batch+h'*p.o.head+n'*p.o.row+d'*p.o.dim) : b=b' ∧ h=h' ∧ n=n' ∧ d=d' := by
  have hH : 0<p.heads := by omega
  have hN : 0<p.length := by omega
  have packed : ((b*p.heads+h)*p.length+n)*128+d=
      ((b'*p.heads+h')*p.length+n')*128+d' := by
    simpa [Parameters.o,Parameters.q,max_eq_left (show 1≤p.heads by omega),
      max_eq_left (show 1≤p.length by omega),Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨major,feature⟩:=two_digits 128 _ d _ d' (by decide) hd hd' packed
  obtain ⟨heads,rows⟩:=two_digits p.length _ n _ n' hN hn hn' major
  obtain ⟨batches,heads⟩:=two_digits p.heads b h b' h' hH hh hh' heads
  exact ⟨batches,heads,rows,feature⟩

#axiomsClean TensorStorage.outputInjective

theorem stride_term_lt (p : Parameters) (s : Strides) (hs : s∈[p.q,p.k,p.v,p.o]) :
    strideEnvelope p s<envelope p := by
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl <;> simp only [envelope] <;> omega

theorem formed_offset_le (p : Parameters) (s : Strides) (b h n d : Nat)
    (hb : b≤p.batches) (hh : h≤p.heads) (hn : n≤p.length+128) (hd : d≤128) :
    b*s.batch+h*s.head+n*s.row+d*s.dim≤strideEnvelope p s := by
  have h0:=Nat.mul_le_mul_right s.batch hb
  have h1:=Nat.mul_le_mul_right s.head hh
  have h2:=Nat.mul_le_mul_right s.row hn
  have h3:=Nat.mul_le_mul_right s.dim hd
  simp only [strideEnvelope]
  omega

/-- Bounds apply to all formed offsets, including padded lanes and pointer
advances; masked-off dereferences are not used to justify integer arithmetic. -/
theorem sourceintegerbounds (p : Parameters) (g : Guard p)
    (liveB : 0<p.batches) (liveH : 0<p.heads) (liveN : 0<p.length)
    (s : Strides) (hs : s∈[p.q,p.k,p.v,p.o]) (b h n d : Nat)
    (hb : b≤p.batches) (hh : h≤p.heads) (hn : n≤p.length+128) (hd : d≤128) :
    b*s.batch+h*s.head+n*s.row+d*s.dim<2147483648 ∧
    0≤((b*s.batch+h*s.head+n*s.row+d*s.dim : Nat) : Int) ∧
    ((b*s.batch+h*s.head+n*s.row+d*s.dim : Nat) : Int)<2147483648 := by
  have bound:=lt_of_le_of_lt (formed_offset_le p s b h n d hb hh hn hd)
    (lt_trans (stride_term_lt p s hs) (g.signed liveB liveH liveN))
  exact ⟨bound,by omega,by exact_mod_cast bound⟩

theorem query_lane_bound (N pid r : Nat) (hp : pid<(N+127)/128) (hr : r<128) :
    pid*128+r<N+128 := by omega

theorem loop_lane_bound (N start n : Nat) (hs : start<N) (hn : n<128) :
    start+n<N+128 ∧ start+64<N+128 := by omega

theorem head_decode (p : Parameters) (hz : Nat) (hhz : hz<p.batches*p.heads) :
    0<p.heads ∧ hz/p.heads<p.batches ∧ hz%p.heads<p.heads := by
  have positive : 0<p.heads := by
    by_contra h
    have he : p.heads=0 := by omega
    simp [he] at hhz
  exact ⟨positive,(Nat.div_lt_iff_lt_mul positive).mpr hhz,Nat.mod_lt _ positive⟩

theorem scale_integer_bound (p : Parameters) (g : Guard p)
    (liveB : 0<p.batches) (liveH : 0<p.heads) (liveN : 0<p.length)
    (hz block blocks : Nat) (hhz : hz<p.batches*p.heads)
    (hb : block≤blocks) (hblocks : blocks=(p.length+127)/128 ∨ blocks=(p.length+63)/64) :
    hz*blocks+block<2147483648 := by
  have hm:=Nat.mul_le_mul_right blocks (show hz+1≤p.batches*p.heads by omega)
  rw [Nat.add_mul,Nat.one_mul] at hm
  have hsmall : p.batches*p.heads*blocks<envelope p := by
    rcases hblocks with rfl | rfl <;>
      simp only [envelope,Nat.add_mul,Nat.mul_add,Nat.one_mul,Nat.mul_one] <;> omega
  exact lt_trans (lt_of_le_of_lt (by omega) hsmall) (g.signed liveB liveH liveN)

/-- Every actual helper iteration of a legal query CTA has an existing key
block. This connects the dynamic bounds to the unmasked K-scale load. -/
theorem helper_key_bound (N pid row : Nat) (hp : pid<(N+127)/128)
    (hr : row<pid*128 ∨ (pid*128≤row ∧ row<min ((pid+1)*128) N)) :
    row<N ∧ row/64<(N+63)/64 := by omega

theorem helper_query_bound (N pid : Nat) (hp : pid<(N+127)/128) :
    pid*128<N ∧ (pid+1)*128<N+128 ∧ (pid*128)%64=0 := by omega

#axiomsClean helper_key_bound
#axiomsClean helper_query_bound

#axiomsClean sourceintegerbounds
#axiomsClean scale_integer_bound
#specNonCircular TensorStorage avoiding [FP.Structural.run,FP.Structural.exec]
end VeriTile.Bench.Repairs.AttnFwdCausal.Whole
