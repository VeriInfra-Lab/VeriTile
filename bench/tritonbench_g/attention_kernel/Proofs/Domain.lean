import bench.tritonbench_g.attention_kernel.Proofs.Engine
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Whole
open VeriTile Triton FP.Structural Source

def strideEnvelope (p : Parameters) (s : Strides) : Nat :=
  p.batches*s.batch+p.heads*s.head+(p.length+p.bm+p.bn)*s.row+p.dim*s.dim+128

def envelope (p : Parameters) : Nat :=
  p.batches+p.heads+p.length+p.dim+p.bm+p.bn+256+p.batches*p.heads+
  strideEnvelope p p.q+strideEnvelope p p.k+strideEnvelope p p.v+
  strideEnvelope p p.o+strideEnvelope p p.bias

structure DenseRows (p : Parameters) (s : Strides) : Prop where
  row : s.row=p.dim
  dim : s.dim=1
  head : p.length*p.dim≤s.head
  batch : p.heads*s.head≤s.batch

structure Guard (p : Parameters) : Prop where
  bm : 0<p.bm
  bn : 0<p.bn
  dim : 0<p.dim
  keyBlocks : (p.length+p.bn-1)/p.bn≤128
  signed : envelope p<2147483648
  launchY : p.batches*p.heads≤65535
  dense : ∀ s∈[p.q,p.k,p.v,p.o],DenseRows p s

/-- Ordinary descriptor/allocation facts. Each input reads its own supplied
logical view and the output is the caller's separate nonoverlapping storage. -/
structure TensorStorage (p : Parameters) where
  capacity : RegionName→Nat
  bounds : ∀ r s,(r,s)∈[(p.Q,p.q),(p.K,p.k),(p.V,p.v),(p.Out,p.o)] →
    ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.dim,
      b*s.batch+h*s.head+n*s.row+d*s.dim<capacity r
  bias : ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ c<128,
    b*p.bias.batch+h*p.bias.head+n*p.bias.row+c<capacity p.Bias
  separated : 0<p.batches ∧ 0<p.heads ∧ 0<p.length →
    p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.Bias

theorem stride_term_lt (p : Parameters) (s : Strides)
    (hs : s∈[p.q,p.k,p.v,p.o,p.bias]) : strideEnvelope p s<envelope p := by
  simp only [List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl <;> simp only [envelope] <;> omega

theorem formed_offset_le (p : Parameters) (s : Strides) (b h n d : Nat)
    (hb : b≤p.batches) (hh : h≤p.heads) (hn : n≤p.length+p.bm+p.bn) (hd : d≤p.dim) :
    b*s.batch+h*s.head+n*s.row+d*s.dim+127≤strideEnvelope p s := by
  have h0:=Nat.mul_le_mul_right s.batch hb
  have h1:=Nat.mul_le_mul_right s.head hh
  have h2:=Nat.mul_le_mul_right s.row hn
  have h3:=Nat.mul_le_mul_right s.dim hd
  simp only [strideEnvelope]
  omega

theorem source_integer_bound (p : Parameters) (g : Guard p) (s : Strides)
    (hs : s∈[p.q,p.k,p.v,p.o,p.bias]) (b h n d : Nat)
    (hb : b≤p.batches) (hh : h≤p.heads) (hn : n≤p.length+p.bm+p.bn) (hd : d≤p.dim) :
    b*s.batch+h*s.head+n*s.row+d*s.dim+127<2147483648 :=
  lt_of_le_of_lt (formed_offset_le p s b h n d hb hh hn hd)
    (lt_trans (stride_term_lt p s hs) g.signed)

theorem head_decode (p : Parameters) (hz : Nat) (hhz : hz<p.batches*p.heads) :
    0<p.heads ∧ hz/p.heads<p.batches ∧ hz%p.heads<p.heads := by
  have pos : 0<p.heads := by
    by_contra h
    have he : p.heads=0 := by omega
    simp [he] at hhz
  exact ⟨pos,(Nat.div_lt_iff_lt_mul pos).mpr hhz,Nat.mod_lt _ pos⟩

theorem ceil_cover (N block : Nat) (hb : 0<block) :
    N≤(N+block-1)/block*block := by
  have rem:=Nat.mod_lt (N+block-1) hb
  have eqn:=Nat.div_add_mod (N+block-1) block
  rw [Nat.mul_comm block] at eqn
  omega

theorem query_lane_bound (N block pid lane : Nat) (hb : 0<block)
    (hp : pid<(N+block-1)/block) (hl : lane<block) : pid*block+lane<N+block := by
  have hprod:=Nat.mul_le_mul_right block (show pid+1≤(N+block-1)/block by omega)
  have hdiv:=Nat.div_mul_le_self (N+block-1) block
  rw [Nat.add_mul,Nat.one_mul] at hprod
  omega

theorem row_program_bound (N block n : Nat) (hb : 0<block) (hn : n<N) :
    n/block<(N+block-1)/block :=
  (Nat.div_lt_iff_lt_mul hb).mpr (lt_of_lt_of_le hn (ceil_cover N block hb))

theorem bias_index_bound (p : Parameters) (g : Guard p) (start : Nat) (hs : start<p.length) :
    start/p.bn<128 := by
  have h:=row_program_bound p.length p.bn start g.bn hs
  have := g.keyBlocks
  omega

theorem within_head (p : Parameters) (s : Strides) (layout : DenseRows p s)
    (n d : Nat) (hn : n<p.length) (hd : d<p.dim) : n*s.row+d*s.dim<s.head := by
  rw [layout.row,layout.dim,Nat.mul_one]
  have hmul:=Nat.mul_le_mul_right p.dim (show n+1≤p.length by omega)
  rw [Nat.add_mul,Nat.one_mul] at hmul
  have := layout.head
  omega

theorem within_batch (p : Parameters) (s : Strides) (layout : DenseRows p s)
    (h n d : Nat) (hh : h<p.heads) (hn : n<p.length) (hd : d<p.dim) :
    h*s.head+n*s.row+d*s.dim<s.batch := by
  have hcell:=within_head p s layout n d hn hd
  have hmul:=Nat.mul_le_mul_right s.head (show h+1≤p.heads by omega)
  rw [Nat.add_mul,Nat.one_mul] at hmul
  have:=layout.batch
  omega

theorem block_unique (pitch x y i j : Nat) (hi : i<pitch) (hj : j<pitch)
    (he : x*pitch+i=y*pitch+j) : x=y ∧ i=j := by
  have hp : 0<pitch := by omega
  have hx : (x*pitch+i)/pitch=x := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hi]
    omega
  have hy : (y*pitch+j)/pitch=y := by
    rw [Nat.add_comm,Nat.add_mul_div_right _ _ hp,Nat.div_eq_of_lt hj]
    omega
  have hxy : x=y := by
    have h:=congrArg (fun n=>n/pitch) he
    simpa only [hx,hy] using h
  exact ⟨hxy,by simpa only [hxy,Nat.add_left_cancel_iff] using he⟩

theorem dense_injective (p : Parameters) (s : Strides) (layout : DenseRows p s)
    (b h n d b' h' n' d' : Nat) (hh : h<p.heads) (hn : n<p.length) (hd : d<p.dim)
    (hh' : h'<p.heads) (hn' : n'<p.length) (hd' : d'<p.dim)
    (he : b*s.batch+h*s.head+n*s.row+d*s.dim=b'*s.batch+h'*s.head+n'*s.row+d'*s.dim) :
    b=b' ∧ h=h' ∧ n=n' ∧ d=d' := by
  obtain ⟨hb,ht⟩:=block_unique s.batch b b' (h*s.head+n*s.row+d*s.dim)
    (h'*s.head+n'*s.row+d'*s.dim)
    (within_batch p s layout h n d hh hn hd) (within_batch p s layout h' n' d' hh' hn' hd')
    (by simpa only [Nat.add_assoc] using he)
  obtain ⟨hh,hn⟩:=block_unique s.head h h' (n*s.row+d*s.dim) (n'*s.row+d'*s.dim)
    (within_head p s layout n d hn hd) (within_head p s layout n' d' hn' hd')
    (by simpa only [Nat.add_assoc] using ht)
  rw [layout.row,layout.dim,Nat.mul_one,Nat.mul_one] at hn
  obtain ⟨hn,hd⟩:=block_unique p.dim n n' d d' hd hd' hn
  exact ⟨hb,hh,hn,hd⟩

#axiomsClean source_integer_bound
#axiomsClean bias_index_bound
#specNonCircular TensorStorage avoiding [FP.Structural.run,FP.Structural.exec]
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Whole
