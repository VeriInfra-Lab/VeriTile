import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Layout
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
open VeriTile.Triton Full

structure Strides where
  (batch head row dim : Nat)
def qStrides (p : Parameters) : Strides := ⟨p.qBatchStride,p.qHeadStride,p.qRowStride,p.qColStride⟩
def kStrides (p : Parameters) : Strides := ⟨p.kBatchStride,p.kHeadStride,p.kSeqStride,p.kDimStride⟩
def vStrides (p : Parameters) : Strides := ⟨p.vBatchStride,p.vHeadStride,p.vSeqStride,p.vDimStride⟩
def oStrides (p : Parameters) : Strides := ⟨p.outBatchStride,p.outHeadStride,p.outRowStride,p.outColStride⟩
def bStrides (p : Parameters) : Strides := ⟨p.biasBatchStride,p.biasHeadStride,p.biasRowStride,1⟩
def offset (s : Strides) (b h n d : Nat) := b*s.batch+h*s.head+n*s.row+d*s.dim

def Nonempty (p : Parameters) : Prop := 0<p.batches ∧ 0<p.heads ∧ 0<p.length
structure Selected (p : Parameters) : Prop where
  blockM : p.blockM∈[16,32,64,128]
  blockN : p.blockN∈[16,32,64,128]
  head : p.head∈[16,32,64,128]
  biasLast : p.biasLast=64
  biasNumel : p.biasNumel=128

theorem Selected.blockM_pos (p : Parameters) (s : Selected p) : 0<p.blockM := by
  have h:=s.blockM
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  omega

theorem Selected.blockN_pos (p : Parameters) (s : Selected p) : 0<p.blockN := by
  have h:=s.blockN
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  omega

theorem Selected.head_pos (p : Parameters) (s : Selected p) : 0<p.head := by
  have h:=s.head
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  omega

def extentAt (p : Parameters) : Fin 4 → Nat := ![p.batches,p.heads,p.length,p.head]
def strideAt (s : Strides) : Fin 4 → Nat := ![s.batch,s.head,s.row,s.dim]
def coordinate (b h n d : Nat) : Fin 4 → Nat := ![b,h,n,d]

structure OutLayout (p : Parameters) where
  order : List (Fin 4)
  permutation : order.Perm (List.finRange 4)
  guard : Layout.Guarded (extentAt p) (strideAt (oStrides p)) order

theorem outputInjective (p : Parameters) (layout : OutLayout p)
    (b h n d b' h' n' d' : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n<p.length) (hd : d<p.head)
    (hb' : b'<p.batches) (hh' : h'<p.heads) (hn' : n'<p.length) (hd' : d'<p.head)
    (he : offset (oStrides p) b h n d=offset (oStrides p) b' h' n' d') :
    b=b' ∧ h=h' ∧ n=n' ∧ d=d' := by
  have perm (x : Fin 4 → Nat) : Layout.offset (strideAt (oStrides p)) layout.order x=
      Layout.offset (strideAt (oStrides p)) (List.finRange 4) x :=
    List.Perm.sum_eq (layout.permutation.map _)
  have he' : Layout.offset (strideAt (oStrides p)) layout.order (coordinate b h n d)=
      Layout.offset (strideAt (oStrides p)) layout.order (coordinate b' h' n' d') := by
    rw [perm,perm]
    simpa [Layout.offset,List.finRange_succ,strideAt,coordinate,offset,Nat.add_assoc] using he
  have hx : ∀ a∈layout.order,coordinate b h n d a<extentAt p a := by
    intro a _
    fin_cases a <;> simp [coordinate,extentAt,hb,hh,hn,hd]
  have hy : ∀ a∈layout.order,coordinate b' h' n' d' a<extentAt p a := by
    intro a _
    fin_cases a <;> simp [coordinate,extentAt,hb',hh',hn',hd']
  have hi:=Layout.injective (extentAt p) (strideAt (oStrides p)) layout.order layout.guard _ _ hx hy he'
  have mem (a : Fin 4) : a∈layout.order := layout.permutation.mem_iff.mpr (List.mem_finRange _)
  exact ⟨hi 0 (mem 0),hi 1 (mem 1),hi 2 (mem 2),hi 3 (mem 3)⟩

def padded (p : Parameters) : Nat :=
  max ((p.length+p.blockM-1)/p.blockM*p.blockM) ((p.length+p.blockN-1)/p.blockN*p.blockN)
def strideEnvelope (p : Parameters) (s : Strides) : Nat :=
  (p.batches-1)*s.batch+(p.heads-1)*s.head+padded p*s.row+(p.head-1)*s.dim
def biasEnvelope (p : Parameters) : Nat :=
  (p.batches-1)*p.biasBatchStride+(p.heads-1)*p.biasHeadStride+(padded p-1)*p.biasRowStride+127
structure Guard (p : Parameters) where
  selected : Selected p
  length : p.length≤64*p.blockN
  layout : Nonempty p → OutLayout p
  launchY : Nonempty p → p.batches*p.heads≤65535
  positive : Nonempty p → ∀ s∈[qStrides p,kStrides p,vStrides p,oStrides p,bStrides p],
    0<s.batch ∧ 0<s.head ∧ 0<s.row ∧ 0<s.dim
  arguments : Nonempty p → ∀ s∈[qStrides p,kStrides p,vStrides p,oStrides p,bStrides p],
    ∀ v∈[p.batches,p.heads,p.length,p.head,s.batch,s.head,s.row,s.dim],v<2147483648
  signed : Nonempty p → ∀ s∈[qStrides p,kStrides p,vStrides p,oStrides p],strideEnvelope p s<2147483648
  biasSigned : Nonempty p → biasEnvelope p<2147483648

/-- Finite ordinary Tensor storage facts. Output injectivity is separately
proved from the wrapper's span guard, not supplied by this contract. -/
structure TensorStorage (p : Parameters) where
  (sizeQ sizeK sizeV sizeBias sizeOut : Nat)
  query : ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,offset (qStrides p) b h n d<sizeQ
  key : ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,offset (kStrides p) b h n d<sizeK
  value : ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,offset (vStrides p) b h n d<sizeV
  bias : ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ j<128,offset (bStrides p) b h n j<sizeBias
  output : ∀ b<p.batches,∀ h<p.heads,∀ n<p.length,∀ d<p.head,offset (oStrides p) b h n d<sizeOut
  separated : Nonempty p → p.output≠p.q ∧ p.output≠p.k ∧ p.output≠p.v ∧ p.output≠p.bias

theorem formed_offset_le (p : Parameters) (s : Strides) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n≤padded p) (hd : d<p.head) :
    offset s b h n d≤strideEnvelope p s := by
  have h0:=Nat.mul_le_mul_right s.batch (show b≤p.batches-1 by omega)
  have h1:=Nat.mul_le_mul_right s.head (show h≤p.heads-1 by omega)
  have h2:=Nat.mul_le_mul_right s.row hn
  have h3:=Nat.mul_le_mul_right s.dim (show d≤p.head-1 by omega)
  simp only [offset,strideEnvelope]
  omega

theorem sourceintegerbounds (p : Parameters) (g : Guard p) (ne : Nonempty p)
    (s : Strides) (hs : s∈[qStrides p,kStrides p,vStrides p,oStrides p]) (b h n d : Nat)
    (hb : b<p.batches) (hh : h<p.heads) (hn : n≤padded p) (hd : d<p.head) :
    offset s b h n d<2147483648 ∧ 0≤(offset s b h n d : Int) ∧ (offset s b h n d : Int)<2147483648 := by
  have bound:=lt_of_le_of_lt (formed_offset_le p s b h n d hb hh hn hd) (g.signed ne s hs)
  exact ⟨bound,by omega,by exact_mod_cast bound⟩

theorem biasintegerbounds (p : Parameters) (g : Guard p) (ne : Nonempty p)
    (b h n j : Nat) (hb : b<p.batches) (hh : h<p.heads) (hn : n<padded p) (hj : j≤127) :
    offset (bStrides p) b h n j<2147483648 := by
  have h0:=Nat.mul_le_mul_right p.biasBatchStride (show b≤p.batches-1 by omega)
  have h1:=Nat.mul_le_mul_right p.biasHeadStride (show h≤p.heads-1 by omega)
  have h2:=Nat.mul_le_mul_right p.biasRowStride (show n≤padded p-1 by omega)
  have hs:=g.biasSigned ne
  simp only [offset,bStrides,biasEnvelope] at *
  omega

theorem blocks_capacity (p : Parameters) (g : Guard p) : p.blocks≤64 := by
  have hn:=g.length
  have h:=g.selected.blockN
  simp only [List.mem_cons,List.not_mem_nil,or_false] at h
  rcases h with h|h|h|h <;> simp only [Parameters.blocks,h] <;> omega

theorem query_lane_bound (p : Parameters) (pid r : Nat)
    (hp : pid<(p.length+p.blockM-1)/p.blockM) (hr : r<p.blockM) :
    pid*p.blockM+r<padded p := by
  have h:=Nat.mul_le_mul_right p.blockM (show pid+1≤(p.length+p.blockM-1)/p.blockM by omega)
  have hm:=Nat.le_max_left ((p.length+p.blockM-1)/p.blockM*p.blockM) ((p.length+p.blockN-1)/p.blockN*p.blockN)
  simp only [Nat.add_mul,Nat.one_mul] at h
  unfold padded
  omega

theorem loop_lane_bound (p : Parameters) (c j : Nat) (hc : c<p.blocks) (hj : j<p.blockN) :
    c*p.blockN+j<padded p ∧ (c+1)*p.blockN≤padded p := by
  have h:=Nat.mul_le_mul_right p.blockN (show c+1≤p.blocks by omega)
  have hm:=Nat.le_max_right ((p.length+p.blockM-1)/p.blockM*p.blockM) ((p.length+p.blockN-1)/p.blockN*p.blockN)
  simp only [Parameters.blocks,Nat.add_mul,Nat.one_mul] at h
  unfold padded
  simp only [Nat.add_mul,Nat.one_mul]
  omega

theorem head_decode (p : Parameters) (hz : Nat) (hhz : hz<p.batches*p.heads) :
    0<p.heads ∧ hz/p.heads<p.batches ∧ hz%p.heads<p.heads := by
  have positive : 0<p.heads := by
    by_contra h
    have he : p.heads=0 := by omega
    simp [he] at hhz
  exact ⟨positive,(Nat.div_lt_iff_lt_mul positive).mpr hhz,Nat.mod_lt _ positive⟩

#axiomsClean outputInjective
#axiomsClean sourceintegerbounds
#axiomsClean biasintegerbounds
#specNonCircular Guard avoiding [FP.Structural.exec]
#specNonCircular TensorStorage avoiding [FP.Structural.exec]
end VeriTile.Bench.Repairs.AttentionKernelAligned.Whole
