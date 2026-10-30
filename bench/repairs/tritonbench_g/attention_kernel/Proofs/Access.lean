import bench.repairs.tritonbench_g.attention_kernel.Proofs.Machine
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.Access
open VeriTile Triton Source Whole Machine

/-- All actual source reads and stores, including both bias windows and
masked query/key tails, fit the actual caller allocation capacities. -/
structure Safe (p : Parameters) (storage : TensorStorage p) : Prop where
  query : ∀ y<p.batches*p.heads,∀ m<p.length,∀ d<p.dim,
    offset p.q (y/p.heads) (y%p.heads) m d<storage.capacity p.Q
  keyValue : ∀ y<p.batches*p.heads,∀ row<p.length,∀ n<p.bn,∀ d<p.dim,
    n<p.length-row →
      offset p.k (y/p.heads) (y%p.heads) (row+n) d<storage.capacity p.K ∧
      offset p.v (y/p.heads) (y%p.heads) (row+n) d<storage.capacity p.V
  output : ∀ y<p.batches*p.heads,∀ m<p.length,∀ d<p.dim,
    offset p.o (y/p.heads) (y%p.heads) m d<storage.capacity p.Out
  biasDynamic : ∀ y<p.batches*p.heads,∀ m<p.length,∀ row<p.length,
    y/p.heads*p.bias.batch+y%p.heads*p.bias.head+m*p.bias.row+row/p.bn<storage.capacity p.Bias
  biasStatic : ∀ y<p.batches*p.heads,∀ m<p.length,∀ n<p.bn,
    y/p.heads*p.bias.batch+y%p.heads*p.bias.head+m*p.bias.row+(n%64+64)<storage.capacity p.Bias

theorem safe (p : Parameters) (guard : Guard p) (storage : TensorStorage p) : Safe p storage := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro y hy m hm d hd
    obtain ⟨_,hb,hh⟩:=head_decode p y hy
    exact storage.bounds p.Q p.q (by simp) _ hb _ hh m hm d hd
  · intro y hy row hr n hn d hd active
    obtain ⟨_,hb,hh⟩:=head_decode p y hy
    exact ⟨storage.bounds p.K p.k (by simp) _ hb _ hh _ (by omega) d hd,
      storage.bounds p.V p.v (by simp) _ hb _ hh _ (by omega) d hd⟩
  · intro y hy m hm d hd
    obtain ⟨_,hb,hh⟩:=head_decode p y hy
    exact storage.bounds p.Out p.o (by simp) _ hb _ hh m hm d hd
  · intro y hy m hm row hr
    obtain ⟨_,hb,hh⟩:=head_decode p y hy
    exact storage.bias _ hb _ hh m hm _ (bias_index_bound p guard row hr)
  · intro y hy m hm n hn
    obtain ⟨_,hb,hh⟩:=head_decode p y hy
    exact storage.bias _ hb _ hh m hm _ (by have:=Nat.mod_lt n (by decide : 0<64);omega)

#axiomsClean safe
#specNonCircular Safe avoiding [FP.Structural.exec,Whole.invoke]
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.Access
