import bench.tritonbench_g.attention_fwd_triton2.Proofs.Machine
namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Access
open VeriTile Triton Source Whole Machine

/-- The five actual masked/unmasked input accesses and actual output stores
at every grid program and loop position, with finite region capacities. -/
structure Safe (p : Parameters) : Prop where
  query : ∀ x<(p.length+127)/128,∀ y<p.batches*p.heads,∀ r<128,∀ d<128,
    x*128+r<p.length → d<96 → offset p.q (y/p.heads) (y%p.heads) (x*128+r) d<words p
  keyValue : ∀ y<p.batches*p.heads,∀ row<p.length,∀ n<64,∀ d<128,
    n<p.length-row → d<96 →
      offset p.k (y/p.heads) (y%p.heads) (row+n) d<words p ∧
      offset p.v (y/p.heads) (y%p.heads) (row+n) d<words p
  output : ∀ x<(p.length+127)/128,∀ y<p.batches*p.heads,∀ r<128,∀ d<128,
    x*128+r<p.length → d<96 → offset p.o (y/p.heads) (y%p.heads) (x*128+r) d<words p
  queryScale : ∀ x<(p.length+127)/128,∀ y<p.batches*p.heads,
    y*((p.length+127)/128)+x<p.batches*p.heads*((p.length+127)/128)
  keyScale : ∀ y<p.batches*p.heads,∀ row<p.length,
    y*((p.length+63)/64)+row/64<p.batches*p.heads*((p.length+63)/64)

theorem safe (p : Parameters) (canon : Canonical p) : Safe p := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro x hx y hy r hr d hd hm hf
    obtain ⟨hH,hB,hh⟩:=head_decode p y hy
    have hc:=(canon ⟨by nlinarith,hH,by omega⟩).1
    rw [hc]
    exact dense_bound p _ _ _ _ hB hh hm hf
  · intro y hy row hrow n hn d hd hn' hd'
    obtain ⟨hH,hB,hh⟩:=head_decode p y hy
    have hc:=canon ⟨by nlinarith,hH,by omega⟩
    rw [hc.2.1,hc.2.2.1]
    have h:=dense_bound p (y/p.heads) (y%p.heads) (row+n) d hB hh (by omega) hd'
    exact ⟨h,h⟩
  · intro x hx y hy r hr d hd hm hf
    obtain ⟨hH,hB,hh⟩:=head_decode p y hy
    have hc:=(canon ⟨by nlinarith,hH,by omega⟩).2.2.2
    rw [hc]
    exact dense_bound p _ _ _ _ hB hh hm hf
  · intro x hx y hy
    exact digit_bound y (p.batches*p.heads) x ((p.length+127)/128) hy hx
  · intro y hy row hrow
    exact digit_bound y (p.batches*p.heads) (row/64) ((p.length+63)/64) hy (by omega)

theorem loop_and_grid_metadata (p : Parameters) (g : Guard p) :
    p.batches<2147483648 ∧ p.heads<2147483648 ∧ p.length+127<2147483648 ∧
    (p.length+127)/128<2147483648 ∧ (p.length+63)/64<2147483648 ∧ p.batches*p.heads≤65535 := by
  have hs:=g.signed
  dsimp [envelope] at hs
  exact ⟨by omega,by omega,by omega,by omega,by omega,g.launchY⟩

#axiomsClean safe
#axiomsClean loop_and_grid_metadata
#specNonCircular Safe avoiding [FP.Structural.exec,Whole.invoke]
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.Access
