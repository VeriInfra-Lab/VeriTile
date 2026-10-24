import bench.repairs.tritonbench_g.matmul_dequant_int4.Proofs.SourceExecution
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.MachineDomain
set_option maxHeartbeats 5000000

/-- Canonical raw fused ABI, independent of the s1 host. Padding includes
inactive M lanes and the formed pointers after the final K increment. -/
structure Fused (s : Shape) (ng : Bool) where
  source : Source.FusedRawScope s ng
  indices : IndexDomain s
  mEnvelope : ceil s.m 128*128 ≤ indices.paddedM
  kEnvelope : s.k+128 ≤ indices.paddedK
  grid : indices.paddedM*indices.paddedN<2^31
theorem fused_envelopes (s : Shape) (ng : Bool) (h : Fused s ng)
    (c : Source.FusedConfig) (hc : c∈Source.fusedConfigs) :
    ceil s.m c.tile.bm*c.tile.bm≤h.indices.paddedM ∧
    ceil s.n c.tile.bn*c.tile.bn≤h.indices.paddedN ∧ s.k+c.tile.bk≤h.indices.paddedK := by
  have ht := Source.fused_valid c hc
  obtain ⟨hm,hn,hk⟩ := Source.fused_divisors c hc
  have hm' := Source.ceil_le_multiple s.m c.tile.bm (ceil s.m 128*128) ht.bmpos
    (ceil_cover _ _ (by decide)) (dvd_mul_of_dvd_right hm _)
  have hn' : c.tile.bn∣s.n := dvd_trans hn h.source.n256
  have hne : ceil s.n c.tile.bn*c.tile.bn=s.n := by
    rw [ceil_of_dvd _ _ ht.bnpos hn',Nat.div_mul_cancel hn']
  have hkle := Nat.le_of_dvd (by decide : 0<128) hk
  exact ⟨hm'.trans h.mEnvelope,by rw [hne];exact h.indices.columns,by have := h.kEnvelope;omega⟩
theorem fused_grid_indices (s : Shape) (ng : Bool) (h : Fused s ng)
    (c : Source.FusedConfig) (hc : c∈Source.fusedConfigs) :
    blocksM s c.tile*blocksN s c.tile<2^31 := by
  have e := fused_envelopes s ng h c hc
  have ht := Source.fused_valid c hc
  have hm : blocksM s c.tile≤h.indices.paddedM := by unfold blocksM;nlinarith [ht.bmpos,e.1]
  have hn : blocksN s c.tile≤h.indices.paddedN := by unfold blocksN;nlinarith [ht.bnpos,e.2.1]
  exact lt_of_le_of_lt (Nat.mul_le_mul hm hn) h.grid
theorem fused_formed (s : Shape) (ng : Bool) (h : Fused s ng)
    (c : Source.FusedConfig) (hc : c∈Source.fusedConfigs) (r n k : Nat)
    (hr : r<ceil s.m c.tile.bm*c.tile.bm) (hn : n<s.n) (hk : k<s.k+c.tile.bk) :
    aAddress s r k<2^31 ∧ qAddress s k n<2^31 ∧ sAddress s k n<2^31 ∧
    zAddress s k n<2^31 ∧ cAddress s r n<2^31 ∧ cAddress s k n<2^31 := by
  have e := fused_envelopes s ng h c hc
  exact formed_addresses s h.indices r n k (hr.trans_le e.1) (hn.trans_le h.indices.columns)
    (Nat.le_trans (Nat.le_of_lt hk) e.2.2)
theorem standalone_grid_indices (s : Shape) (strides : Strides.Packed) (h : Strides.Guard strides s)
    (c : Source.DequantConfig) (hc : c∈Source.dequantConfigs) : ceil s.k c.bk*ceil s.n c.bn<2^31 := by
  have e := Source.dequant_envelopes s c hc
  obtain ⟨hk,hn,_,_⟩ := Source.dequant_valid c hc
  have hk' : ceil s.k c.bk≤Source.paddedK s := by nlinarith [e.1]
  have hn' : ceil s.n c.bn≤Source.paddedN s := by nlinarith [e.2]
  exact lt_of_le_of_lt (Nat.mul_le_mul hk' hn') h.grid

#axiomsClean fused_envelopes
#axiomsClean fused_grid_indices
#axiomsClean fused_formed
#axiomsClean standalone_grid_indices
end VeriTile.Bench.Repairs.MatmulDequantInt4.MachineDomain
