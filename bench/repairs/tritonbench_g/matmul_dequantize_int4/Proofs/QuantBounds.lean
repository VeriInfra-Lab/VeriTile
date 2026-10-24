import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantHost
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantBounds
/-- Tensor-library indices are signed64; Python dimension arithmetic is exact.
Packing intentionally wraps signed32; it is not given a no-overflow premise. -/
structure Signed64 (c : QuantHost.Call) : Prop where
  input : 4*c.W.capacity<2^63
  copied : 4*(c.shape.K*c.shape.N)<2^63

theorem input_address (c : QuantHost.Call) (h : c.Valid) (b : Signed64 c)
    (k n byte : Nat) (hk : k<c.shape.K) (hn : n<c.shape.N) (hb : byte<4) :
    4*c.W.addr k n+byte<4*c.W.capacity ∧ 4*c.W.addr k n+byte<2^63 := by
  have hv := MatmulKernel.view_bound c.W h.view k n
    (by simpa only [h.wShape.1] using hk) (by simpa only [h.wShape.2] using hn)
  have he : 4*c.W.addr k n+byte<4*c.W.capacity := by omega
  exact ⟨he,lt_trans he b.input⟩

theorem grouped_address (c : QuantHost.Call) (h : c.Valid) (b : Signed64 c)
    (n g t byte : Nat) (hn : n<c.shape.N) (hg : g<c.shape.groups)
    (ht : t<c.shape.group) (hb : byte<4) :
    4*((n*c.shape.groups+g)*c.shape.group+t)+byte<2^63 := by
  have hk : c.shape.groups*c.shape.group=c.shape.K := Nat.div_mul_cancel h.shape.groups
  have hgt : g*c.shape.group+t<c.shape.K := by
    have := Nat.mul_le_mul_right c.shape.group (show g+1≤c.shape.groups by omega)
    rw [Nat.add_mul,hk] at this
    omega
  have hflat := MatmulKernel.tiled_bound n (g*c.shape.group+t) c.shape.N c.shape.K hn hgt
  have he : (n*c.shape.groups+g)*c.shape.group+t=n*c.shape.K+(g*c.shape.group+t) := by
    rw [Nat.add_mul,Nat.mul_assoc,hk];omega
  rw [he]
  have hm : c.shape.N*c.shape.K=c.shape.K*c.shape.N := Nat.mul_comm _ _
  have := b.copied
  omega

theorem packed_indices (s : Quantizer.Shape) (h : s.Valid) (k n : Nat) (i : Fin 8)
    (hk : k<s.K/8) (hn : n<s.N/8) : k*8+i.val<s.K ∧ n*8+i.val<s.N := by
  have hh := Nat.mul_le_mul_right 8 (show k+1≤s.K/8 by omega)
  have hj := Nat.mul_le_mul_right 8 (show n+1≤s.N/8 by omega)
  rw [Nat.add_mul,Nat.div_mul_cancel h.k8] at hh
  rw [Nat.add_mul,Nat.div_mul_cancel h.n8] at hj
  have := i.isLt
  omega
#axiomsClean input_address
#axiomsClean grouped_address
#axiomsClean packed_indices
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.QuantBounds
