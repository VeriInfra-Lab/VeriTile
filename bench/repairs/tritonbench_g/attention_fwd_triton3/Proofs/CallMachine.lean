import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Machine

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.CallMachine
open VeriTile.Triton Host Machine
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

/-- Runtime INT32 metadata and the CUDA two-dimensional launch limits.
Tensor bases/byte strides additionally obey the ordinary nonwrapping pointer
ABI documented in the task contract. No floating-value guard appears here. -/
structure ABI (c : Call) : Prop where
  query : c.NQ≤2^31-65
  keys : c.NK≤2^31-65
  rounded : c.rounded<2^31
  grid : c.Z*c.H≤65536
  blockM : c.BM≤64
  blockN : c.BN≤64
  offset : signed32 c.offset
  size : signed32 c.size

theorem row32 (c : Call) (h : Valid c) (a : ABI c) (idx : GridIndex c.grid) (r : Fin c.BM) :
    0≤(c.row idx r.val : Int) ∧ (c.row idx r.val : Int)<2^31 := by
  have hb:=lt_trans (row_bound c h idx r) a.rounded
  exact ⟨by omega,by exact_mod_cast hb⟩

theorem tile32 (c : Call) (h : Valid c) (a : ABI c) (idx : GridIndex c.grid) :
    0≤(c.tile idx*c.BM : Int) ∧ (c.tile idx*c.BM : Int)<2^31 := by
  simpa [Call.row] using row32 c h a idx ⟨0,h.blockM⟩

theorem upper_le (c : Call) (idx : GridIndex c.grid) :
    RealKeys.upper (c.tile idx) c.BM c.BN c.NK c.window c.complement c.offset≤c.NK := by
  unfold RealKeys.upper
  split
  · unfold WindowBounds.upperBound
    have hh:=Int.toNat_le_toNat (min_le_right
      (((((c.tile idx+1)*c.BM : Nat) : Int)-1+c.offset+c.BN).tdiv (c.BN : Int)*c.BN) (c.NK : Int))
    simpa using hh
  · exact le_refl _

theorem advance32 (c : Call) (a : ABI c) (idx : GridIndex c.grid)
    (ha : RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size <
      RealKeys.upper (c.tile idx) c.BM c.BN c.NK c.window c.complement c.offset) :
    (BitVec.ofInt 32 (RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size)).toInt=
      (RealKeys.lower (c.tile idx) c.BM c.BN c.window c.complement c.offset c.size : Int) := by
  apply active_advance_exact _ (RealKeys.upper (c.tile idx) c.BM c.BN c.NK c.window c.complement c.offset) c.NK
  · omega
  · exact_mod_cast ha
  · exact_mod_cast upper_le c idx
  · have hk:=a.keys
    norm_num at hk ⊢
    omega

/-- Every launched row uses the widened exact window numerators and widened
M/L base, while Q/O row offsets retain their exact required INT32 ABI. -/
theorem complete (c : Call) (h : Valid c) (a : ABI c) (idx : GridIndex c.grid) (r : Fin c.BM) :
    (lowerNumerator (c.tile idx*c.BM) c.offset c.size).toInt=
      (c.tile idx*c.BM : Int)+c.offset-c.size+1 ∧
    (upperNumerator (c.tile idx*c.BM) c.BM c.offset c.BN).toInt=
      (c.tile idx*c.BM : Int)+c.BM-1+c.offset+c.BN ∧
    (word (c.hz idx)*word c.rounded+word (c.row idx r.val)).toInt=
      (c.hz idx : Int)*c.rounded+c.row idx r.val ∧
    (BitVec.ofInt 32 (c.row idx r.val)).toInt=(c.row idx r.val : Int) := by
  have hm : 0<(c.BM : Int) ∧ (c.BM : Int)≤64 := ⟨by exact_mod_cast h.blockM,by exact_mod_cast a.blockM⟩
  have hn : 0<(c.BN : Int) ∧ (c.BN : Int)≤64 := ⟨by exact_mod_cast h.blockN,by exact_mod_cast a.blockN⟩
  refine ⟨lower_exact _ _ _ (tile32 c h a idx) a.offset a.size,
    upper_exact _ _ _ _ (tile32 c h a idx) hm a.offset hn,?_,block_offset_exact _ (row32 c h a idx r)⟩
  apply carried_address_exact
  · have hb:=lt_of_lt_of_le (hz_bound c idx) a.grid
    exact ⟨by omega,by exact_mod_cast hb⟩
  · exact ⟨by omega,by exact_mod_cast a.rounded⟩
  · exact ⟨by omega,by exact_mod_cast row_bound c h idx r⟩

#print axioms complete
#print axioms advance32
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.CallMachine
