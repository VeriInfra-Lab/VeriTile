import bench.tritonbench_g.parallel_retention_attention.Proofs.Host

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

theorem pack_divmod (b c i : Nat) (hi : i<b) :
    (c*b+i)/b=c ∧ (c*b+i)%b=i := by
  have hb : 0<b := by omega
  constructor
  · rw [Nat.mul_comm c b,Nat.mul_add_div hb,Nat.div_eq_of_lt hi,Nat.add_zero]
  · rw [Nat.mul_comm c b,Nat.mul_add_mod,Nat.mod_eq_of_lt hi]

/-- Inverse of the exact canonical dense source scratch address. -/
theorem scratch_decode (M T D shard bh t e : Nat) (hb : bh<M) (ht : t<T) (he : e<D) :
    let i := shard*(M*T*D)+dense T D bh t e
    i%D=e ∧ i/D%T=t ∧ i/D/T%M=bh ∧ i/(M*T*D)=shard := by
  have bound := dense_bound T D M bh t e hb ht he
  have eD : shard*(M*T*D)+dense T D bh t e=((shard*M+bh)*T+t)*D+e := by unfold dense;ring
  have pD := pack_divmod D ((shard*M+bh)*T+t) e he
  have pT := pack_divmod T (shard*M+bh) t ht
  have pM := pack_divmod M shard bh hb
  dsimp only
  refine ⟨?_,?_,?_,(pack_divmod (M*T*D) shard (dense T D bh t e) bound).1⟩
  · rw [eD,pD.2]
  · rw [eD,pD.1,pT.2]
  · rw [eD,pD.1,pT.1,pM.2]

structure ProgramDomain (s : Shape) (L : Nat) (tile : Tile) : Prop where
  bh : tile.bh<s.B*s.H
  chunk : tile.chunk<blocks s.T L
  key : tile.key<blocks s.DK s.BK
  value : tile.value<blocks s.DV s.BV

def ProgramDomain.gridX (s : Shape) (tile : Tile) := tile.key*blocks s.DV s.BV+tile.value

theorem grid_decode (s : Shape) (L : Nat) (tile : Tile) (h : ProgramDomain s L tile) :
    ProgramDomain.gridX s tile<blocks s.DK s.BK*blocks s.DV s.BV ∧
    ProgramDomain.gridX s tile/blocks s.DV s.BV=tile.key ∧
    ProgramDomain.gridX s tile%blocks s.DV s.BV=tile.value := by
  have e := pack_divmod (blocks s.DV s.BV) tile.key tile.value h.value
  refine ⟨?_,e⟩
  unfold ProgramDomain.gridX
  have hk : tile.key+1≤blocks s.DK s.BK := by have := h.key;omega
  nlinarith [h.value]

theorem forward_owner (s : Shape) (shard bh t v : Nat) (hbk : 0<s.BK) (hbv : 0<s.BV)
    (hs : shard<blocks s.DK s.BK) (hb : bh<s.B*s.H) (ht : t<s.T) (hv : v<s.DV) :
    ProgramDomain s 128 ⟨bh,t/128,shard,v/s.BV⟩ :=
  ⟨hb,(tile_owner _ _ _ (by norm_num) ht).1,hs,(tile_owner _ _ _ hbv hv).1⟩

theorem query_key_owner (s : Shape) (shard bh t e : Nat) (hbk : 0<s.BK) (hbv : 0<s.BV)
    (hs : shard<blocks s.DV s.BV) (hb : bh<s.B*s.H) (ht : t<s.T) (he : e<s.DK) :
    ProgramDomain s 64 ⟨bh,t/64,e/s.BK,shard⟩ :=
  ⟨hb,(tile_owner _ _ _ (by norm_num) ht).1,(tile_owner _ _ _ hbk he).1,hs⟩

theorem value_owner (s : Shape) (shard bh t v : Nat) (hbk : 0<s.BK) (hbv : 0<s.BV)
    (hs : shard<blocks s.DK s.BK) (hb : bh<s.B*s.H) (ht : t<s.T) (hv : v<s.DV) :
    ProgramDomain s 64 ⟨bh,t/64,shard,v/s.BV⟩ :=
  ⟨hb,(tile_owner _ _ _ (by norm_num) ht).1,hs,(tile_owner _ _ _ hbv hv).1⟩

/-- Every active source store has one owner; equal scratch addresses cannot
belong to distinct shards, heads, rows or features. -/
theorem scratch_unique (M T D sh sh' bh bh' t t' e e' : Nat)
    (hb : bh<M) (hb' : bh'<M) (ht : t<T) (ht' : t'<T) (he : e<D) (he' : e'<D)
    (eq : sh*(M*T*D)+dense T D bh t e=sh'*(M*T*D)+dense T D bh' t' e') :
    sh=sh' ∧ bh=bh' ∧ t=t' ∧ e=e' := by
  obtain ⟨hsh,hcell⟩ := tile_injective (M*T*D) sh sh' _ _
    (dense_bound T D M bh t e hb ht he) (dense_bound T D M bh' t' e' hb' ht' he') eq
  exact ⟨hsh,dense_injective T D bh bh' t t' e e' ht ht' he he' hcell⟩

#axiomsClean scratch_decode
#axiomsClean grid_decode
#axiomsClean forward_owner
#axiomsClean query_key_owner
#axiomsClean value_owner
#axiomsClean scratch_unique
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
