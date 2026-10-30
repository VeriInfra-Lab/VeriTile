import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.Admission

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Admission
set_option maxHeartbeats 4000000

/-- Canonical flattened addresses remain bounded even for masked row/feature
lanes beyond the logical tensor. This is stronger than active-load safety. -/
theorem padded_address (heads T F rows tail h row feature : Nat)
    (hh : h<heads) (hr : row<T+rows) (hf : feature<F+tail) :
    h*(T*F)+row*F+feature<heads*T*F+rows*F+tail := by
  have hhead:=Nat.mul_le_mul_right (T*F) (show h+1≤heads by omega)
  have hrow:=Nat.mul_le_mul_right F (show row+1≤T+rows by omega)
  nlinarith

theorem extent_bound (c : Host.Dimensions) (n : Nat) (hn : n∈extents c) :
    n≤(extents c).foldl max 0 := by
  simp only [extents,List.mem_cons,List.not_mem_nil,or_false] at hn
  simp only [extents,List.foldl_cons,List.foldl_nil]
  rcases hn with h|h|h|h|h <;> subst n <;> omega

theorem padded_bound (c : Host.Dimensions) (base rows features tail : Nat)
    (hb : base∈extents c) (hr : rows≤c.BT)
    (hf : features≤max (max (max c.K c.V) c.BT) 16)
    (ht : tail≤256*max c.V 1+256) :
    base+rows*features+tail≤bound c := by
  have hb:=extent_bound c base hb
  have hm:=Nat.mul_le_mul hr hf
  unfold bound
  omega

theorem query_address (mode : Mode) (c : Host.Dimensions) (finite : Finite mode c)
    (h row feature : Nat) (hh : h<c.heads) (hr : row<c.T+c.BT) (hf : feature<c.K+256) :
    h*(c.T*c.K)+row*c.K+feature<2^31 := by
  have hp:=padded_address c.heads c.T c.K c.BT 256 h row feature hh hr hf
  have hb:=padded_bound c (c.heads*c.T*c.K) c.BT c.K 256
    (by simp [extents]) (by omega) (by omega) (by omega)
  exact lt_trans (lt_of_lt_of_le hp hb) finite.2.1

theorem attention_address (mode : Mode) (c : Host.Dimensions) (finite : Finite mode c)
    (h row col : Nat) (hh : h<c.heads) (hr : row<c.T+c.BT) (hc : col<c.BT) :
    h*(c.T*c.BT)+row*c.BT+col<2^31 := by
  have hp:=padded_address c.heads c.T c.BT c.BT 0 h row col hh hr (by simpa using hc)
  have hb:=padded_bound c (c.heads*c.T*c.BT) c.BT c.BT 0
    (by simp [extents]) (by omega) (by omega) (by omega)
  exact lt_trans (lt_of_lt_of_le hp hb) finite.2.1

theorem output_address (mode : Mode) (c : Host.Dimensions) (finite : Finite mode c)
    (h row value : Nat) (hh : h<c.heads) (hr : row<c.T+c.BT) (hv : value<c.V+32) :
    h*(c.T*c.V)+row*c.V+value<2^31 := by
  have hp:=padded_address c.heads c.T c.V c.BT 32 h row value hh hr hv
  have hb:=padded_bound c (c.heads*c.T*c.V) c.BT c.V 32
    (by simp [extents]) (by omega) (by omega) (by omega)
  exact lt_trans (lt_of_lt_of_le hp hb) finite.2.1

theorem scratch_address (mode : Mode) (c : Host.Dimensions) (finite : Finite mode c)
    (block h row col : Nat) (hk : block<c.NK) (hh : h<c.heads)
    (hr : row<c.T+c.BT) (hc : col<16) :
    (block*c.heads+h)*(c.T*16)+row*16+col<2^31 := by
  have hblock : block*c.heads+h<c.NK*c.heads := by
    have hm:=Nat.mul_le_mul_right c.heads (show block+1≤c.NK by omega)
    nlinarith
  have hp:=padded_address (c.NK*c.heads) c.T 16 c.BT 0 (block*c.heads+h) row col
    hblock hr (by simpa using hc)
  have hb:=padded_bound c (c.NK*c.heads*c.T*16) c.BT 16 0
    (by simp [extents]) (by omega) (by omega) (by omega)
  exact lt_trans (lt_of_lt_of_le hp hb) finite.2.1

theorem history_address (mode : Mode) (c : Host.Dimensions) (finite : Finite mode c)
    (h chunk feature value : Nat) (hh : h<c.heads) (hc : chunk<c.NT)
    (hk : feature<c.K+32) (hv : value<c.V+32) :
    (h*c.NT+chunk)*(c.K*c.V)+feature*c.V+value<2^31 := by
  have hhead : h*c.NT+chunk<c.heads*c.NT := by
    have hm:=Nat.mul_le_mul_right c.NT (show h+1≤c.heads by omega)
    nlinarith
  have hp:=padded_address (c.heads*c.NT) c.K c.V 32 32 (h*c.NT+chunk) feature value hhead hk hv
  have hb:=extent_bound c (c.heads*c.NT*c.K*c.V) (by simp [extents])
  have hm:=Nat.le_max_left c.V 1
  have hn:=finite.2.1
  unfold bound at hn
  omega

#axiomsClean query_address
#axiomsClean attention_address
#axiomsClean output_address
#axiomsClean scratch_address
#axiomsClean history_address
end VeriTile.Bench.Repairs.ChunkGlaFwd.Admission
