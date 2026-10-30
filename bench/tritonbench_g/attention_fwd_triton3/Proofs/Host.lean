import bench.tritonbench_g.attention_fwd_triton3.Proofs.Geometry

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Triton Geometry
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

def Call.grid (c : Call) : Grid := ⟨[tiles c.NQ c.BM,c.Z*c.H]⟩

def Call.io (c : Call) := RealContract.io c.BM c.BN c.D c.Z c.finish c.params Bool.false

def Call.source (c : Call) := c.io.kernel.toAlgKernel

def Call.tile (c : Call) (idx : GridIndex c.grid) := (idx ⟨0,by simp [Call.grid,Grid.rank]⟩).val

def Call.hz (c : Call) (idx : GridIndex c.grid) := (idx ⟨1,by simp [Call.grid,Grid.rank]⟩).val

def Call.row (c : Call) (idx : GridIndex c.grid) (r : Nat) := c.tile idx*c.BM+r

def Call.carryAddress (c : Call) (idx : GridIndex c.grid) (r : Nat) := c.hz idx*c.rounded+c.row idx r

def Call.outAddress (c : Call) (idx : GridIndex c.grid) (r d : Nat) :=
  c.o.at (c.hz idx/c.H) (c.hz idx%c.H) (c.row idx r) d

structure Valid (c : Call) : Prop where
  blockM : 0<c.BM
  blockN : 0<c.BN
  divides64 : c.BM∣64
  heads : 0<c.H
  kvHeads : 0<c.HK
  grouped : c.HK∣c.H
  output : Ordered c.o c.H c.rounded c.D

structure Allocated (c : Call) (extent : RegionBounds) : Prop where
  q : c.q.span c.Z c.H c.NQ c.D≤extent "Q"
  k : c.k.span c.Z c.HK c.NK c.D≤extent "K"
  v : c.v.span c.Z c.HK c.NK c.D≤extent "V"
  o : c.o.span c.Z c.H c.rounded c.D≤extent "O"
  m : c.Z*c.H*c.rounded≤extent "M"
  l : c.Z*c.H*c.rounded≤extent "L"

@[simp] theorem pid0 (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    (s.withGridIndex idx).pids 0=c.tile idx := rfl
@[simp] theorem pid1 (c : Call) (s : BlockState) (idx : GridIndex c.grid) :
    (s.withGridIndex idx).pids 1=c.hz idx := rfl

theorem tile_bound (c : Call) (idx : GridIndex c.grid) : c.tile idx<tiles c.NQ c.BM :=
  (idx ⟨0,by simp [Call.grid,Grid.rank]⟩).isLt

theorem hz_bound (c : Call) (idx : GridIndex c.grid) : c.hz idx<c.Z*c.H :=
  (idx ⟨1,by simp [Call.grid,Grid.rank]⟩).isLt

theorem batch_bound (c : Call) (h : Valid c) (idx : GridIndex c.grid) : c.hz idx/c.H<c.Z :=
  (Nat.div_lt_iff_lt_mul h.heads).mpr (hz_bound c idx)

theorem head_bound (c : Call) (h : Valid c) (idx : GridIndex c.grid) : c.hz idx%c.H<c.H :=
  Nat.mod_lt _ h.heads

theorem kv_bound (c : Call) (h : Valid c) (idx : GridIndex c.grid) :
    (c.hz idx%c.H)/(c.H/c.HK)<c.HK := by
  have he := Nat.div_mul_cancel h.grouped
  have hg : 0<c.H/c.HK := Nat.div_pos (Nat.le_of_dvd h.heads h.grouped) h.kvHeads
  apply (Nat.div_lt_iff_lt_mul hg).mpr
  have hh:=head_bound c h idx
  nlinarith

theorem row_bound (c : Call) (h : Valid c) (idx : GridIndex c.grid) (r : Fin c.BM) :
    c.row idx r.val<c.rounded :=
  tile_rows_bounded c.NQ c.BM c.rounded h.blockM (rounded_covers c.NQ)
    (rounded_divisible c.NQ c.BM h.divides64) _ _ (tile_bound c idx) r.isLt

theorem carry_bound (c : Call) (h : Valid c) (idx : GridIndex c.grid) (r : Fin c.BM) :
    c.carryAddress idx r.val<c.Z*c.H*c.rounded := by
  have hr:=row_bound c h idx r
  have hh:=Nat.mul_le_mul_right c.rounded (show c.hz idx+1≤c.Z*c.H by have := hz_bound c idx; omega)
  unfold Call.carryAddress
  nlinarith

theorem evenN (c : Call) : c.params.evenN≠0 → c.params.layout.kvContext%c.BN=0 := by
  simp [Call.params]

theorem unchecked_query_bound (c : Call) (h : Valid c) (idx : GridIndex c.grid)
    (r : Fin c.BM) : c.params.evenM≠0 → c.row idx r.val<c.NQ := by
  intro he
  have he' : c.NQ%c.BM=0 := by simpa [Call.params] using he
  exact tile_rows_bounded c.NQ c.BM c.NQ h.blockM (le_refl _)
    (Nat.dvd_of_mod_eq_zero he') _ _ (tile_bound c idx) r.isLt

/-- All six active windows are derived from descriptor extents and grid bounds.
This is not an allocation or writer oracle supplied to the headline theorem. -/
theorem bounds (c : Call) (h : Valid c) (a : Allocated c extent)
    (s : BlockState) (idx : GridIndex c.grid) : RealIO.Bounds c.io (s.withGridIndex idx) extent := by
  have hz:=batch_bound c h idx
  have hh:=head_bound c h idx
  have hk:=kv_bound c h idx
  constructor
  · intro i hi
    have hr : c.row idx i.1.val<c.NQ := by
      rcases hi with he|hr
      · exact unchecked_query_bound c h idx i.1 he
      · exact hr
    exact lt_of_lt_of_le (address_bound c.q _ _ _ _ _ _ _ _ hz hh hr i.2.1.isLt) a.q
  · intro i
    have hb:=lt_of_lt_of_le (address_bound c.k _ _ _ _ _ _ _ _ hz hk i.1.isLt i.2.1.isLt) a.k
    simpa [Call.io,RealContract.io,Call.params,Strides.at,Nat.add_comm,Nat.add_left_comm,Nat.add_assoc] using hb
  · intro i
    exact lt_of_lt_of_le (address_bound c.v _ _ _ _ _ _ _ _ hz hk i.1.isLt i.2.1.isLt) a.v
  · intro i
    simpa [Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc] using lt_of_lt_of_le (carry_bound c h idx i.1) a.m
  · intro _ i
    simpa [Call.io,RealContract.io,Call.params,Call.carryAddress,Call.row,Nat.add_assoc] using lt_of_lt_of_le (carry_bound c h idx i.1) a.l
  · intro i
    exact lt_of_lt_of_le (address_bound c.o _ _ _ _ _ _ _ _ hz hh (row_bound c h idx i.1) i.2.1.isLt) a.o

#print axioms bounds
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
