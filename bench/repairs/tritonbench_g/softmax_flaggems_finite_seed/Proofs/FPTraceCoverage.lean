import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTraceBounds

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPTrace
open VeriTile Triton FP.Structural FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
variable {α : Type} [Inhabited α]

theorem reverse_visit (N B n : Nat) (hB : 0<B) (hn : n<N) :
    ∃ cur, cur∈visits N B 0 ∧ RealAddress.previous N B-cur+n%B=n ∧
      (cur=0 ∨ cur∈visits N B B) := by
  let q := RealAddress.rounds N B
  let j := q-1-n/B
  have hq : n/B<q := owner_lt hB hn
  have hqp : 0<q := Nat.lt_of_le_of_lt (Nat.zero_le (n/B)) hq
  have hj : j<q := lt_of_le_of_lt (Nat.sub_le (q-1) (n/B)) (Nat.sub_lt hqp (by decide))
  have hcur : j*B<N := (RealAddress.tile_start_lt_iff (by omega) hB).2 hj
  have hv : j*B∈visits N B 0 := by simpa using visits_contains N B 0 j hB (by simpa using hcur)
  refine ⟨j*B,hv,?_,?_⟩
  · rw [RealAddress.reverse_base]
    have he : RealAddress.rounds N B-1-j=n/B := by dsimp [j,q] at *; omega
    rw [he]
    exact owner_offset n B
  · by_cases hz : j=0
    · exact Or.inl (by simp [hz])
    · right
      have he : B+(j-1)*B=j*B := by
        calc
          B+(j-1)*B=(1+(j-1))*B := by simp [Nat.add_mul]
          _=j*B := by congr 1; omega
      rw [←he]
      exact visits_contains N B B (j-1) hB (by simpa [he] using hcur)

def backwardOwner (d : Dimensions) (c : LaunchConfig) (r k : Nat) : Program :=
  if 1<d.K then (r,k/c.C) else (r/c.C,0)
def forwardOwner (d : Dimensions) (c : LaunchConfig) (r k : Nat) : Program :=
  if 1<d.K then (r,k/c.C) else (r,0)

theorem backward_contains (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (r n k : Nat) (hr : r<d.M) (hn : n<d.N) (hk : k<d.K) (base : State α) :
    let p := backwardOwner d c r k
    (Y,r*d.N*d.K+n*d.K+k)∈events (backwardTiles A P G Y d c p.1 p.2 base) := by
  dsimp only
  have hdiv := owner_offset n c.B
  have hcol := owner_offset k c.C
  have hrow := owner_offset r c.C
  have hv : n/c.B*c.B∈visits d.N c.B 0 := by
    have hb := Nat.div_mul_le_self n c.B
    simpa using visits_contains d.N c.B 0 (n/c.B) hB (by omega)
  unfold backwardTiles backwardOwner
  split <;> cases hc : c.one
  · simp only [hc,Bool.false_eq_true,if_false,mem_events_map,Tile.mem_events]
    refine ⟨n/c.B*c.B,hv,⟨⟨n%c.B,Nat.mod_lt n hB⟩,⟨k%c.C,Nat.mod_lt k hC⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [backwardNoninnerTile,FPBackwardNoninner.active,hdiv,hcol] using And.intro hn hk
    · simp [backwardNoninnerTile,FPBackwardNoninner.offset,hdiv,hcol]
  · have hNB : d.N≤c.B := by have := hone; simp [hc,oneTile] at this; omega
    simp only [hc,if_true,mem_events_single,Tile.mem_events]
    refine ⟨⟨⟨n,lt_of_lt_of_le hn hNB⟩,⟨k%c.C,Nat.mod_lt k hC⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [noninnerActive,hcol] using And.intro hn hk
    · simp [noninnerOffset,hcol]
  · have hk1 : d.K=1 := by omega
    have hk0 : k=0 := by omega
    simp only [hc,Bool.false_eq_true,if_false,mem_events_map,Tile.mem_events]
    refine ⟨n/c.B*c.B,hv,⟨⟨r%c.C,Nat.mod_lt r hC⟩,⟨n%c.B,Nat.mod_lt n hB⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [backwardInnerTile,FPBackwardInner.active,hdiv,hrow] using And.intro hr hn
    · simp [backwardInnerTile,FPBackwardInner.offset,hdiv,hrow,hk1,hk0]
  · have hNB : d.N≤c.B := by have := hone; simp [hc,oneTile] at this; omega
    have hk1 : d.K=1 := by omega
    have hk0 : k=0 := by omega
    simp only [hc,if_true,mem_events_single,Tile.mem_events]
    refine ⟨⟨⟨r%c.C,Nat.mod_lt r hC⟩,⟨n,lt_of_lt_of_le hn hNB⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [backwardInnerActive,hrow] using And.intro hr hn
    · simp [backwardInnerOffset,hrow,hk1,hk0]

theorem forward_contains (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (r n k : Nat) (hn : n<d.N) (hk : k<d.K) (base : State α) :
    let p := forwardOwner d c r k
    (Y,r*d.N*d.K+n*d.K+k)∈events (forwardTiles A X Y d c p.1 p.2 base) := by
  dsimp only
  have hcol := owner_offset k c.C
  obtain ⟨cur,hcur,hrev,hcases⟩ := reverse_visit d.N c.B n hB hn
  have hrevI : FPForwardInner.previous d.N c.B-cur+n%c.B=n := hrev
  have hrevN : FPForwardNoninner.previous d.N c.B-cur+n%c.B=n := hrev
  unfold forwardTiles forwardOwner
  split <;> cases hc : c.one
  · simp only [hc,Bool.false_eq_true,if_false,mem_events_map,Tile.mem_events]
    refine ⟨cur,hcur,⟨⟨n%c.B,Nat.mod_lt n hB⟩,⟨k%c.C,Nat.mod_lt k hC⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [forwardNoninnerTile,FPForwardNoninner.active,hrevN,hcol] using And.intro hn hk
    · simp [forwardNoninnerTile,FPForwardNoninner.offset,hrevN,hcol]
  · have hNB : d.N≤c.B := by have := hone; simp [hc,oneTile] at this; omega
    simp only [hc,if_true,mem_events_single,Tile.mem_events]
    refine ⟨⟨⟨n,lt_of_lt_of_le hn hNB⟩,⟨k%c.C,Nat.mod_lt k hC⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [noninnerActive,hcol] using And.intro hn hk
    · simp [noninnerOffset,hcol]
  · have hk1 : d.K=1 := by omega
    have hk0 : k=0 := by omega
    simp only [hc,Bool.false_eq_true,if_false,mem_events_append,mem_events_map,Tile.mem_events]
    rcases hcases with hz|hv
    · left
      have hv : cur∈visits c.B c.B 0 := by
        subst cur
        simpa using visits_contains c.B c.B 0 0 hB (by omega)
      refine ⟨cur,hv,⟨⟨n%c.B,Nat.mod_lt n hB⟩,PUnit.unit⟩,?_,?_⟩
      · simpa [forwardInnerTile,FPForwardInner.active,hrevI] using hn
      · simp [forwardInnerTile,FPForwardInner.offset,hrevI,hk1,hk0]
    · right
      refine ⟨cur,hv,⟨⟨n%c.B,Nat.mod_lt n hB⟩,PUnit.unit⟩,?_,?_⟩
      · simp [forwardInnerTile,FPForwardInner.active]
      · simp [forwardInnerTile,FPForwardInner.offset,hrevI,hk1,hk0]
  · have hNB : d.N≤c.B := by have := hone; simp [hc,oneTile] at this; omega
    have hk1 : d.K=1 := by omega
    have hk0 : k=0 := by omega
    simp only [hc,if_true,mem_events_single,Tile.mem_events]
    refine ⟨⟨⟨n,lt_of_lt_of_le hn hNB⟩,PUnit.unit⟩,?_,?_⟩
    · simpa [innerActive] using hn
    · simp [innerOffset,hk1,hk0]

theorem backward_owner_mem (d : Dimensions) (c : LaunchConfig) (hC : 0<c.C)
    (r k : Nat) (hr : r<d.M) (hk : k<d.K) :
    backwardOwner d c r k∈grid (FPGrid.backwardGrid d c).1 (FPGrid.backwardGrid d c).2 := by
  unfold backwardOwner FPGrid.backwardGrid
  split <;> simp only [mem_grid]
  · exact ⟨hr,owner_lt hC hk⟩
  · exact ⟨owner_lt hC hr,by decide⟩

theorem forward_owner_mem (d : Dimensions) (c : LaunchConfig) (hC : 0<c.C)
    (r k : Nat) (hr : r<d.M) (hk : k<d.K) :
    forwardOwner d c r k∈grid (FPGrid.forwardGrid d c).1 (FPGrid.forwardGrid d c).2 := by
  unfold forwardOwner FPGrid.forwardGrid
  split <;> simp only [mem_grid]
  · exact ⟨hr,owner_lt hC hk⟩
  · exact ⟨hr,by decide⟩

#axiomsClean backward_contains
#axiomsClean forward_contains
#axiomsClean backward_owner_mem
#axiomsClean forward_owner_mem
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPTrace
