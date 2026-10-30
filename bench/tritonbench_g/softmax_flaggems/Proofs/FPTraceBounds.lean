import bench.tritonbench_g.softmax_flaggems.Proofs.FPTraceBlocks

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPTrace
open VeriTile Triton FP.Structural FPReference FPLoop
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 30000
variable {α β : Type} [Inhabited α]

@[simp] theorem mem_events_single (a : RegionName × Nat) (t : Tile α) :
    a∈events [t] ↔ a∈t.events := by simp [events]

@[simp] theorem mem_events_map (a : RegionName × Nat) (xs : List β) (f : β → Tile α) :
    a∈events (xs.map f) ↔ ∃ x∈xs, a∈(f x).events := by simp [events]

@[simp] theorem mem_events_append (a : RegionName × Nat) (xs ys : List (Tile α)) :
    a∈events (xs++ys) ↔ a∈events xs ∨ a∈events ys := by simp [events]

/-- Only output addresses can occur in the actual block's store trace. -/
theorem backward_bounds (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hK : 0<d.K) (row col : Nat) (hr : 1<d.K → row<d.M) (base : State α)
    (a : RegionName × Nat) (ha : a∈events (backwardTiles A P G Y d c row col base)) :
    a.1=Y ∧ a.2<d.M*d.N*d.K := by
  unfold backwardTiles at ha
  split at ha <;> cases hc : c.one
  · simp only [hc,Bool.false_eq_true,if_false,mem_events_map,Tile.mem_events] at ha
    obtain ⟨cur,hcur,i,hi,rfl⟩ := ha
    have hact : cur+i.1.val<d.N ∧ col*c.C+i.2.1.val<d.K := by
      simpa [backwardNoninnerTile,FPBackwardNoninner.active] using hi
    exact ⟨rfl,RealAddress.non_inner_bound (hr ‹1<d.K›) hact.1 hact.2⟩
  · simp only [hc,if_true,mem_events_single,Tile.mem_events] at ha
    obtain ⟨i,hi,rfl⟩ := ha
    have hact : i.1.val<d.N ∧ col*c.C+i.2.1.val<d.K := by
      simpa [noninnerActive] using hi
    exact ⟨rfl,RealAddress.non_inner_bound (hr ‹1<d.K›) hact.1 hact.2⟩
  · simp only [hc,Bool.false_eq_true,if_false,mem_events_map,Tile.mem_events] at ha
    obtain ⟨cur,hcur,i,hi,rfl⟩ := ha
    have hact : row*c.C+i.1.val<d.M ∧ cur+i.2.1.val<d.N := by
      simpa [backwardInnerTile,FPBackwardInner.active] using hi
    have hk : d.K=1 := by omega
    exact ⟨rfl,by simpa [hk,backwardInnerTile,FPBackwardInner.offset] using inner_bound hact.1 hact.2⟩
  · simp only [hc,if_true,mem_events_single,Tile.mem_events] at ha
    obtain ⟨i,hi,rfl⟩ := ha
    have hact : row*c.C+i.1.val<d.M ∧ i.2.1.val<d.N := by
      simpa [backwardInnerActive] using hi
    have hk : d.K=1 := by omega
    exact ⟨rfl,by simpa [hk,backwardInnerOffset] using inner_bound hact.1 hact.2⟩

theorem forward_bounds (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hone : c.one=oneTile d.N c.B)
    (row col : Nat) (hr : row<d.M) (base : State α)
    (a : RegionName × Nat) (ha : a∈events (forwardTiles A X Y d c row col base)) :
    a.1=Y ∧ a.2<d.M*d.N*d.K := by
  unfold forwardTiles at ha
  split at ha <;> cases hc : c.one
  · simp only [hc,Bool.false_eq_true,if_false,mem_events_map,Tile.mem_events] at ha
    obtain ⟨cur,hcur,i,hi,rfl⟩ := ha
    have hact : FPForwardNoninner.previous d.N c.B-cur+i.1.val<d.N ∧ col*c.C+i.2.1.val<d.K := by
      simpa [forwardNoninnerTile,FPForwardNoninner.active] using hi
    exact ⟨rfl,RealAddress.non_inner_bound hr hact.1 hact.2⟩
  · simp only [hc,if_true,mem_events_single,Tile.mem_events] at ha
    obtain ⟨i,hi,rfl⟩ := ha
    have hact : i.1.val<d.N ∧ col*c.C+i.2.1.val<d.K := by simpa [noninnerActive] using hi
    exact ⟨rfl,RealAddress.non_inner_bound hr hact.1 hact.2⟩
  · have hk : d.K=1 := by omega
    have hBN : c.B<d.N := by simp [hc,oneTile] at hone; omega
    simp only [hc,Bool.false_eq_true,if_false,mem_events_append,mem_events_map,Tile.mem_events] at ha
    rcases ha with ⟨cur,hcur,i,hi,rfl⟩|⟨cur,hcur,i,hi,rfl⟩
    · have hact : FPForwardInner.previous d.N c.B-cur+i.1.val<d.N := by
        simpa [forwardInnerTile,FPForwardInner.active] using hi
      exact ⟨rfl,by simpa [hk,forwardInnerTile,FPForwardInner.offset] using inner_bound hr hact⟩
    · have hb := (visits_bounds d.N c.B c.B cur hcur).1
      have hp : FPForwardInner.previous d.N c.B<d.N := RealAddress.previous_lt hN hB
      have hi := i.1.isLt
      have hact : FPForwardInner.previous d.N c.B-cur+i.1.val<d.N := by omega
      exact ⟨rfl,by simpa [hk,forwardInnerTile,FPForwardInner.offset] using inner_bound hr hact⟩
  · simp only [hc,if_true,mem_events_single,Tile.mem_events] at ha
    obtain ⟨i,hi,rfl⟩ := ha
    have hact : i.1.val<d.N := by simpa [innerActive] using hi
    have hk : d.K=1 := by omega
    exact ⟨rfl,by simpa [hk,innerOffset] using inner_bound hr hact⟩

#axiomsClean backward_bounds
#axiomsClean forward_bounds
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPTrace
