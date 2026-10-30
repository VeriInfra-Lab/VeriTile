import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_RealFlat
import VeriTile.Triton.Launch
import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_HostGeometry

/-! Host tensor/layout binding. The output is fresh, contiguous and FP32. The
history stride is V because the two middle history axes are packed into NT*K.
No history producer, allocator implementation or runtime implementation is
assumed. Allocation disjointness is the ordinary caller memory contract. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Host
open VeriTile.Triton Math Real
noncomputable section
set_option maxHeartbeats 2000000

def layout (c : Shape) : Layout :=
  ⟨c.T*c.K, c.K, c.T*c.V, c.V, c.NT*c.K*c.V, c.V,
    c.T, c.K, c.V, c.BT, c.BK, c.BV⟩
def regions : Regions := ⟨"q", "k", "v", "h", "g", "o"⟩

/-- Flattening B/H is the source's `i_bh`; the inner history stride is V,
not K*V. This is the packed `[B,H,NT*K,V]` tensor address. -/
theorem packed_history_address (c : Shape) (l : Launch) (k j : Nat) :
    historyAddress (layout c) l k j =
      ((l.head*c.NT+l.chunk)*c.K+k)*c.V+(l.value*c.BV+j) := by
  simp [historyAddress,layout,column]
  ring

def headIndex (c : Shape) (b : Fin c.B) (h : Fin c.H) : Fin c.heads :=
  ⟨b.val*c.H+h.val,by
    have hm := Nat.mul_le_mul_right c.H (Nat.succ_le_of_lt b.isLt)
    have hh := h.isLt
    dsimp [Shape.heads]
    nlinarith⟩

theorem returned_tensor_address (c : Shape) (b : Fin c.B) (h : Fin c.H) (t v : Nat) :
    (headIndex c b h).val*(c.T*c.V)+t*c.V+v =
      b.val*(c.H*c.T*c.V)+h.val*(c.T*c.V)+t*c.V+v := by
  simp [headIndex]
  ring

def launchIndex (c : Shape) (idx : GridIndex (grid c)) : Launch :=
  ⟨(idx ⟨0, by simp [grid,Shape.grid,Grid.rank]⟩).val,
   (idx ⟨1, by simp [grid,Shape.grid,Grid.rank]⟩).val,
   (idx ⟨2, by simp [grid,Shape.grid,Grid.rank]⟩).val⟩

@[simp] theorem launch_grid (c : Shape) (s : BlockState) (idx : GridIndex (grid c)) :
    launch (s.withGridIndex idx) = launchIndex c idx := by
  simp [launch,launchIndex,BlockState.withGridIndex,BlockState.withPids,
    BlockState.withNumPids,GridIndex.toPids,grid,Shape.grid,Grid.rank]

theorem launch_lt (c : Shape) (idx : GridIndex (grid c)) :
    (launchIndex c idx).value < c.NV ∧ (launchIndex c idx).chunk < c.NT ∧
      (launchIndex c idx).head < c.heads := by
  exact ⟨(idx ⟨0, by simp [grid,Shape.grid,Grid.rank]⟩).isLt,
    (idx ⟨1, by simp [grid,Shape.grid,Grid.rank]⟩).isLt,
    (idx ⟨2, by simp [grid,Shape.grid,Grid.rank]⟩).isLt⟩

theorem launchIndex_injective (c : Shape) : Function.Injective (launchIndex c) := by
  intro i j he
  funext a
  apply Fin.ext
  have h0 := congrArg Launch.value he
  have h1 := congrArg Launch.chunk he
  have h2 := congrArg Launch.head he
  have ha : a.val < 3 := a.isLt
  rcases a with ⟨a,ha⟩
  interval_cases a <;> assumption

/-- Capacity in elements, including all NT history matrices. -/
def capacity (c : Shape) (r : RegionName) : Nat :=
  if r = "q" ∨ r = "k" then c.heads*c.T*c.K
  else if r = "v" ∨ r = "o" then c.heads*c.T*c.V
  else if r = "h" then c.heads*c.NT*c.K*c.V
  else if r = "g" then c.heads*c.T else 0

def Allocated (c : Shape) (A : FlatAlloc) : Prop :=
  A.Disjoint ∧ (∀ r, r ∉ A.regions → A.extent r = 0) ∧
    ∀ r, capacity c r ≤ A.extent r

theorem pair_lt {a b n m : Nat} (ha : a < n) (hb : b < m) : a*m+b < n*m := by
  have h := Nat.mul_le_mul_right m (Nat.succ_le_of_lt ha)
  nlinarith

theorem triple_lt {a b d n m k : Nat} (ha : a < n) (hb : b < m) (hd : d < k) :
    a*(m*k)+b*k+d < n*m*k := by
  have h := pair_lt (pair_lt ha hb) hd
  nlinarith

theorem bounds (c : Shape) (idx : GridIndex (grid c)) (A : FlatAlloc)
    (hc : ∀ r, capacity c r ≤ A.extent r) :
    Safe.Bounds (layout c) (launchIndex c idx) regions A.extent := by
  obtain ⟨hv,ht,hh⟩ := launch_lt c idx
  have qcap : c.heads*c.T*c.K ≤ A.extent "q" := by simpa [capacity] using hc "q"
  have kcap : c.heads*c.T*c.K ≤ A.extent "k" := by simpa [capacity] using hc "k"
  have vcap : c.heads*c.T*c.V ≤ A.extent "v" := by simpa [capacity] using hc "v"
  have ocap : c.heads*c.T*c.V ≤ A.extent "o" := by simpa [capacity] using hc "o"
  have hcap : c.heads*c.NT*c.K*c.V ≤ A.extent "h" := by simpa [capacity] using hc "h"
  have gcap : c.heads*c.T ≤ A.extent "g" := by simpa [capacity] using hc "g"
  constructor
  · intro kb _
    constructor
    · intro i j h
      exact lt_of_lt_of_le (by simpa [layout,regions,Nat.mul_one] using triple_lt hh h.1 h.2) qcap
    · intro i j h
      have hb := triple_lt hh h.2 h.1
      dsimp [layout,regions] at h hb ⊢
      simp only [Nat.mul_one]
      nlinarith
    · intro i j h
      have hb := pair_lt (pair_lt (pair_lt hh ht) h.1) h.2
      dsimp [layout,regions] at h hb ⊢
      simp only [Nat.mul_one]
      nlinarith
  · constructor
    · intro i h
      exact lt_of_lt_of_le (pair_lt hh h) gcap
    · intro i j h
      exact lt_of_lt_of_le (by simpa [layout,regions,Nat.mul_one] using triple_lt hh h.1 h.2) vcap
    · intro i j h
      exact lt_of_lt_of_le (by simpa [layout,regions,Nat.mul_one] using triple_lt hh h.1 h.2) ocap

theorem pair_injective {a b x y m : Nat} (hx : x < m) (hy : y < m)
    (he : a*m+x = b*m+y) : a = b ∧ x = y := by
  have hxy := congrArg (fun z => z % m) he
  simp [Nat.add_mod,Nat.mod_eq_of_lt hx,Nat.mod_eq_of_lt hy] at hxy
  subst y
  have hab : a*m = b*m := Nat.add_right_cancel he
  exact ⟨Nat.eq_of_mul_eq_mul_right (by omega) hab,rfl⟩

/-- Equal active output addresses identify both the program and the lane. -/
theorem address_injective (c : Shape) (l m : Launch)
    (i j : TileIndex [c.BT,c.BV])
    (hi : active (layout c) l i.1.val i.2.1.val)
    (hj : active (layout c) m j.1.val j.2.1.val)
    (he : outputAddress (layout c) l i.1.val i.2.1.val =
      outputAddress (layout c) m j.1.val j.2.1.val) : l = m ∧ i = j := by
  dsimp [active,layout,row,column] at hi hj
  have he' : (l.head*c.T+(l.chunk*c.BT+i.1.val))*c.V+(l.value*c.BV+i.2.1.val) =
      (m.head*c.T+(m.chunk*c.BT+j.1.val))*c.V+(m.value*c.BV+j.2.1.val) := by
    simpa [outputAddress,layout,row,column,Nat.add_mul,Nat.mul_assoc] using he
  obtain ⟨hr,hc⟩ := pair_injective hi.2 hj.2 he'
  obtain ⟨hh,ht⟩ := pair_injective hi.1 hj.1 hr
  obtain ⟨hlt,hit⟩ := pair_injective i.1.isLt j.1.isLt ht
  obtain ⟨hlv,hiv⟩ := pair_injective i.2.1.isLt j.2.1.isLt hc
  have hl : l = m := by cases l; cases m; simp_all
  refine ⟨hl,?_⟩
  rcases i with ⟨i,v,⟨⟩⟩
  rcases j with ⟨j,w,⟨⟩⟩
  have : i = j := Fin.ext hit
  have : v = w := Fin.ext hiv
  simp_all

theorem noCollision (c : Shape) (l : Launch) : NoCollision (layout c) l := by
  intro i j hi hj he
  exact (address_injective c l l i j hi hj he).2

theorem source_binding (c : Shape) (scale : ℝ) (swapped : Bool) :
    program (layout c) regions .fp32 .fp32 .fp32 .fp32 scale swapped = kernel c scale swapped := by
  cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Host
