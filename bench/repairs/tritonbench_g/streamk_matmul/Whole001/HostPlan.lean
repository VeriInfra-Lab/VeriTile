import bench.repairs.tritonbench_g.streamk_matmul.Whole001.Geometry
namespace VeriTile.Bench.Repairs.StreamK.Whole001
open VeriTile.Bench.TritonBenchG.StreamK
namespace Plan
set_option maxHeartbeats 4000000

theorem selected_bounded (h : HostInput) : (select h).streamTiles ≤ tiles h := by
  by_cases hk:h.k=0
  · simp [select,hk,tiles]
  · simp only [select,hk,if_false,hostPlan]
    split_ifs with hp ht
    · have hsmall:=Nat.mod_le (ceilDiv h.m h.bm*ceilDiv h.n h.bn) h.programs
      simp only [Bool.and_eq_true,decide_eq_true_eq] at ht
      unfold tiles
      omega
    · exact Nat.mod_le _ _
    · exact Nat.zero_le _

theorem selected_blocking (h : HostInput) : (select h).blockingTiles=tiles h-(select h).streamTiles := by
  unfold select hostPlan tiles
  split_ifs <;> simp

theorem selected_partition (h : HostInput) :
    (select h).streamTiles+(select h).blockingTiles=tiles h := by
  rw [selected_blocking]
  exact Nat.add_sub_of_le (selected_bounded h)

theorem selected_divmod (h : HostInput) (hs : 0 < (select h).streamTiles) :
    (select h).fullIterations=((select h).streamTiles*(select h).iterations)/h.programs ∧
    (select h).partialIterations=((select h).streamTiles*(select h).iterations)%h.programs := by
  have hk: h.k≠0:=by have hh:=(selected_active h hs).2; omega
  simp [select,hk,hostPlan]

/-- The repaired host submits exactly the nonempty grids, in this order. -/
inductive Launch where | first (programs : Nat) | reduce (tiles : Nat) | full (tiles : Nat)
  deriving DecidableEq, Repr

def launches (h : HostInput) : List Launch :=
  if h.m=0 ∨ h.n=0 then [] else
    (if (select h).streamTiles=0 then [] else [.first h.programs,.reduce (select h).streamTiles]) ++
    (if (select h).blockingTiles=0 then [] else [.full (select h).blockingTiles])

/-- Positive-grid condition follows from host control; zero/negative user P
maps to programs=0 before constructing this natural-number plan. -/
theorem launches_positive (h : HostInput) (l : Launch) (hl : l ∈ launches h) :
    match l with | .first n | .reduce n | .full n => 0 < n := by
  unfold launches at hl
  split_ifs at hl with he hs hb hb
  · simp at hl
  · simp at hl
  · simp only [List.nil_append,List.mem_singleton] at hl
    subst l
    omega
  · simp only [List.append_nil,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hl
    rcases hl with rfl|rfl
    · exact (selected_active h (by omega)).1
    · omega
  · simp only [List.mem_append,List.mem_cons,List.mem_singleton,List.not_mem_nil,or_false] at hl
    rcases hl with (rfl|rfl)|rfl
    · exact (selected_active h (by omega)).1
    · omega
    · omega

#axiomsClean selected_bounded
#axiomsClean selected_partition
#axiomsClean selected_active
#axiomsClean selected_divmod
#axiomsClean launches_positive
end Plan
end VeriTile.Bench.Repairs.StreamK.Whole001
