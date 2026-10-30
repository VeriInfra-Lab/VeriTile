import bench.repairs.tritonbench_g.int4_matmul.Proofs.Memory
import bench.repairs.tritonbench_g.int4_matmul.Proofs.Bounds
namespace VeriTile.Bench.Repairs.Int4Matmul
set_option maxHeartbeats 4000000
abbrev LaneIndex (s : Shape) (t : Tile) :=
  Fin (blocksM s t*blocksN s t) × Fin t.splits × Fin t.bm × Fin t.bn

def toLane (s : Shape) (t : Tile) (i : LaneIndex s t) : Lane :=
  ⟨i.1,i.2.1,i.2.2.1,i.2.2.2⟩
noncomputable def rawLanes (s : Shape) (t : Tile) : List Lane :=
  (Finset.univ : Finset (LaneIndex s t)).toList.map (toLane s t)

theorem toLane_injective (s : Shape) (t : Tile) : Function.Injective (toLane s t) := by
  rintro ⟨a,b,c,d⟩ ⟨e,f,g,h⟩ he
  simp only [toLane,Lane.mk.injEq] at he
  rcases he with ⟨ha,hb,hc,hd⟩
  simp_all [Fin.ext_iff]

theorem raw_mem (s : Shape) (t : Tile) (l : Lane) : l∈rawLanes s t ↔ l.Valid s t := by
  simp only [rawLanes,List.mem_map,Finset.mem_toList,Finset.mem_univ,true_and]
  constructor
  · rintro ⟨i,rfl⟩;exact ⟨i.1.isLt,i.2.1.isLt,i.2.2.1.isLt,i.2.2.2.isLt⟩
  · rintro ⟨hp,hs,hi,hj⟩
    exact ⟨(⟨l.pid,hp⟩,⟨l.split,hs⟩,⟨l.i,hi⟩,⟨l.j,hj⟩),rfl⟩

noncomputable def sourceSchedule (s : Shape) (t : Tile) : List Work :=
  ((rawLanes s t).filter (fun l => decide (s.Active t (laneWork s t l)))).map (laneWork s t)

theorem sourceSchedule_mem (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid) (w : Work) :
    w∈sourceSchedule s t ↔ s.Active t w := by
  simp only [sourceSchedule,List.mem_map,List.mem_filter,decide_eq_true_eq,raw_mem]
  constructor
  · rintro ⟨l,⟨hl,ha⟩,rfl⟩;exact ha
  · intro hw
    obtain ⟨l,hl,he⟩ := lane_coverage s t hs ht w hw
    exact ⟨l,⟨hl,by simpa [he] using hw⟩,he⟩

theorem sourceSchedule_nodup (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid) :
    (sourceSchedule s t).Nodup := by
  have raw : (rawLanes s t).Nodup :=
    (List.nodup_map_iff (toLane_injective s t)).mpr (Finset.nodup_toList _)
  apply (List.nodup_map_iff_inj_on (raw.filter _)).mpr
  intro a ha b hb he
  exact lane_injective s t hs ht a b
    ((raw_mem s t a).mp (List.mem_filter.mp ha).1)
    ((raw_mem s t b).mp (List.mem_filter.mp hb).1) he

/-- Permutations represent arbitrary complete atomic/store serialization. The concrete
source schedule's coverage and unique ownership were proved above. -/
def Complete (s : Shape) (t : Tile) (ws : List Work) :=
  ws.Perm (sourceSchedule s t)

theorem complete_active (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid)
    (ws : List Work) (h : Complete s t ws) (w : Work) : w∈ws ↔ s.Active t w :=
  h.mem_iff.trans (sourceSchedule_mem s t hs ht w)

def call (s : Shape) (t : Tile) (p : Ops α) (ws : List Work) (initial : State α) :=
  execute s t p ws (zeroOutput s p initial)

theorem call_result (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid) (p : Ops α)
    (ws : List Work) (complete : Complete s t ws) (initial : State α)
    (r c : Nat) (hr : r<s.m) (hc : c<s.n) :
    (call s t p ws initial).values .output (cAddress s r c)=
      referenceOutput s t p initial r c ws p.zero := by
  unfold call
  rw [execute_result s t p ws initial (zeroOutput s p initial) (zero_readonly s p initial)
    (fun w hw => (complete_active s t hs ht ws complete w).mp hw) r c hc]
  simp [zeroOutput,cAddress,rowMajor_bound s.m s.n r c hr hc]

theorem call_frame (s : Shape) (t : Tile) (hs : s.Valid) (ht : t.Valid) (p : Ops α)
    (ws : List Work) (complete : Complete s t ws) (initial : State α)
    (r : Region) (a : Nat) (outside : r≠.output ∨ s.m*s.n≤a) :
    (call s t p ws initial).values r a=initial.values r a := by
  unfold call
  rw [execute_frame]
  · rcases outside with h|h
    · simp [zeroOutput,h]
    · simp [zeroOutput,not_lt_of_ge h]
  · intro w hw ⟨hr,ha⟩
    have active := (complete_active s t hs ht ws complete w).mp hw
    have bound := rowMajor_bound s.m s.n w.row w.col active.1 active.2.1
    rcases outside with h|h
    · exact h hr
    · change w.row*s.n+w.col<s.m*s.n at bound
      unfold cAddress at ha
      omega
#axiomsClean sourceSchedule_mem
#axiomsClean sourceSchedule_nodup
#axiomsClean call_result
#axiomsClean call_frame
end VeriTile.Bench.Repairs.Int4Matmul
