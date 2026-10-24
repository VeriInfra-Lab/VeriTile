import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeWindowHits

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open NativeValue
noncomputable section
set_option maxHeartbeats 4000000

def Typed (writes : List Store) (r : RegionName) (d : TileDType) : Prop :=
  ∀ w∈writes,w.region=r → ∃ v : Value NativeGraph.Expr d,w.cell=.mk d v

theorem apply_typed (writes : List Store) (memory : Memory) (r : RegionName) (off : Nat)
    (d : TileDType) (typed : Typed writes r d) (hit : Hits writes r off) :
    ∃ v : Value NativeGraph.Expr d,apply writes memory r off=.mk d v := by
  classical
  induction writes generalizing memory with
  | nil => simp [Hits] at hit
  | cons w rest ih =>
    by_cases tail : Hits rest r off
    · exact ih (set w memory) (fun v hv=>typed v (List.mem_cons_of_mem w hv)) tail
    · rw [apply_cons,frame rest _ r off tail]
      obtain ⟨v,hv,hr,ho⟩ := hit
      rcases List.mem_cons.mp hv with equal|member
      · subst v
        obtain ⟨value,hvalue⟩ := typed w List.mem_cons_self hr
        exact ⟨value,by simp [set,hr,ho,hvalue]⟩
      · exact False.elim (tail ⟨v,member,hr,ho⟩)

theorem apply_tag (writes : List Store) (memory : Memory) (r : RegionName) (off : Nat)
    (d : TileDType) (typed : Typed writes r d) (hit : Hits writes r off) :
    apply writes memory r off=.mk d ((apply writes memory r off).read d) := by
  obtain ⟨v,cell⟩ := apply_typed writes memory r off d typed hit
  rw [cell,Cell.read_same]

def InnerTag (w : Store) : Prop :=
  (w.region="h" ∧ ∃ v : NativeGraph.Expr,w.cell=.mk .real v) ∨
  (w.region="ht" ∧ ∃ v : NativeGraph.Expr,w.cell=.mk .fp32 v)

theorem history_tag (c : NativeValue.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c → NativeGraph.Expr) :
    ∀ w∈history c base s chunk acc,InnerTag w := by
  intro w member
  obtain ⟨i,_,rfl⟩ := List.mem_map.mp member
  exact Or.inl ⟨rfl,_,rfl⟩

theorem final_tag (c : NativeValue.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (enabled : Bool) (acc : HIndex c → NativeGraph.Expr) :
    ∀ w∈final c base s enabled acc,InnerTag w := by
  cases enabled
  · simp [final]
  · intro w member
    obtain ⟨i,_,rfl⟩ := List.mem_map.mp member
    exact Or.inr ⟨rfl,_,rfl⟩

theorem range_tag (c : NativeValue.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (gateK : Bool) (cur stop : Nat) (acc : Accumulator c)
    (typed : ∀ w∈acc.2,InnerTag w) :
    ∀ w∈(foldRange (innerStep c base s gateK) cur stop 1 acc).2,InnerTag w := by
  intro w member
  rw [foldRange] at member
  simp only [one_ne_zero,↓reduceDIte] at member
  by_cases active : cur<stop
  · simp only [active,↓reduceDIte] at member
    apply range_tag c base s gateK (cur+1) stop (innerStep c base s gateK cur acc) ?_ w member
    intro v hv
    rcases List.mem_append.mp hv with old|new
    · exact typed v old
    · exact history_tag c base s cur acc.1 v new
  · exact typed w (by simpa only [active,↓reduceDIte] using member)
termination_by stop-cur
decreasing_by omega

theorem inner_tag (c : NativeValue.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) :
    ∀ w∈inner c base s gateK hasSeed storeFinal,InnerTag w := by
  intro w member
  rcases List.mem_append.mp member with hist|last
  · exact range_tag c base s gateK 0 c.NT (NativeValue.initial algebra c base s hasSeed,[])
      (by simp) w hist
  · exact final_tag c base s storeFinal _ w last

theorem prefix_tag (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr) :
    Typed (prefixStores c base s) "o" .fp32 := by
  intro w member _
  by_cases active : NativePrefix.enabled c base s
  · simp only [prefixStores,if_pos active] at member
    obtain ⟨i,_,rfl⟩ := List.mem_map.mp member
    exact ⟨_,rfl⟩
  · simp [prefixStores,active] at member

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section

theorem inner_all_typed (c : HostFP.Inner) (s : State NativeGraph.Expr) :
    NativeStores.Typed ((allJobs c.plans).flatMap fun j=>innerStores c j s) "h" .real ∧
    NativeStores.Typed ((allJobs c.plans).flatMap fun j=>innerStores c j s) "ht" .fp32 := by
  constructor
  · intro w member hr
    obtain ⟨j,_,hw⟩ := List.mem_flatMap.mp member
    rcases NativeStores.inner_tag _ _ _ _ _ _ w hw with ⟨_,typed⟩|⟨ht,_⟩
    · exact typed
    · exact False.elim (by simpa [hr] using ht)
  · intro w member hr
    obtain ⟨j,_,hw⟩ := List.mem_flatMap.mp member
    rcases NativeStores.inner_tag _ _ _ _ _ _ w hw with ⟨hh,_⟩|⟨_,typed⟩
    · exact False.elim (by simpa [hr] using hh)
    · exact typed

theorem prefix_all_typed (c : HostFP.Cumulative) (s : State NativeGraph.Expr) :
    NativeStores.Typed ((allJobs c.plans).flatMap fun j=>prefixStores c j s) "o" .fp32 := by
  intro w member hr
  obtain ⟨j,_,hw⟩ := List.mem_flatMap.mp member
  exact NativeStores.prefix_tag _ _ _ w hw hr

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
