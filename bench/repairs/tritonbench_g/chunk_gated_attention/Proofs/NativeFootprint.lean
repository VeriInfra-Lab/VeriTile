import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeReadLocality

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open NativeValue
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

@[simp] theorem hits_nil (r : RegionName) (off : Nat) : ¬ Hits [] r off := by simp [Hits]
@[simp] theorem hits_append (left right : List Store) (r : RegionName) (off : Nat) :
    Hits (left++right) r off ↔ Hits left r off ∨ Hits right r off := by
  simp only [Hits,List.mem_append]
  constructor
  · rintro ⟨w,hl|hr,h⟩
    · exact Or.inl ⟨w,hl,h⟩
    · exact Or.inr ⟨w,hr,h⟩
  · rintro (⟨w,hl,h⟩|⟨w,hr,h⟩)
    · exact ⟨w,Or.inl hl,h⟩
    · exact ⟨w,Or.inr hr,h⟩

def HistoryHit (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (chunk : Nat) (r : RegionName) (off : Nat) : Prop :=
  r="h" ∧ ∃ i : HIndex c,off=historyOffset c base initial chunk i

theorem history_hits (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c → NativeGraph.Expr) (r : RegionName) (off : Nat) :
    Hits (history c base initial chunk acc) r off ↔ HistoryHit c base initial chunk r off := by
  simp only [Hits,history,List.mem_map,HistoryHit]
  constructor
  · rintro ⟨w,⟨i,_,rfl⟩,hr,ho⟩
    exact ⟨hr.symm,i,ho.symm⟩
  · rintro ⟨rfl,i,ho⟩
    exact ⟨_,⟨i,TileShape.mem_allIndices _ i,rfl⟩,rfl,ho.symm⟩

theorem final_hits (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (enabled : Bool) (acc : HIndex c → NativeGraph.Expr) (r : RegionName) (off : Nat) :
    Hits (final c base initial enabled acc) r off ↔
      enabled=Bool.true ∧ r="ht" ∧ ∃ i : HIndex c,off=stateOffset c base initial i := by
  cases enabled
  · simp [final]
  · simp only [final,↓reduceIte,Hits,List.mem_map,true_and]
    constructor
    · rintro ⟨w,⟨i,_,rfl⟩,hr,ho⟩
      exact ⟨hr.symm,i,ho.symm⟩
    · rintro ⟨rfl,i,ho⟩
      exact ⟨_,⟨i,TileShape.mem_allIndices _ i,rfl⟩,rfl,ho.symm⟩

theorem range_hits (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (gateK : Bool) (cur stop : Nat) (acc : Accumulator c) (r : RegionName) (off : Nat) :
    Hits (foldRange (innerStep c base initial gateK) cur stop 1 acc).2 r off ↔
      Hits acc.2 r off ∨ ∃ n,cur≤n ∧ n<stop ∧ HistoryHit c base initial n r off := by
  conv_lhs => rw [foldRange]
  simp only [one_ne_zero,↓reduceDIte]
  by_cases active : cur<stop
  · simp only [active,↓reduceDIte]
    rw [range_hits c base initial gateK (cur+1) stop (innerStep c base initial gateK cur acc) r off]
    simp only [innerStep,hits_append,history_hits]
    constructor
    · rintro ((old|now)|⟨n,hn,hs,hit⟩)
      · exact Or.inl old
      · exact Or.inr ⟨cur,le_rfl,active,now⟩
      · exact Or.inr ⟨n,by omega,hs,hit⟩
    · rintro (old|⟨n,hn,hs,hit⟩)
      · exact Or.inl (Or.inl old)
      · by_cases equal : n=cur
        · subst n; exact Or.inl (Or.inr hit)
        · exact Or.inr ⟨n,by omega,hs,hit⟩
  · simp only [active,↓reduceDIte]
    constructor
    · exact Or.inl
    · rintro (old|⟨n,hn,hs,_⟩)
      · exact old
      · omega
termination_by stop-cur
decreasing_by omega

theorem inner_hits (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) (r : RegionName) (off : Nat) :
    Hits (inner c base initial gateK hasSeed storeFinal) r off ↔
      (∃ n,n<c.NT ∧ HistoryHit c base initial n r off) ∨
      (storeFinal=Bool.true ∧ r="ht" ∧ ∃ i : HIndex c,off=stateOffset c base initial i) := by
  have reached := range_hits c base initial gateK 0 c.NT
    (NativeValue.initial algebra c base initial hasSeed,[]) r off
  simp only [hits_nil,false_or,Nat.zero_le,true_and] at reached
  simp only [inner,hits_append,final_hits]
  exact or_congr reached Iff.rfl

theorem prefix_hits (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) :
    Hits (prefixStores c base s) r off ↔ NativePrefix.enabled c base s ∧ r="o" ∧
      ∃ i : NativePrefix.Index c,off=NativePrefix.offset c base s i := by
  by_cases active : NativePrefix.enabled c base s
  · simp only [prefixStores,if_pos active,Hits,List.mem_map,active,true_and,↓reduceIte]
    constructor
    · rintro ⟨w,⟨i,_,rfl⟩,hr,ho⟩
      exact ⟨hr.symm,i,ho.symm⟩
    · rintro ⟨rfl,i,ho⟩
      exact ⟨_,⟨i,TileShape.mem_allIndices _ i,rfl⟩,rfl,ho.symm⟩
  · simp [prefixStores,active]

theorem inner_input_frame (c : NativeValue.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) (r : RegionName) (off : Nat)
    (history : r≠"h") (final : r≠"ht") :
    (NativeValue.result c base s gateK hasSeed storeFinal).mem r off=s.mem r off := by
  rw [inner_memory]
  apply frame
  rw [inner_hits]
  rintro (⟨_,_,hr,_⟩|⟨_,hr,_⟩)
  · exact history hr
  · exact final hr

theorem prefix_input_frame (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (r : RegionName) (off : Nat) (other : r≠"o") :
    (NativePrefix.written c base s).mem r off=s.mem r off := by
  rw [prefix_memory]
  apply frame
  rw [prefix_hits]
  rintro ⟨_,hr,_⟩
  exact other hr

#axiomsClean inner_hits
#axiomsClean prefix_hits
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
