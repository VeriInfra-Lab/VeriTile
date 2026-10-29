import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeGrid

/-! Finite ordinary stores extracted from the existing independent native
recurrences. These are arithmetic/reference data, not a second executor. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (scatter_memory)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

abbrev Memory := RegionName → Nat → Cell NativeGraph.Expr
structure Store where
  region : RegionName
  offset : Nat
  cell : Cell NativeGraph.Expr

def set (write : Store) (memory : Memory) : Memory := fun r off =>
  if r=write.region ∧ off=write.offset then write.cell else memory r off
def apply (writes : List Store) (memory : Memory) : Memory := writes.foldl (fun m w => set w m) memory
def Hits (writes : List Store) (r : RegionName) (off : Nat) : Prop :=
  ∃ w ∈ writes, w.region=r ∧ w.offset=off

@[simp] theorem apply_nil (memory : Memory) : apply [] memory=memory := rfl
@[simp] theorem apply_cons (w : Store) (rest : List Store) (memory : Memory) :
    apply (w::rest) memory = apply rest (set w memory) := rfl
@[simp] theorem apply_append (a b : List Store) (memory : Memory) :
    apply (a++b) memory = apply b (apply a memory) := List.foldl_append

theorem apply_scatter {ι : Type} (xs : List ι) (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell NativeGraph.Expr) (s : State NativeGraph.Expr) :
    (xs.foldl (fun t i => t.write region (offset i) (value i)) s).mem =
      apply (xs.map fun i => ⟨region,offset i,value i⟩) s.mem := by
  rw [scatter_memory]
  simp only [apply,List.foldl_map]
  rfl

theorem frame (writes : List Store) (memory : Memory) (r : RegionName) (off : Nat)
    (outside : ¬ Hits writes r off) : apply writes memory r off=memory r off := by
  induction writes generalizing memory with
  | nil => rfl
  | cons w rest ih =>
    rw [apply_cons,ih _ (by rintro ⟨v,hv,hr,ho⟩; exact outside ⟨v,List.mem_cons_of_mem w hv,hr,ho⟩)]
    simp only [set]
    split
    · rename_i hit
      exact False.elim (outside ⟨w,List.mem_cons_self,hit.1.symm,hit.2.symm⟩)
    · rfl

theorem at_hit (writes : List Store) (left right : Memory) (r : RegionName) (off : Nat)
    (hit : Hits writes r off) : apply writes left r off=apply writes right r off := by
  classical
  induction writes generalizing left right with
  | nil => simp [Hits] at hit
  | cons w rest ih =>
    by_cases tail : Hits rest r off
    · exact ih _ _ tail
    · rw [apply_cons,apply_cons,frame rest _ r off tail,frame rest _ r off tail]
      rcases hit with ⟨v,hv,hr,ho⟩
      rcases List.mem_cons.mp hv with equal|member
      · subst v
        simp [set,hr,ho]
      · exact False.elim (tail ⟨v,member,hr,ho⟩)

theorem pointwise (writes : List Store) (left right : Memory) (r : RegionName) (off : Nat)
    (equal : left r off=right r off) : apply writes left r off=apply writes right r off := by
  classical
  by_cases hit : Hits writes r off
  · exact at_hit writes left right r off hit
  · rw [frame writes left r off hit,frame writes right r off hit,equal]

open NativeValue

def history (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c → NativeGraph.Expr) : List Store :=
  (TileShape.allIndices [c.BK,c.BV]).map fun i =>
    ⟨"h",historyOffset c base initial chunk i,.mk .real (.cast (some .fp32) .real .real (acc i))⟩
def final (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (enabled : Bool) (acc : HIndex c → NativeGraph.Expr) : List Store :=
  if enabled then (TileShape.allIndices [c.BK,c.BV]).map fun i =>
    ⟨"ht",stateOffset c base initial i,.mk .fp32 (.cast (some .fp32) .real .fp32 (acc i))⟩
  else []

abbrev Accumulator (c : NativeValue.Shape) := (HIndex c → NativeGraph.Expr) × List Store
def innerStep (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr) (gateK : Bool)
    (chunk : Nat) (acc : Accumulator c) : Accumulator c :=
  (update c base initial gateK chunk acc.1,acc.2++history c base initial chunk acc.1)
def inner (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) : List Store :=
  let result := foldRange (innerStep c base initial gateK) 0 c.NT 1
    (NativeValue.initial algebra c base initial hasSeed,[])
  result.2++final c base initial storeFinal result.1

theorem history_memory (c : NativeValue.Shape) (base : Nat) (initial current : State NativeGraph.Expr)
    (chunk : Nat) (acc : HIndex c → NativeGraph.Expr) :
    (historyWrite c base initial current chunk acc).mem=apply (history c base initial chunk acc) current.mem :=
  apply_scatter _ _ _ _ _

theorem final_memory (c : NativeValue.Shape) (base : Nat) (initial current : State NativeGraph.Expr)
    (enabled : Bool) (acc : HIndex c → NativeGraph.Expr) :
    (finalWrite c base initial current enabled acc).mem=apply (final c base initial enabled acc) current.mem := by
  cases enabled
  · rfl
  · exact apply_scatter _ _ _ _ _

theorem range_memory (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (gateK : Bool) (cur stop : Nat) (acc : Accumulator c) :
    foldRange (referenceStep c base initial gateK) cur stop 1 (acc.1,apply acc.2 initial.mem) =
      let result := foldRange (innerStep c base initial gateK) cur stop 1 acc
      (result.1,apply result.2 initial.mem) := by
  conv_lhs => rw [foldRange]
  conv_rhs => rw [foldRange]
  simp only [one_ne_zero,↓reduceDIte]
  split_ifs with active
  · have h := range_memory c base initial gateK (cur+1) stop (innerStep c base initial gateK cur acc)
    simpa only [innerStep,referenceStep,apply_append,history_memory] using h
  · rfl
termination_by stop-cur
decreasing_by omega

theorem inner_memory (c : NativeValue.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) :
    (NativeValue.result c base s gateK hasSeed storeFinal).mem =
      apply (inner c base s gateK hasSeed storeFinal) s.mem := by
  have h := range_memory c base s gateK 0 c.NT (NativeValue.initial algebra c base s hasSeed,[])
  change reference c base s gateK hasSeed = _ at h
  simp only [NativeValue.result,h,final_memory,inner,apply_append]

def prefixStores (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr) : List Store :=
  if NativePrefix.enabled c base s then
    (TileShape.allIndices [c.BT,c.BS]).map fun i =>
      ⟨"o",NativePrefix.offset c base s i,.mk .fp32 (NativePrefix.value c base s i)⟩
  else []

theorem prefix_memory (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr) :
    (NativePrefix.written c base s).mem=apply (prefixStores c base s) s.mem := by
  by_cases active : NativePrefix.enabled c base s
  · simp only [NativePrefix.written,prefixStores,if_pos active]
    exact apply_scatter _ _ _ _ _
  · simp [NativePrefix.written,prefixStores,active]

#specNonCircular inner avoiding [FP.Structural.exec,FP.Structural.run]
#specNonCircular prefixStores avoiding [FP.Structural.exec,FP.Structural.run]
#axiomsClean inner_memory
#axiomsClean prefix_memory
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeStores
