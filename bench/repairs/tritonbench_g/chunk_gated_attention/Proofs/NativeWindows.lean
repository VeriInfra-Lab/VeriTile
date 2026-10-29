import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeJobs
import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeFootprint

/-! Finite source windows of all original grid programs. These address lists
include every reached source lane; native shape records remain unrestricted. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

def emptyState : State NativeGraph.Expr where
  mem := fun _ _ => .mk .real default
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 0
  undef := fun d _ _ => defaultValue d

def probe {plans : List LaunchPlan.Item} (j : Job plans) : State NativeGraph.Expr := atJob j emptyState

def keyReads (c : HostFP.Inner) (j : Job c.plans) : List Nat :=
  (List.range c.NT).flatMap fun n => (TileShape.allIndices [c.BK,c.BT]).map fun i =>
    NativeValue.keyOffset (NativeGrid.innerShape c) (plan j).base (probe j) n i
def valueReads (c : HostFP.Inner) (j : Job c.plans) : List Nat :=
  (List.range c.NT).flatMap fun n => (TileShape.allIndices [c.BT,c.BV]).map fun i =>
    NativeValue.valueOffset (NativeGrid.innerShape c) (plan j).base (probe j) n i
def gateReads (c : HostFP.Inner) (j : Job c.plans) : List Nat :=
  if c.gateK then keyReads c j ++ (List.range c.NT).flatMap (fun n =>
    (List.finRange c.BK).map fun i =>
      NativeValue.keyEndpoint (NativeGrid.innerShape c) (plan j).base (probe j) n i)
  else valueReads c j ++ (List.range c.NT).flatMap (fun n =>
    (List.finRange c.BV).map fun i =>
      NativeValue.valueEndpoint (NativeGrid.innerShape c) (plan j).base (probe j) n i)
def seedReads (c : HostFP.Inner) (j : Job c.plans) : List Nat :=
  if c.hasSeed then (TileShape.allIndices [c.BK,c.BV]).map fun i =>
    NativeValue.stateOffset (NativeGrid.innerShape c) (plan j).base (probe j) i else []
def historyWrites (c : HostFP.Inner) (j : Job c.plans) : List Nat :=
  (List.range c.NT).flatMap fun n => (TileShape.allIndices [c.BK,c.BV]).map fun i =>
    NativeValue.historyOffset (NativeGrid.innerShape c) (plan j).base (probe j) n i
def finalWrites (c : HostFP.Inner) (j : Job c.plans) : List Nat :=
  if c.storeFinal then (TileShape.allIndices [c.BK,c.BV]).map fun i =>
    NativeValue.stateOffset (NativeGrid.innerShape c) (plan j).base (probe j) i else []
def prefixCells (c : HostFP.Cumulative) (j : Job c.plans) : List Nat :=
  if NativePrefix.enabled (NativeGrid.prefixShape c) (plan j).base (probe j) then
    (TileShape.allIndices [c.BT,c.BS]).map fun i =>
      NativePrefix.offset (NativeGrid.prefixShape c) (plan j).base (probe j) i else []

def collect {plans : List LaunchPlan.Item} (f : Job plans → List Nat) : List Nat :=
  (allJobs plans).flatMap f

theorem mem_collect {plans : List LaunchPlan.Item} (f : Job plans → List Nat)
    (j : Job plans) {off : Nat} (h : off ∈ f j) : off ∈ collect f := by
  exact List.mem_flatMap.mpr ⟨j,by simp [allJobs],h⟩

theorem key_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (n : Nat) (hn : n<c.NT) (i : NativeValue.KIndex (NativeGrid.innerShape c)) :
    NativeValue.keyOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ keyReads c j := by
  apply List.mem_flatMap.mpr
  refine ⟨n,List.mem_range.mpr hn,?_⟩
  exact List.mem_map.mpr ⟨i,TileShape.mem_allIndices _ i,rfl⟩

theorem value_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (n : Nat) (hn : n<c.NT) (i : NativeValue.VIndex (NativeGrid.innerShape c)) :
    NativeValue.valueOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ valueReads c j := by
  exact List.mem_flatMap.mpr ⟨n,List.mem_range.mpr hn,
    List.mem_map.mpr ⟨i,TileShape.mem_allIndices _ i,rfl⟩⟩

theorem gate_key_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (hg : c.gateK=Bool.true) (n : Nat) (hn : n<c.NT)
    (i : NativeValue.KIndex (NativeGrid.innerShape c)) :
    NativeValue.keyOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ gateReads c j := by
  simp only [gateReads,hg,↓reduceIte,List.mem_append]
  exact Or.inl (key_mem c j s n hn i)

theorem gate_value_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (hg : c.gateK=Bool.false) (n : Nat) (hn : n<c.NT)
    (i : NativeValue.VIndex (NativeGrid.innerShape c)) :
    NativeValue.valueOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ gateReads c j := by
  simp only [gateReads,hg,Bool.false_eq_true,↓reduceIte,List.mem_append]
  exact Or.inl (value_mem c j s n hn i)

theorem key_end_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (hg : c.gateK=Bool.true) (n : Nat) (hn : n<c.NT) (i : Fin c.BK) :
    NativeValue.keyEndpoint (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ gateReads c j := by
  simp only [gateReads,hg,↓reduceIte,List.mem_append]
  exact Or.inr (List.mem_flatMap.mpr ⟨n,List.mem_range.mpr hn,
    List.mem_map.mpr ⟨i,List.mem_finRange i,rfl⟩⟩)

theorem value_end_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (hg : c.gateK=Bool.false) (n : Nat) (hn : n<c.NT) (i : Fin c.BV) :
    NativeValue.valueEndpoint (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ gateReads c j := by
  simp only [gateReads,hg,Bool.false_eq_true,↓reduceIte,List.mem_append]
  exact Or.inr (List.mem_flatMap.mpr ⟨n,List.mem_range.mpr hn,
    List.mem_map.mpr ⟨i,List.mem_finRange i,rfl⟩⟩)

theorem seed_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (hs : c.hasSeed=Bool.true) (i : NativeValue.HIndex (NativeGrid.innerShape c)) :
    NativeValue.stateOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) i ∈ seedReads c j := by
  simp only [seedReads,hs,↓reduceIte]
  exact List.mem_map.mpr ⟨i,TileShape.mem_allIndices _ i,rfl⟩

theorem history_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (n : Nat) (hn : n<c.NT) (i : NativeValue.HIndex (NativeGrid.innerShape c)) :
    NativeValue.historyOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) n i ∈ historyWrites c j := by
  exact List.mem_flatMap.mpr ⟨n,List.mem_range.mpr hn,
    List.mem_map.mpr ⟨i,TileShape.mem_allIndices _ i,rfl⟩⟩

theorem final_mem (c : HostFP.Inner) (j : Job c.plans) (s : State NativeGraph.Expr)
    (hf : c.storeFinal=Bool.true) (i : NativeValue.HIndex (NativeGrid.innerShape c)) :
    NativeValue.stateOffset (NativeGrid.innerShape c) (plan j).base (atJob j s) i ∈ finalWrites c j := by
  simp only [finalWrites,hf,↓reduceIte]
  exact List.mem_map.mpr ⟨i,TileShape.mem_allIndices _ i,rfl⟩

theorem prefix_mem (c : HostFP.Cumulative) (j : Job c.plans) (s : State NativeGraph.Expr)
    (enabled : NativePrefix.enabled (NativeGrid.prefixShape c) (plan j).base (atJob j s))
    (i : NativePrefix.Index (NativeGrid.prefixShape c)) :
    NativePrefix.offset (NativeGrid.prefixShape c) (plan j).base (atJob j s) i ∈ prefixCells c j := by
  simp only [prefixCells,if_pos (show NativePrefix.enabled _ _ (probe j) from enabled)]
  exact List.mem_map.mpr ⟨i,TileShape.mem_allIndices _ i,rfl⟩

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.KernelOnly.NativeInterface
