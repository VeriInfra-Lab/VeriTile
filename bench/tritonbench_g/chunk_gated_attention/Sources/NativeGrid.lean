import bench.tritonbench_g.chunk_gated_attention.Sources.NativePrefix
import bench.tritonbench_g.chunk_gated_attention.Sources.HostFP

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeGrid
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000

abbrev State := FP.Structural.State NativeGraph.Expr

def program (source : Nat→ComputeKernel) (base : Nat) : Program := ⟨source base,[]⟩

/-- Deterministic disjoint-output merge of independent per-program recurrences.
The numerical reference is already a complete scalar graph, not a callback
provided by the allocator or numerical interpretation. -/
def merge (reference : Nat→State→State) (owns : (p : LaunchPlan.Item)→HostFP.Ownership p.size)
    (p : LaunchPlan.Item) (s : State) : State := by
  classical
  exact {s with mem:=fun r off=>if h : ∃ i,owns p i r off then
    (reference p.base (HostFP.gridState s p.size (Classical.choose h).val)).mem r off else s.mem r off}

theorem launch (source : Nat→ComputeKernel) (reference : Nat→State→State)
    (complete : ∀ base s,∃ t,FP.Structural.exec (engine NativeValue.algebra) (source base) s=some t ∧
      t.mem=(reference base s).mem)
    (owns : (p : LaunchPlan.Item)→HostFP.Ownership p.size) (p : LaunchPlan.Item) (s : State) :
    HostFP.launch NativeValue.algebra (program source) owns p s=some (merge reference owns p s) := by
  classical
  have success (i : Fin p.size) :
      (FP.Structural.exec (engine NativeValue.algebra) (source p.base) (HostFP.gridState s p.size i.val)).isSome := by
    obtain ⟨t,exec,_⟩:=complete p.base (HostFP.gridState s p.size i.val)
    rw [exec]
    rfl
  unfold HostFP.launch HostFP.merge program
  rw [dif_pos success]
  apply congrArg some
  congr 1
  funext r off
  dsimp only [merge]
  split
  · rename_i ownsCell
    obtain ⟨t,exec,memory⟩:=complete p.base (HostFP.gridState s p.size (Classical.choose ownsCell).val)
    simp only [exec,Option.get_some]
    exact congrFun (congrFun memory r) off
  · rfl

def referenceBatches (reference : Nat→State→State) (owns : (p : LaunchPlan.Item)→HostFP.Ownership p.size) :
    List LaunchPlan.Item→State→State
  | [],s=>s
  | p::ps,s=>referenceBatches reference owns ps (merge reference owns p s)

theorem batches (source : Nat→ComputeKernel) (reference : Nat→State→State)
    (complete : ∀ base s,∃ t,FP.Structural.exec (engine NativeValue.algebra) (source base) s=some t ∧
      t.mem=(reference base s).mem)
    (owns : (p : LaunchPlan.Item)→HostFP.Ownership p.size) (plans : List LaunchPlan.Item) (s : State) :
    HostFP.batches NativeValue.algebra (program source) owns plans s=some (referenceBatches reference owns plans s) := by
  induction plans generalizing s with
  | nil=>rfl
  | cons p ps ih=>simp only [HostFP.batches,launch source reference complete,Option.bind_some,referenceBatches,ih]

def prefixShape (c : HostFP.Cumulative) : NativePrefix.Shape := ⟨c.PT,c.PS,c.BT,c.BS,c.heads⟩

def prefixSource (c : HostFP.Cumulative) (base : Nat) := NativePrefix.source (prefixShape c) base
def prefixReference (c : HostFP.Cumulative) (base : Nat) (s : State) := NativePrefix.written (prefixShape c) base s

/-- Region names are fixed to the same literal roles as the typed source. -/
def prefixOwns (c : HostFP.Cumulative) := ({c with input:="s",output:="o"}).owns

def prefixResult (c : HostFP.Cumulative) (s : State) := referenceBatches (prefixReference c) (prefixOwns c) c.plans s

theorem prefix_complete (c : HostFP.Cumulative) (s : State) :
    HostFP.batches NativeValue.algebra (program (prefixSource c)) (prefixOwns c) c.plans s=some (prefixResult c s) :=
  batches _ _ (fun base s=>NativePrefix.source_memory _ base s) _ _ _

#specNonCircular prefixResult avoiding [FP.Structural.exec,FP.Structural.run,FP.Structural.range,HostFP.batches]
#axiomsClean prefix_complete
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeGrid
