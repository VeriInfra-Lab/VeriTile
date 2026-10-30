import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.FPScatter
import bench.tritonbench_g.context_attn_llama.Proofs.Host

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPSuccess
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnLlama
noncomputable section
set_option maxHeartbeats 2000000
variable {α : Type} [Inhabited α]

def atIndex {g : Grid} (s : State α) (idx : GridIndex g) : State α :=
  {s with pids := idx.toPids, numPids := g.toNumPids}

def runGrid (c : Host.Call) (M : Algebra α) :
    List (GridIndex c.grid) → State α → Option (State α)
  | [], s => some s
  | idx :: rest, s => do
    let t ← exec (M.withDefaultPrecision .fp32)
      (kernel c.layout c.regions .fp32 (Host.scale c.layout.D)) (atIndex s idx)
    runGrid c M rest t

def allPrograms (c : Host.Call) : List (GridIndex c.grid) := Finset.univ.toList

theorem allPrograms_complete (c : Host.Call) (idx : GridIndex c.grid) :
    idx ∈ allPrograms c := by simp [allPrograms]

def Metadata (c : Host.Call) (s : State α) : Prop :=
  (∀ b, b < c.batches → (s.mem c.regions.start b).read .int = (c.start b : Int)) ∧
  (∀ b, b < c.batches → (s.mem c.regions.sequence b).read .int = ((c.length b+c.prompt b : Nat) : Int)) ∧
  (∀ b, b < c.batches → (s.mem c.regions.prompt b).read .int = (c.prompt b : Int))

def writes (c : Host.Call) (order : List (GridIndex c.grid)) (rg : RegionName) (o : Nat) : Prop :=
  ∃ idx ∈ order, ∃ i, c.active idx i ∧ rg = c.regions.out ∧ o = c.offset idx i


theorem active_matches (c : Host.Call) (s : State α) (hm : Metadata c s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    active c.layout c.regions (atIndex s idx) i ↔ c.active idx i := by
  have hb := Host.batch_bound c idx
  change idx.toPids 1 / c.layout.heads < c.batches at hb
  change ((c.layout.BM * idx.toPids 0 + i.1.val : Nat) : Int) <
    (s.mem c.regions.sequence (idx.toPids 1 / c.layout.heads)).read .int -
    (s.mem c.regions.prompt (idx.toPids 1 / c.layout.heads)).read .int ↔ _
  rw [hm.2.1 _ hb,hm.2.2 _ hb]
  simp only [Host.Call.active,Formula.queryRow,Formula.batch,Host.Call.launchAt]
  rw [Nat.mul_comm c.layout.BM (idx.toPids 0)]
  omega

theorem offset_matches (c : Host.Call) (s : State α) (hm : Metadata c s)
    (idx : GridIndex c.grid) (i : TileIndex [c.layout.BM,c.layout.D]) :
    offset c.layout c.regions (atIndex s idx) i = c.offset idx i := by
  have hb := Host.batch_bound c idx
  change idx.toPids 1 / c.layout.heads < c.batches at hb
  unfold offset start row atIndex
  simp only
  rw [hm.1 _ hb]
  simp only [Host.Call.offset,Host.Call.output,Formula.queryRow,Formula.batch,
    Formula.head,Host.Call.launchAt]
  norm_cast
  simp [Nat.mul_comm]
  rfl

end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.FPSuccess
