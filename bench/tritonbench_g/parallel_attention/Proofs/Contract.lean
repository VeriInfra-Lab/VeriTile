import bench.tritonbench_g.parallel_attention.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Source max(16,min(128,next_power_of_2(n))) on positive dimensions. -/
def selectedTile (n : Nat) := if n ≤ 16 then 16 else if n ≤ 32 then 32 else if n ≤ 64 then 64 else 128

theorem selectedTile_positive (n : Nat) : 0 < selectedTile n := by
  unfold selectedTile
  split_ifs <;> norm_num

theorem selectedTile_fits (n : Nat) (hn : n ≤ 128) : n ≤ selectedTile n := by
  unfold selectedTile
  split_ifs <;> omega

/-- Revised public dense FP32 shape domain; zero B/H/T are logically permitted. -/
structure ShapeContract (s : Shape) : Prop where
  key_positive : 0 < s.K
  key_limit : s.K ≤ 128
  value_positive : 0 < s.V
  key_tile : s.BK=selectedTile s.K
  value_tile : s.BV=selectedTile s.V

theorem ShapeContract.key_fits (s : Shape) (h : ShapeContract s) : s.K ≤ s.BK := by
  rw [h.key_tile]
  exact selectedTile_fits s.K h.key_limit

theorem ShapeContract.value_tile_positive (s : Shape) (h : ShapeContract s) : 0 < s.BV := by
  rw [h.value_tile]
  exact selectedTile_positive s.V

/-- Only memory layout and finite schedule obligations. No desired value,
completed output, whole numerical reduction or successful-run premise. -/
structure RunLayout (s : Shape) (b : Bases) (trees : OutputKind → SumPlan)
    (scratch output : OutputKind → Nat) (kinds : List OutputKind) (ks hs : List Event) : Prop where
  kernelCoverage : ks.Perm (events s true kinds)
  hostCoverage : hs.Perm (events s false kinds)
  kernelFresh : ∀ e∈ks,tensorAddress scratch e∉inputCells s b
  kernelUnique : ∀ e∈ks,∀ f∈ks,tensorAddress scratch e=tensorAddress scratch f → e=f
  hostFresh : ∀ e∈hs,tensorAddress output e∉allScratchCells s scratch kinds
  hostUnique : ∀ e∈hs,∀ f∈hs,tensorAddress output e=tensorAddress output f → e=f
  hostTrees : ∀ kind∈kinds,(trees kind).Covers (kind.shards s)

def forwardKinds : List OutputKind := [.forward,.normalizer]
def backwardKinds : List OutputKind := [.query,.key,.value]

#axiomsClean selectedTile_positive
#axiomsClean selectedTile_fits
end
end VeriTile.Bench.Repairs.ParallelAttention
