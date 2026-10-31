import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_KcacheCopyTriton
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Typed execution of the already selected n_tokens = 1, single split-x
source. Floating payloads are opaque; the model never uses real arithmetic.
The fp32 surface uses a precision-tagged load, whose raw-memory backend is
identity. All other numerical callbacks are arbitrary and unused. Metadata
retains the existing Nat address model, including its existing no-overflow
abstraction. This is not a proof of the unselected n_tokens > 1 branch. -/
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
open Lean VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

inductive Storage where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

def Storage.tag : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 | .fp64 => .real
  | .f8e4 => .f8e4 | .f8e5 => .f8e5

structure Parameters where
  (K KCache : RegionName)
  (blockTables lengths : Region .nat)
  (splitX strideKt strideKh strideKd strideKcb strideKch strideKcsplit
    strideKcs strideBts strideBtb blockSize width : Nat)

/-- Raw homogeneous copy: the explicit fp32 memory read preserves its payload.
This is a defined load backend, not a universal rounding identity assumption. -/
def copyBackend {α : Type} (rest : Algebra α) : Algebra α :=
  { rest with fp32Load := id }

def past {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.lengths.cast (s.pids 0)).read .nat - 1

def blockId {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem p.blockTables.cast
    (s.pids 0 * p.strideBts + past p s / p.blockSize * p.strideBtb)).read .nat

def inputOffset {α : Type} (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  s.pids 0 * p.strideKt + s.pids 1 * p.strideKh +
    (p.splitX * p.width + i.1.val) * p.strideKd

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Nat :=
  blockId p s * p.strideKcb + s.pids 1 * p.strideKch +
    p.splitX * p.strideKcsplit + past p s % p.blockSize * p.strideKcs + i.1.val

def loaded {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α)
    (i : TileIndex [p.width]) : Cell α :=
  .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag)

def copied {α : Type} [Inhabited α] (d : Storage) (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.width]).foldl
    (fun t i => t.write p.KCache (outputOffset p s i) (loaded d p s i)) s

theorem scatter_memory {α ι : Type} (xs : List ι) (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => t.write region (offset i) (value i)) s).mem =
      xs.foldl (fun mem i => fun r o =>
        if r = region ∧ o = offset i then value i else mem r o) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih => simp only [List.foldl_cons, ih]; rfl

@[simp] theorem masked_scatter_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    split <;> rfl

theorem masked_scatter_memory {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (region : RegionName)
    (offset : ι → Nat) (value : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write region (offset i) (value i)
      else t) s).mem =
      xs.foldl (fun mem i => if active i then fun r o =>
        if r = region ∧ o = offset i then value i else mem r o else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i <;> simp only [hi, ↓reduceIte] <;> rfl

/-- Readback needs injectivity only among active lanes, allowing arbitrary
inactive offsets and the one-active-lane case even with a zero stride. -/
theorem masked_readback_active {α : Type} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i)
    (hinj : ∀ a b, active a → active b → off a = off b → a = b) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i) = val i := by
  rw [State.masked_scatter_eq_filter]
  have himem : i ∈ (TileShape.allIndices shape).filter (fun j => decide (active j)) := by
    simp [TileShape.mem_allIndices, hi]
  obtain ⟨before, after, hl⟩ := List.append_of_mem himem
  have hnd := (TileShape.allIndices_nodup shape).filter (fun j => decide (active j))
  rw [hl, List.nodup_append, List.nodup_cons] at hnd
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hnd
  rw [hl, List.foldl_append, List.foldl_cons, State.scatter_frame]
  · exact State.write_same _ _ _ _
  · right
    intro j hj heq
    have hjmem : j ∈ (TileShape.allIndices shape).filter (fun j => decide (active j)) := by
      rw [hl]
      simp only [List.mem_append, List.mem_cons]
      exact Or.inr (Or.inr hj)
    have hjactive : active j := by simpa using (List.mem_filter.mp hjmem).2
    exact hnot ((hinj i j hi hjactive heq) ▸ hj)

 theorem output_injective {α : Type} [Inhabited α] (p : Parameters) (s : State α) :
    Function.Injective (outputOffset p s) := by
  intro i j h
  have hv : i.1.val = j.1.val := Nat.add_left_cancel h
  exact Prod.ext (Fin.ext hv) (Subsingleton.elim _ _)

end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
