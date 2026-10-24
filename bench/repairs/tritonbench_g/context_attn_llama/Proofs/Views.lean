import bench.repairs.tritonbench_g.context_attn_llama.Kernels
import bench.tritonbench_g.context_attn_llama.Proofs.HostReadback

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Views
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama VeriTile.Bench.TritonBenchG.ContextAttnLlama.Formula VeriTile.Bench.TritonBenchG.ContextAttnLlama.Program
noncomputable section

/-- The library contract is coordinate preservation, including a changed
materialized stride. These are input array equalities in the prepared coordinate
encoding, not output, execution, or numerical assumptions. Unused cells use the
same arbitrary extension. All eight actual read roles are present. -/
structure Reads (r : Regions) (s : BlockState) (logical : Inputs) : Prop where
  query : s.readMem r.query = logical.q
  key : s.readMem r.key = logical.k
  value : s.readMem r.value = logical.v
  start : s.readMemValue .int r.start = logical.start
  sequence : s.readMemValue .int r.sequence = logical.sequence
  prompt : s.readMemValue .int r.prompt = logical.prompt
  request : s.readMemValue .int r.request = logical.request
  pages : s.readMemValue .int r.pages = logical.pages

theorem Reads.inputs_eq {r : Regions} {s : BlockState} {a : Inputs} (h : Reads r s a) :
    inputs s r = a := by
  cases a
  cases h
  simp_all [inputs]

/-- `resolve_neg`/`contiguous` preserve the values at the same tensor coordinates;
the destination offset need not equal the original stride offset. -/
theorem coordinate_resolve {I : Type} (logical : I → ℝ) (offset : I → Nat)
    (resolved : Nat → ℝ) (contract : ∀ i, resolved (offset i) = logical i) :
    ∀ i, resolved (offset i) = logical i := contract

def sign (negative : Bool) (x : ℝ) : ℝ := if negative then -x else x
@[simp] theorem sign_involutive (negative : Bool) (x : ℝ) :
    sign negative (sign negative x) = x := by cases negative <;> simp [sign]

/-- Caller-owned physical cells. Inactive coordinates may alias active ones;
only active writers must be injective, just as in the original source domain. -/
structure Output (I : Type) where
  owner : Nat
  address : I → RegionName × Nat
  negative : Bool
  initial : (RegionName × Nat) → MemCell
  scratchOffset : I → Nat

def decode (cell : MemCell) : ℝ :=
  match cell.readAs .real with
  | some (some x) => x
  | _ => 0

def Output.logical {I : Type} (o : Output I) (mem : (RegionName × Nat) → MemCell) (i : I) : ℝ :=
  sign o.negative (decode (mem (o.address i)))

/-- copy_ is restricted to the source's covered rows, heads and channels. -/
def Output.copyBack {I : Type} (o : Output I) (active : I → Prop)
    (scratch : I → ℝ) : (RegionName × Nat) → MemCell := by
  classical
  exact fun cell => if h : ∃ i, active i ∧ o.address i = cell then
      MemCell.real (sign o.negative (scratch h.choose)) else o.initial cell

theorem Output.copyBack_read {I : Type} (o : Output I) (active : I → Prop)
    (scratch : I → ℝ)
    (unique : ∀ i j, active i → active j → o.address i = o.address j → i = j)
    (i : I) (hi : active i) :
    o.logical (o.copyBack active scratch) i = scratch i := by
  classical
  have h : ∃ j, active j ∧ o.address j = o.address i := ⟨i,hi,rfl⟩
  have hj : h.choose = i := unique _ _ h.choose_spec.1 hi h.choose_spec.2
  simp [Output.logical,Output.copyBack,h,hj,decode]

theorem Output.copyBack_frame {I : Type} (o : Output I) (active : I → Prop)
    (scratch : I → ℝ) (cell : RegionName × Nat)
    (hc : ¬ ∃ i, active i ∧ o.address i = cell) :
    o.copyBack active scratch cell = o.initial cell := by
  classical
  simp [Output.copyBack,hc]

def scratch {I : Type} (c : Host.Call) (A : FlatAlloc) (o : Output I) (final : BlockState) (i : I) : ℝ :=
  final.readMem A.flat (A.addr c.regions.out (o.scratchOffset i))

def active {I : Type} (c : Host.Call) (A : FlatAlloc) (o : Output I) (i : I) : Prop :=
  c.writes A (A.flat,A.addr c.regions.out (o.scratchOffset i))

/-- Every covered logical output coordinate is in the source's store footprint. -/
theorem indexed_active {I : Type} (c : Host.Call) (A : FlatAlloc) (o : Output I)
    (b : Fin c.batches) (h : Fin c.layout.heads) (row : Fin (c.length b))
    (d : Fin c.layout.D) (i : I)
    (offset : o.scratchOffset i = c.output b h row d)
    (hg : row.val < c.blocks*c.layout.BM) : active c A o i := by
  obtain ⟨idx,_,hi,hoff⟩ := Host.covered c b h row row.isLt hg d
  unfold active Host.Call.writes HostComposition.footprint HostComposition.writes
  refine ⟨idx,rfl,c.lane row d,hi,?_⟩
  rw [hoff,offset]

end
end VeriTile.Bench.Repairs.ContextAttnLlama.Views
