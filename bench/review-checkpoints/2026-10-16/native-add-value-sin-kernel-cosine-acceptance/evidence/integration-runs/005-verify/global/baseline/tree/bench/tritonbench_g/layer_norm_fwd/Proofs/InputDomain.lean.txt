import bench.tritonbench_g.layer_norm_fwd.Proofs.Domain
import bench.tritonbench_g.layer_norm_fwd.Proofs.Memory

/-! Explicit separation of active input regions from written regions. The host
allocates every written buffer separately from its inputs. Input buffers may
alias one another; inactive arguments impose no separation requirement. This
domain supplies the memory preservation facts needed to identify each later
load with the initial mathematical snapshot. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.Domain
open VeriTile.Triton
noncomputable section

def readRegion (f : Flags) (r : Regions) (region : RegionName) : Prop :=
  region = r.x ∨ region = r.weight ∨
  (f.x1 = Bool.true ∧ region = r.x1) ∨
  (f.residual = Bool.true ∧ region = r.residual) ∨
  (f.bias = Bool.true ∧ region = r.bias) ∨
  (f.rowscale = Bool.true ∧ region = r.rowscale) ∨
  (f.dropout = Bool.true ∧ region = r.seeds) ∨
  (f.w1 = Bool.true ∧ region = r.weight1) ∨
  (f.w1 = Bool.true ∧ f.b1 = Bool.true ∧ region = r.bias1)

def writeRegion (f : Flags) (r : Regions) (region : RegionName) : Prop :=
  region = r.y ∨ region = r.rstd ∨
  (f.rms = Bool.false ∧ region = r.mean) ∨
  (f.residualOut = Bool.true ∧ region = r.residualOut) ∨
  (f.w1 = Bool.true ∧ region = r.y1) ∨
  (f.dropout = Bool.true ∧ f.dropoutMask = Bool.true ∧ region = r.dropoutMask)

/-- A sufficient host-compatible alias domain, stated independently of
execution and of the output contract. Output-to-output aliases are handled
by the contract's explicit last-store order, not excluded by this predicate. -/
def ReadWriteSeparated (f : Flags) (r : Regions) : Prop :=
  ∀ input, readRegion f r input → ∀ output, writeRegion f r output → input ≠ output

/-- Exact tagged-cell agreement on every active input region. -/
def ReadsUnchanged (f : Flags) (r : Regions) (current initial : BlockState) : Prop :=
  ∀ region, readRegion f r region → ∀ offset,
    current.mem region offset = initial.mem region offset

theorem readsUnchanged_refl (f : Flags) (r : Regions) (s : BlockState) :
    ReadsUnchanged f r s s := by
  intro region hr offset
  rfl

theorem readsUnchanged_trans (f : Flags) (r : Regions) (a b c : BlockState)
    (hab : ReadsUnchanged f r a b) (hbc : ReadsUnchanged f r b c) :
    ReadsUnchanged f r a c := by
  intro region hr offset
  exact (hab region hr offset).trans (hbc region hr offset)

theorem readsUnchanged_of_mem (f : Flags) (r : Regions) (a b : BlockState)
    (h : a.mem = b.mem) : ReadsUnchanged f r a b := by
  intro region hr offset
  rw [h]

theorem readsUnchanged_readMem (f : Flags) (r : Regions) (a b : BlockState)
    (h : ReadsUnchanged f r a b) (region : RegionName)
    (hr : readRegion f r region) (offset : Nat) :
    a.readMem region offset = b.readMem region offset := by
  simp only [BlockState.readMem, h region hr offset]

theorem readsUnchanged_seed (f : Flags) (r : Regions) (a b : BlockState)
    (h : ReadsUnchanged f r a b) (hd : f.dropout = Bool.true) (offset : Nat) :
    a.readMemValue .int r.seeds offset = b.readMemValue .int r.seeds offset := by
  have hr : readRegion f r r.seeds := by simp [readRegion, hd]
  simp only [BlockState.readMemValue, BlockState.readMemTyped, h r.seeds hr offset]

theorem writeCell_readsUnchanged (f : Flags) (r : Regions)
    (hsep : ReadWriteSeparated f r) (output : RegionName)
    (hw : writeRegion f r output) (offset : Nat) (value : MemCell) (s : BlockState) :
    ReadsUnchanged f r (Memory.writeCell s output offset value) s := by
  intro region hr off
  simp only [Memory.writeCell_mem, hsep region hr output hw, false_and, if_false]

theorem scatter_readsUnchanged (f : Flags) (r : Regions)
    (hsep : ReadWriteSeparated f r) (output : RegionName)
    (hw : writeRegion f r output) (start N block : Nat) (hN : N ≤ block)
    (values : Nat → MemCell) (s : BlockState) :
    ReadsUnchanged f r (Memory.scatter output start N block values s) s := by
  intro region hr offset
  exact Memory.scatter_frame output start N block hN values s region offset
    (by simp only [hsep region hr output hw, false_and, not_false_eq_true])

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.Domain
