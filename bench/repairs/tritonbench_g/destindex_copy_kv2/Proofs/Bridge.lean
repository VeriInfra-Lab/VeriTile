import bench.repairs.tritonbench_g.destindex_copy_kv2.Contract
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.SourceAgreement
import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.PhysicalCorrect

namespace VeriTile.Bench.Repairs.DestindexCopyKv2
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

theorem checkInputs_iff (m : Inputs) (w : Nat → Word) :
    checkInputs m w=.ok () ↔ InputContract m w := by
  have common := DestindexCopyKv1.checkInputs_iff m w
  cases he : DestindexCopyKv1.checkInputs m w with
  | error e =>
    have rejected : ¬DestindexCopyKv1.InputContract m w := by intro h; simpa [he] using common.mpr h
    simp only [checkInputs,he]
    constructor
    · intro h; cases h
    · intro h; exact False.elim (rejected h.common)
  | ok result =>
    cases result
    have admitted := common.mp he
    simp only [checkInputs,he]
    constructor
    · intro h
      have hw : Width m := by split at h <;> simp_all
      exact ⟨admitted,hw⟩
    · intro h; simp [h.width]

theorem checkArguments_iff (m : Option Inputs) (w : Nat → Word) :
    checkArguments m w=.ok () ↔ Admitted m w := by
  cases m <;> simp [checkArguments,Admitted,checkInputs_iff]

namespace Physical
abbrev Lane := DestindexCopyKv1.Physical.Lane

def lanes (c : Config) : List (Lane c) :=
  (TileShape.allIndices [block c.heads,block c.dims]).filter
    (fun i => (BitVec.ofNat 32 i.1.val).slt (BitVec.ofNat 32 c.heads))
def program (m : Inputs) (p : Nat) (s : Memory) : Memory :=
  DestindexCopyKv1.Physical.stores (DestindexCopyKv1.Physical.outputAddress m p s)
    (fun i => s (DestindexCopyKv1.Physical.sourceAddress m p i)) (lanes (config m)) s
def launch (m : Inputs) (ps : List Nat) (s : Memory) : Memory :=
  ps.foldl (fun t p => program m p t) s

theorem lanes_eq (m : Inputs) (w : Nat → Word) (h : InputContract m w) (ne : Nonempty m) :
    lanes (config m)=DestindexCopyKv1.Physical.lanes (config m) := by
  apply List.filter_congr
  intro i _
  have bounds := DestindexCopyKv1.Native.block_bounds m w h.common
  change block (config m).heads≤2^20 ∧ block (config m).dims≤2^20 at bounds
  have dim : i.2.1.val<(config m).dims := lt_of_lt_of_le i.2.1.isLt (le_of_eq (h.width ne))
  have heads := Nat.le_pow_clog (by decide : 1<2) (config m).heads
  change (config m).heads≤block (config m).heads at heads
  have headLane : i.1.val<block (config m).heads := i.1.isLt
  rw [DestindexCopyKv1.Native.mask_exact m w h.common _ _ i.1.isLt i.2.1.isLt,
    BitVec.slt_eq_decide,DestindexCopyKv1.Native.signed32 _ (by omega),
    DestindexCopyKv1.Native.signed32 _ (by omega)]
  simp [dim]

theorem program_eq (m : Inputs) (w : Nat → Word) (h : InputContract m w) (ne : Nonempty m)
    (p : Nat) (s : Memory) : program m p s=DestindexCopyKv1.Physical.program m p s := by
  unfold program DestindexCopyKv1.Physical.program
  rw [lanes_eq m w h ne]

theorem launch_eq (m : Inputs) (w : Nat → Word) (h : InputContract m w) (ne : Nonempty m)
    (ps : List Nat) (s : Memory) : launch m ps s=DestindexCopyKv1.Physical.launch m ps s := by
  unfold launch DestindexCopyKv1.Physical.launch
  congr 1
  funext t p
  exact program_eq m w h ne p t

/-- Actual kv2 typed program is linked to the mutable finite-pointer source.
The common grid proof is reused only after the selected sources' effects agree. -/
theorem program_linked (rest : Algebra Word) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (ne : Nonempty m) (initial current : Memory)
    (binding : TensorBinding m w initial) (reads : DestindexCopyKv1.Physical.Reads m w initial current)
    (s : State Word)
    (bound : DestindexCopyKv1.Grid.Bound s (fun x => initial (addr (base m .source) x)) (destination w))
    (link : DestindexCopyKv1.Physical.Linked m s current) (p : Nat) (hp : p<(config m).count) :
    DestindexCopyKv1.Physical.Linked m
      (Tile.execution rest (layout (config m)) {s with pids := fun d => if d=0 then p else s.pids d}).val
      (program m p current) := by
  intro x
  rw [same_source_memory rest _ (h.width ne).symm,program_eq m w h ne]
  exact DestindexCopyKv1.Physical.program_linked rest m w h.common initial current binding reads
    s bound link p hp x
end Physical

#axiomsClean same_source_memory
#axiomsClean checkArguments_iff
#axiomsClean Physical.program_linked
#axiomsClean Physical.launch_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2
