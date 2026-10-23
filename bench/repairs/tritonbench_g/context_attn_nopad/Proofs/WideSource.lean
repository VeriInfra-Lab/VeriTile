import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideNative
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideLogical
import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.HostSource

namespace VeriTile.Bench.Repairs.ContextAttnNopad.WideSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
open Logical HostPlan
noncomputable section

theorem execute_projects {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    {types : WideNative.MetadataTypes} {j : Job c} (hp : WideNative.Projects c r A s types j)
    (u : BlockState) : NativeProgram.execute c r A j u =
      exec (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel)
        (u.withGridIndex (index c j)) := by
  unfold NativeProgram.execute NativeProgram.atProgram
  rw [WideNative.nativePids_eq hp]
  rfl

/-- The existing ordinary grid merge is indexed through the proved bijection
from native chunks/local PIDs. Every merged frame is the result of that actual
native program. The final state is the complete disjoint-frame merge.
Integer projection, final backedges and each metadata element size are explicit. -/
structure Launched (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (types : WideNative.MetadataTypes) (final : BlockState) where
  grid : Kernel.GridLaunchedOrdinary
    (A.flattenKernel (c.source r Bool.false).eraseDType.toAlgKernel) c.grid (A.flattenState s) final
  projects : ∀ j : Job c, WideNative.Projects c r A s types j
  execution : ∀ j : Job c, NativeProgram.execute c r A j (A.flattenState s) =
    some (grid.frames (index c j)).final
  backedge : ∀ j : Job c, ∀ iter, iter ≤ (global j.2.2).val →
    (NativeProgram.word (iter*128) + NativeProgram.word 128).toInt = ((iter*128+128 : Nat) : Int)
  metadataBytes : ∀ b, b < c.batches →
    (NativeProgram.word b * NativeProgram.word types.start.bytes).toInt = ((b*types.start.bytes : Nat) : Int) ∧
    (NativeProgram.word b * NativeProgram.word types.length.bytes).toInt = ((b*types.length.bytes : Nat) : Int)

def Returns (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (types : WideNative.MetadataTypes) (result : BlockState × Option Unit) : Prop :=
  ∃ synchronizedMaximum : Nat, synchronizedMaximum = metadataMaximum c r s ∧
    Nonempty (Launched (NativeIndex.capCall c synchronizedMaximum) r A s types result.1) ∧
    result.2 = none

theorem from_logical_returns {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types) {result : BlockState × Option Unit}
    (hr : (effective c r s).Returns r A s Bool.false result) : Returns c r A s types result := by
  refine ⟨metadataMaximum c r s,rfl,?_,hr.2⟩
  obtain ⟨hg⟩ := hr.1
  let hp := fun j => (WideNative.complete_projection hm ha types fits j).1
  refine ⟨⟨hg,hp,?_,?_,?_⟩⟩
  · intro j
    exact (execute_projects (hp j) (A.flattenState s)).trans (hg.frames (index _ j)).h_exec
  · exact fun j => (WideNative.complete_projection hm ha types fits j).2
  · exact WideNative.metadata_byte_offsets hm types alloc

theorem effective_covered {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (b t : Nat) (hb : b < c.batches) (ht : t < c.length b) :
    t < (effective c r s).blocks*128 ↔ t < c.blocks*128 := by
  change t < ((NativeIndex.capMaxInputLen c.maxInputLen (metadataMaximum c r s)+127)/128)*128 ↔ _
  rw [WideDomain.metadataMaximum_eq hm]
  exact NativeIndex.cap_preserves_active c.maxInputLen (maximum c) t
    (lt_of_lt_of_le ht (length_le_maximum c b hb))

theorem written_effective {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (i : Coord c) : Written (effective c r s) i ↔ Written c i := by
  constructor
  · rintro ⟨b,t,hb,ht,hg,hr,hh,hd⟩
    exact ⟨b,t,hb,ht,(effective_covered hm b t hb ht).mp hg,hr,hh,hd⟩
  · rintro ⟨b,t,hb,ht,hg,hr,hh,hd⟩
    exact ⟨b,t,hb,ht,(effective_covered hm b t hb ht).mpr hg,hr,hh,hd⟩

def Contract (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (types : WideNative.MetadataTypes) (hs : WideDomain.Shape c)
    (input : Logical.Inputs) (view : OutputView c) (before : Nat → ℝ) : Prop :=
  ∃ raw, Returns c r A s types raw ∧
    let result := finish view before (Values c r A raw.1)
    result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧
    result.pythonReturn = none ∧
    (∀ b h t d hb hh ht hd, t < c.blocks*128 →
      view.read result.memory (WideLogical.outCoord hs b h t d hb hh ht hd) = expected c input b h t d) ∧
    (∀ addr, (¬ ∃ i, Written c i ∧ view.address i = addr) → result.memory addr = before addr) ∧
    BlockState.WriteWithin ((effective c r s).writes r A s) (A.flattenState s) raw.1 ∧
    (∀ rg ∈ c.inputRegions r, ∀ off, off < A.extent rg →
      raw.1.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off))

/-- Whole prepared int32/int64 candidate: actual metadata maximum, every
native job, exact pointer sizes and backedges, independent logical causal
target, original caller identities/None, and full caller/flat-memory frame. -/
theorem whole_entry_correct {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (ha : NativeProgram.ABI c A)
    (types : WideNative.MetadataTypes) (fits : WideNative.MetadataFits c types)
    (alloc : WideNative.MetadataAlloc r A types) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (input : Logical.Inputs) (hn : Normalizes c (inputs s r) input)
    (view : OutputView c) (hinj : Function.Injective view.address)
    (before : Nat → ℝ) (hi : FlatInitialized c r A s view before) :
    Contract c r A s types hm.shape input view before := by
  have hm' := WideNative.capMemory hm (metadataMaximum c r s)
  have hn' : Normalizes (effective c r s) (inputs s r) input := ⟨hn.query,hn.key,hn.value⟩
  have hinj' : Function.Injective (HostSource.effectiveView r s view).address := hinj
  have hi' : FlatInitialized (effective c r s) r A s (HostSource.effectiveView r s view) before := hi
  obtain ⟨raw,hr,_,_,_,ho,hf,hflat,hinputs⟩ :=
    WideLogical.whole_host_correct hm' hd hcov input hn' (HostSource.effectiveView r s view) hinj' before hi'
  refine ⟨raw,from_logical_returns hm ha types fits alloc hr,rfl,rfl,rfl,?_,?_,hflat,hinputs⟩
  · intro b h t d hb hh ht hdim hg
    exact ho b h t d hb hh ht hdim ((effective_covered hm b t hb ht).mpr hg)
  · intro addr hout
    apply hf addr
    rintro ⟨i,hw,he⟩
    exact hout ⟨i,(written_effective hm i).mp hw,he⟩

#print axioms whole_entry_correct
end
end VeriTile.Bench.Repairs.ContextAttnNopad.WideSource
