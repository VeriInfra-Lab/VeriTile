import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeLaunch

namespace VeriTile.Bench.Repairs.ContextAttnNopad.HostSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program
open VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open Logical HostPlan
noncomputable section
set_option maxHeartbeats 2000000

/-- Capping changes no tensor/view identity or coordinate descriptor. -/
def effectiveView {c : Call} (r : Regions) (s : BlockState) (v : OutputView c) :
    OutputView (effective c r s) :=
  ⟨v.objectId, v.allocationId, v.base, v.rowStride, v.headStride, v.dimStride, v.negative⟩

theorem written_effective {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (i : Coord c) : Written (effective c r s) i ↔ Written c i := by
  constructor
  · rintro ⟨b,t,hb,ht,hg,hr,hh,hd⟩
    exact ⟨b,t,hb,ht,(effective_covered hm b t hb ht).mp hg,hr,hh,hd⟩
  · rintro ⟨b,t,hb,ht,hg,hr,hh,hd⟩
    exact ⟨b,t,hb,ht,(effective_covered hm b t hb ht).mpr hg,hr,hh,hd⟩

/-- Public source host contract. The input c retains the user's original
max_input_len: metadata synchronization, capping, all native split axes and
all int64 casts are inside NativeLaunch.Returns, not caller premises. -/
def Contract (c : Call) (r : Regions) (A : FlatAlloc) (s : BlockState)
    (hs : Shape c) (a : Logical.Inputs) (view : OutputView c) (before : Nat → ℝ) : Prop :=
  ∃ raw, NativeLaunch.Returns c r A s raw ∧
    let result := finish view before (Values c r A raw.1)
    result.objectId = view.objectId ∧ result.allocationId = view.allocationId ∧
    result.pythonReturn = none ∧
    (∀ b h t d hb hh ht hd, t < c.blocks*128 →
      view.read result.memory (outCoord hs b h t d hb hh ht hd) = expected c a b h t d) ∧
    (∀ addr, (¬ ∃ i, Written c i ∧ view.address i = addr) → result.memory addr = before addr) ∧
    BlockState.WriteWithin ((effective c r s).writes r A s) (A.flattenState s) raw.1 ∧
    (∀ rg ∈ c.inputRegions r, ∀ off, off < A.extent rg →
      raw.1.mem A.flat (A.addr rg off) = (A.flattenState s).mem A.flat (A.addr rg off))

theorem whole_entry_correct {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : Memory c r A s) (ha : NativeProgram.ABI c A) (hd : A.Disjoint)
    (hcov : ∀ rg, rg ∉ A.regions → A.extent rg = 0)
    (a : Logical.Inputs) (hn : Normalizes c (inputs s r) a)
    (view : OutputView c) (hinj : Function.Injective view.address)
    (before : Nat → ℝ) (hi : FlatInitialized c r A s view before) :
    Contract c r A s hm.shape a view before := by
  have hm' := effective_memory hm
  have hn' : Normalizes (effective c r s) (inputs s r) a := ⟨hn.query,hn.key,hn.value⟩
  have hinj' : Function.Injective (effectiveView r s view).address := hinj
  have hi' : FlatInitialized (effective c r s) r A s (effectiveView r s view) before := hi
  obtain ⟨raw,hr,hid,halloc,hnone,ho,hf,hflat,hinputs⟩ :=
    whole_host_correct hm' hd hcov a hn' (effectiveView r s view) hinj' before hi'
  refine ⟨raw, NativeLaunch.from_logical_returns hm ha hr, rfl,rfl,rfl, ?_, ?_, hflat,hinputs⟩
  · intro b h t d hb hh ht hdim hg
    exact ho b h t d hb hh ht hdim ((effective_covered hm b t hb ht).mpr hg)
  · intro addr hout
    apply hf addr
    rintro ⟨i,hw,he⟩
    exact hout ⟨i,(written_effective hm i).mp hw,he⟩

theorem complete_readback {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hs : Shape c) {a : Logical.Inputs} {view : OutputView c} {before : Nat → ℝ}
    (hc : c.completeGrid) (h : Contract c r A s hs a view before) :
    ∃ raw, NativeLaunch.Returns c r A s raw ∧
      ∀ b head t d hb hh ht hd,
        view.read (finish view before (Values c r A raw.1)).memory
          (outCoord hs b head t d hb hh ht hd) = expected c a b head t d := by
  obtain ⟨raw,hr,_,_,_,ho,_⟩ := h
  refine ⟨raw,hr,?_⟩
  intro b head t d hb hh ht hd
  exact ho b head t d hb hh ht hd
    (lt_of_lt_of_le (lt_of_lt_of_le ht (hc b hb)) (blocks_cover c))

end
end VeriTile.Bench.Repairs.ContextAttnNopad.HostSource
