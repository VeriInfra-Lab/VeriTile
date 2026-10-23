import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.WideLogical

/-! Same-object output is an observation of the kernel's actual allocation.
The local identity contract binds allocation identity, coordinate placement and
all initial allocation cells. The final memory equation is proved from the
source write footprint; it is not supplied by the ordinary tensor API. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.DirectOutput
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program Formula
open Logical
noncomputable section

structure Binding {α : Type} (c : Call) (r : Regions) (A : FlatAlloc)
    (address : Coord c → Nat) (callerAllocation kernelAllocation : Nat)
    (initial before : Nat → α) : Prop where
  allocation : kernelAllocation = callerAllocation
  placement : ∀ i, address i = A.addr r.out (coordOffset c i)
  initialMemory : ∀ off, off < A.extent r.out → initial off = before (A.addr r.out off)

/-- Read the actual output allocation. Outside that allocation, the caller's
previous memory is retained; no coordinate copy is executed by this observer. -/
def observe {α : Type} (r : Regions) (A : FlatAlloc) (before raw : Nat → α) : Nat → α :=
  fun addr => if A.base r.out ≤ addr ∧ addr < A.base r.out + A.extent r.out
    then raw (addr - A.base r.out) else before addr

theorem observe_at {α : Type} (r : Regions) (A : FlatAlloc) (before raw : Nat → α)
    (off : Nat) (h : off < A.extent r.out) :
    observe r A before raw (A.addr r.out off) = raw off := by
  simp only [observe,FlatAlloc.addr]
  rw [if_pos (by omega),Nat.add_sub_cancel_left]

theorem write_has_coordinate {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}
    (hm : WideDomain.Memory c r A s) (off : Nat)
    (hw : c.writes r A s (A.flat,A.addr r.out off)) :
    ∃ i : Coord c, coordOffset c i = off := by
  obtain ⟨idx,hw⟩ := hw
  obtain ⟨_,lane,hlane,he⟩ := (programWrites_iff c r A s idx _).mp hw
  have he' := Nat.add_left_cancel he
  rw [WideHost.write_eq hm] at he'
  refine ⟨WideLogical.outCoord hm.shape (c.launch idx).batch (c.launch idx).head
    (queryRow c.layout (c.launch idx) lane.1.val) lane.2.1.val
    (launch_batch_lt c idx) (launch_head_lt c idx)
    ((WideHost.active_iff hm idx lane).mp hlane) lane.2.1.isLt,?_⟩
  exact he'

/-- The direct observer equals coordinate writeback as a derived memory
identity. The hypotheses concern initial placement and the source's local frame. -/
theorem observe_eq_writeback {α : Type} {c : Call} {r : Regions} {A : FlatAlloc}
    (address : Coord c → Nat) (initial before raw : Nat → α)
    (placement : ∀ i, address i = A.addr r.out (coordOffset c i))
    (bounded : ∀ i, coordOffset c i < A.extent r.out)
    (initialized : ∀ off, off < A.extent r.out → initial off = before (A.addr r.out off))
    (frame : ∀ off, (¬ ∃ i : Coord c, coordOffset c i = off) → raw off = initial off) :
    observe r A before raw = fun addr =>
      if h : ∃ i : Coord c, address i = addr then raw (coordOffset c (Classical.choose h))
      else before addr := by
  funext addr
  split
  next h =>
    have he := Classical.choose_spec h
    calc
      observe r A before raw addr = observe r A before raw (address (Classical.choose h)) :=
        congrArg (observe r A before raw) he.symm
      _ = raw (coordOffset c (Classical.choose h)) := by
        rw [placement]
        exact observe_at r A before raw _ (bounded _)
  next h =>
    unfold observe
    split
    next inside =>
      have hb : addr - A.base r.out < A.extent r.out := by omega
      have ha : A.addr r.out (addr - A.base r.out) = addr := by
        simp only [FlatAlloc.addr];omega
      have hn : ¬ ∃ i : Coord c, coordOffset c i = addr - A.base r.out := by
        rintro ⟨i,hi⟩
        exact h ⟨i,by rw [placement,hi,ha]⟩
      rw [frame _ hn,initialized _ hb,ha]
    next => rfl

#print axioms observe_eq_writeback
#print axioms write_has_coordinate
end
end VeriTile.Bench.Repairs.ContextAttnNopad.DirectOutput
