import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardReal
import bench.tritonbench_g.fused_recurrent_retention.Proofs.ForwardSafety
import VeriTile.Triton.Memory.FlattenFrame

/-! Complete forward execution in a finite flat allocation, with independent
initial-snapshot output/final-state formulas and a full tagged-cell frame.
All masked loads specify other=0, so the internal oracle normalization required
by the generic bridge imposes no oracle premise on the initial state. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def Written (p : Layout) (l : Launch) (r : ForwardRegions) (storeFinal : Bool)
    (region : RegionName) (off : Nat) : Prop :=
  (∃ (t : Nat) (j : Fin p.BV), t < p.T ∧ activeValue p l j.val ∧
    (r.o,outputAddress p l t j.val) = (region,off)) ∨
  (storeFinal = Bool.true ∧ ∃ (k : Fin p.BK) (j : Fin p.BV),
    activeKey p l k.val ∧ activeValue p l j.val ∧
    (r.final,stateAddress p l k.val j.val) = (region,off))

def Outside (A : FlatAlloc) (p : Layout) (l : Launch) (r : ForwardRegions)
    (storeFinal : Bool) (region : RegionName) (off : Nat) : Prop :=
  region ≠ A.flat ∨ ∀ q z, Written p l r storeFinal q z → off ≠ A.addr q z

structure Post (A : FlatAlloc) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (initial final : BlockState) : Prop where
  output : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p (launch initial) j.val →
    final.mem A.flat (A.addr r.o (outputAddress p (launch initial) t j.val)) =
      MemCell.real (MaskedMath.output p (launch initial) initial r hasSeed t j)
  finalState : storeFinal = Bool.true → ∀ (k : Fin p.BK) (j : Fin p.BV),
    activeKey p (launch initial) k.val → activeValue p (launch initial) j.val →
    final.mem A.flat (A.addr r.final (stateAddress p (launch initial) k.val j.val)) =
      MemCell.real (MaskedMath.stateValue p (launch initial) initial r.k r.v r.initial hasSeed p.T k j)
  frame : ∀ region off, Outside A p (launch initial) r storeFinal region off →
    final.mem region off = (A.flattenState initial).mem region off

theorem flatten_post (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ region, region ∉ A.regions → A.extent region = 0)
    (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool) (initial final : BlockState)
    (hb : SafetyAccess.Bounds p (launch initial) r hasSeed storeFinal A.extent)
    (h : ForwardReal.Post p r hasSeed storeFinal initial final) :
    Post A p r hasSeed storeFinal initial (A.flattenState final) := by
  constructor
  · intro t j ht hj
    have ho := hb.output t j ht hj
    have hr : r.o ∈ A.regions := by
      by_contra hn
      rw [hc r.o hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho,h.output t j ht hj]
    rfl
  · intro hf k j hk hj
    have ho := hb.finalState hf k j hk hj
    have hr : r.final ∈ A.regions := by
      by_contra hn
      rw [hc r.final hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho,h.finalState hf k j hk hj]
    rfl
  · intro region off hn
    apply A.flattenState_frame initial final (Written p (launch initial) r storeFinal) _ region off hn
    intro q z hmiss
    apply h.frame q z
    · intro t j ht hj heq
      exact hmiss (Or.inl ⟨t,j,ht,hj,heq⟩)
    · intro hf k j hk hj heq
      exact hmiss (Or.inr ⟨hf,k,j,hk,hj,heq⟩)

theorem source_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ region, region ∉ A.regions → A.extent region = 0)
    (p : Layout) (r : ForwardRegions) (hasSeed storeFinal swapped : Bool) (initial : BlockState)
    (hqo : r.q ≠ r.o) (hko : r.k ≠ r.o) (hvo : r.v ≠ r.o)
    (hfinal : storeFinal = Bool.true → r.o ≠ r.final)
    (hb : SafetyAccess.Bounds p (launch initial) r hasSeed storeFinal A.extent) :
    ∃ final, exec (A.flattenKernel (ForwardSource.kernel p r hasSeed storeFinal swapped).toAlgKernel)
      (A.flattenState initial) = some final ∧ Post A p r hasSeed storeFinal initial final := by
  let initial0 : BlockState := {initial with undef := fun _ _ => 0}
  obtain ⟨final,he,hp⟩ := ForwardReal.source_correct p r hasSeed storeFinal swapped initial0
    hqo hko hvo hfinal
  have hs := ForwardSafety.source_safe p r hasSeed storeFinal swapped initial0 hqo hko hvo A.extent hb
  have hf := A.exec_flatten hd hc _ initial0 hs
    (ForwardSafety.source_flatten_ok p r hasSeed storeFinal swapped) rfl
  rw [he,Option.map_some] at hf
  refine ⟨A.flattenState final,hf,?_⟩
  have hpost := flatten_post A hd hc p r hasSeed storeFinal initial0 final hb hp
  exact ⟨hpost.output,hpost.finalState,hpost.frame⟩

#axiomsClean flatten_post
#axiomsClean source_correct
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.ForwardIO
