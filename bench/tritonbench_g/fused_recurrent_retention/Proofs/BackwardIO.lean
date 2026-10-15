import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardReal
import bench.tritonbench_g.fused_recurrent_retention.Proofs.BackwardSafety
import VeriTile.Triton.Memory.FlattenFrame

/-! Finite-memory postcondition and tagged-cell transport for the independent
backward formulas. Actual flattened execution remains to be connected: the
last pointer subtractions can clip, so the global FlattenOk bridge is inapplicable. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardIO
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

def Written (p : Layout) (l : Launch) (r : BackwardRegions)
    (region : RegionName) (off : Nat) : Prop :=
  (∃ (t : Nat) (k : Fin p.BK), t < p.T ∧ activeKey p l k.val ∧
    (r.dq,keyGradientAddress p l t k.val) = (region,off)) ∨
  (∃ (t : Nat) (k : Fin p.BK), t < p.T ∧ activeKey p l k.val ∧
    (r.dk,keyGradientAddress p l t k.val) = (region,off)) ∨
  (∃ (t : Nat) (j : Fin p.BV), t < p.T ∧ activeValue p l j.val ∧
    (r.dv,outputAddress p l t j.val) = (region,off))

def Outside (A : FlatAlloc) (p : Layout) (l : Launch) (r : BackwardRegions)
    (region : RegionName) (off : Nat) : Prop :=
  region ≠ A.flat ∨ ∀ q z, Written p l r q z → off ≠ A.addr q z

structure Post (A : FlatAlloc) (p : Layout) (r : BackwardRegions) (hasSeed : Bool)
    (initial final : BlockState) : Prop where
  dq : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p (launch initial) k.val →
    final.mem A.flat (A.addr r.dq (keyGradientAddress p (launch initial) t k.val)) =
      MemCell.real (MaskedMath.dq p (launch initial) initial r hasSeed t k)
  dk : ∀ (t : Nat) (k : Fin p.BK), t < p.T → activeKey p (launch initial) k.val →
    final.mem A.flat (A.addr r.dk (keyGradientAddress p (launch initial) t k.val)) =
      MemCell.real (MaskedMath.dk p (launch initial) initial r t k)
  dv : ∀ (t : Nat) (j : Fin p.BV), t < p.T → activeValue p (launch initial) j.val →
    final.mem A.flat (A.addr r.dv (outputAddress p (launch initial) t j.val)) =
      MemCell.real (MaskedMath.dv p (launch initial) initial r t j)
  frame : ∀ region off, Outside A p (launch initial) r region off →
    final.mem region off = (A.flattenState initial).mem region off

theorem flatten_post (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ region, region ∉ A.regions → A.extent region = 0)
    (p : Layout) (r : BackwardRegions) (hasSeed : Bool) (initial final : BlockState)
    (hb : BackwardSafety.Bounds p (launch initial) r hasSeed A.extent)
    (h : BackwardReal.Post p r hasSeed initial final) :
    Post A p r hasSeed initial (A.flattenState final) := by
  constructor
  · intro t k ht hk
    have ho := hb.dq t k ht hk
    have hr : r.dq ∈ A.regions := by
      by_contra hn
      rw [hc r.dq hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho,h.dq t k ht hk]
    rfl
  · intro t k ht hk
    have ho := hb.dk t k ht hk
    have hr : r.dk ∈ A.regions := by
      by_contra hn
      rw [hc r.dk hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho,h.dk t k ht hk]
    rfl
  · intro t j ht hj
    have ho := hb.dv t j ht hj
    have hr : r.dv ∈ A.regions := by
      by_contra hn
      rw [hc r.dv hn] at ho
      omega
    rw [A.flattenState_mem_addr hd final hr ho,h.dv t j ht hj]
    rfl
  · intro region off hn
    apply A.flattenState_frame initial final (Written p (launch initial) r) _ region off hn
    intro q z hmiss
    apply h.frame q z
    · intro t k ht hk heq
      exact hmiss (Or.inl ⟨t,k,ht,hk,heq⟩)
    · intro t k ht hk heq
      exact hmiss (Or.inr (Or.inl ⟨t,k,ht,hk,heq⟩))
    · intro t j ht hj heq
      exact hmiss (Or.inr (Or.inr ⟨t,j,ht,hj,heq⟩))

#axiomsClean flatten_post
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.BackwardIO
