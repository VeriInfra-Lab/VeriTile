import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.IO
import bench.repairs.tritonbench_g.token_attn_reduceV_grid_y_checked.Proofs.AccessSafety
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TokenAttnReduceV
noncomputable section
variable {A : Type} [Inhabited A]
set_option maxHeartbeats 3000000

theorem input_reads (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (q : Fin (inputs p B H counts).length) : Grid.Reads p (inputs p B H counts)[q].buf := by
  fin_cases q <;> simp [inputs,Grid.Reads]

structure CallInputs (p : Parameters) (B H : Nat) (counts : Fin B → Nat)
    (v : Inputs A p B H counts) (s : State A) (b : Fin B) (h : Fin H) : Prop where
  pid0 : s.pids 0=b.val
  pid1 : s.pids 1=h.val
  length : count p s=counts b
  start : startIndex p s=start v b
  request : requestIndex p s=request v b
  tokens : ∀ n (hn : n<counts b), p.block≠0 →
    (s.mem (Region.cast p.tokens) (gatherOffset p s n)).read .nat=
      token v b ⟨n,lt_of_lt_of_le hn (count_le_capacity counts b)⟩
  probability : ∀ n, n<counts b → p.block≠0 →
    (s.mem p.prob (probOffset p s n)).read .fp32=probAt v b h n
  value : ∀ n, n<counts b → p.block≠0 → ∀ d : Fin p.width,
    (s.mem p.value (valueOffset p s d.val+
      (s.mem (Region.cast p.tokens) (gatherOffset p s n)).read .nat*p.valueBatchStride)).read .fp32=
      valAt v b h n d

theorem call_inputs (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (v : Inputs A p B H counts) (domain : LengthsMatch p B H counts v)
    (s : State A) (hin : (programIO M R p B H counts jobs).InputBinding v s)
    (b : Fin B) (h : Fin H) (h0 : s.pids 0=b.val) (h1 : s.pids 1=h.val) :
    CallInputs p B H counts v s b h := by
  have live : 0<H := by have := h.isLt; omega
  have hl := hin ⟨0,by simp [programIO,memory,inputs]⟩ (b,PUnit.unit) live
  have hs := hin ⟨1,by simp [programIO,memory,inputs]⟩ (b,PUnit.unit) live
  have hr := hin ⟨2,by simp [programIO,memory,inputs]⟩ (b,PUnit.unit) live
  have hl' : count p s=length v b := by simpa only [count,h0] using hl
  have hs' : startIndex p s=start v b := by simpa only [startIndex,h0] using hs
  have hr' : requestIndex p s=request v b := by simpa only [requestIndex,h0] using hr
  have ht : ∀ n (hn : n<counts b), p.block≠0 →
      (s.mem (Region.cast p.tokens) (gatherOffset p s n)).read .nat=
        token v b ⟨n,lt_of_lt_of_le hn (count_le_capacity counts b)⟩ := by
    intro n hn nonzero
    have t := hin ⟨3,by simp [programIO,memory,inputs]⟩
      (b,⟨n,lt_of_lt_of_le hn (count_le_capacity counts b)⟩,PUnit.unit) ⟨live,nonzero,hn⟩
    simpa only [gatherOffset,hr'] using t
  refine ⟨h0,h1,?_,hs',hr',ht,?_,?_⟩
  · exact hl'.trans (domain b live)
  · intro n hn nonzero
    have cap := lt_of_lt_of_le hn (count_le_capacity counts b)
    have x := hin ⟨4,by simp [programIO,memory,inputs]⟩ (b,h,⟨n,cap⟩,PUnit.unit) ⟨nonzero,hn⟩
    simpa only [probOffset,h1,hs',probAt,dif_pos cap] using x
  · intro n hn nonzero d
    have cap := lt_of_lt_of_le hn (count_le_capacity counts b)
    have x := hin ⟨5,by simp [programIO,memory,inputs]⟩ (b,h,⟨n,cap⟩,d,PUnit.unit) ⟨nonzero,hn⟩
    simpa only [valueOffset,h1,ht n hn nonzero,valAt,dif_pos cap] using x

theorem call_bounds (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (v : Inputs A p B H counts) (bounds : RegionName → Nat)
    (hb : (programIO M R p B H counts jobs).InBounds bounds v)
    (s : State A) (b : Fin B) (h : Fin H) (binding : CallInputs p B H counts v s b h) :
    SourceSafety.Bounds p s bounds := by
  have live : 0<H := by have := h.isLt; omega
  refine ⟨?_,?_,?_,?_,?_,?_,?_⟩
  · simpa only [binding.pid0] using
      hb.2.1 ⟨0,by simp [programIO,memory,inputs]⟩ (b,PUnit.unit) live
  · simpa only [binding.pid0] using
      hb.2.1 ⟨1,by simp [programIO,memory,inputs]⟩ (b,PUnit.unit) live
  · simpa only [binding.pid0] using
      hb.2.1 ⟨2,by simp [programIO,memory,inputs]⟩ (b,PUnit.unit) live
  · intro nonzero n hn
    have hi : n<counts b := by simpa only [binding.length] using hn
    have cap := lt_of_lt_of_le hi (count_le_capacity counts b)
    simpa only [probOffset,binding.pid1,binding.start] using
      hb.2.1 ⟨4,by simp [programIO,memory,inputs]⟩ (b,h,⟨n,cap⟩,PUnit.unit) ⟨nonzero,hi⟩
  · intro nonzero n hn
    have hi : n<counts b := by simpa only [binding.length] using hn
    have cap := lt_of_lt_of_le hi (count_le_capacity counts b)
    simpa only [gatherOffset,binding.request] using
      hb.2.1 ⟨3,by simp [programIO,memory,inputs]⟩ (b,⟨n,cap⟩,PUnit.unit) ⟨live,nonzero,hi⟩
  · intro nonzero n hn d
    have hi : n<counts b := by simpa only [binding.length] using hn
    have cap := lt_of_lt_of_le hi (count_le_capacity counts b)
    simpa only [valueOffset,binding.pid1,binding.tokens n hi nonzero] using
      hb.2.1 ⟨5,by simp [programIO,memory,inputs]⟩ (b,h,⟨n,cap⟩,d,PUnit.unit) ⟨nonzero,hi⟩
  · intro d
    simpa only [outputOffset,binding.pid0,binding.pid1,Grid.offset] using
      hb.2.2.1 ⟨0,by simp [programIO,memory,outputs]⟩ (b,h,d,PUnit.unit) trivial

theorem bound_after_frame (M : Algebra A) (R : SumPrimitive A) (p : Parameters)
    (B H : Nat) (counts : Fin B → Nat) (jobs : List Grid.PID)
    (v : Inputs A p B H counts) (s t : State A)
    (hs : (programIO M R p B H counts jobs).InputBinding v s)
    (frame : ∀ r, Grid.Reads p r → t.mem r=s.mem r) :
    (programIO M R p B H counts jobs).InputBinding v t := by
  intro q j active
  have hr := input_reads p B H counts q
  have eq := frame (inputs p B H counts)[q].buf hr
  change (t.mem (inputs p B H counts)[q].buf _).read _ = _
  rw [eq]
  exact hs q j active

#axiomsClean call_inputs
#axiomsClean call_bounds
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked.Interface
