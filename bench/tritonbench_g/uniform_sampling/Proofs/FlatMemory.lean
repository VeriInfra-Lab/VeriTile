import bench.tritonbench_g.uniform_sampling.Proofs.Memory
import VeriTile.Triton.Memory.Flatten

/-! Finite allocation and full tagged-cell frame for the independently
specified four-store writeback. A separate execution theorem connects this
memory operation to the source. No successful execution is assumed here.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.FlatMemory
open VeriTile.Triton
noncomputable section

structure Bounds (y : RegionName) (N B pid : Nat) (bounds : RegionBounds) : Prop where
  output : ∀ f : Fin 4, ∀ j : Fin B, SourceDomain.outputIndex pid B f j < N →
    SourceDomain.outputIndex pid B f j < bounds y

theorem bounds_of_extent (y : RegionName) (N B pid : Nat)
    (bounds : RegionBounds) (hy : N ≤ bounds y) : Bounds y N B pid bounds :=
  ⟨fun _ _ hj => lt_of_lt_of_le hj hy⟩

def Outside (A : FlatAlloc) (pid : Nat) (y : RegionName)
    (N B : Nat) (r : RegionName) (o : Nat) : Prop :=
  r ≠ A.flat ∨ ∀ f : Fin 4, ∀ j : Fin B, SourceDomain.outputIndex pid B f j < N →
    o ≠ A.addr y (SourceDomain.outputIndex pid B f j)

theorem storeBlock_active_cell (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (lower upper : ℝ)
    (hb : Bounds y N B pid A.extent) (f : Fin 4) (j : Fin B)
    (hj : SourceDomain.outputIndex pid B f j < N) :
    (A.flattenState (Memory.storeBlock pid base y N B seed offset lower upper)).mem A.flat
      (A.addr y (SourceDomain.outputIndex pid B f j)) =
        MemCell.real (Formula.expected B seed offset lower upper (SourceDomain.outputIndex pid B f j)) := by
  have hy := hb.output f j hj
  have hr : y ∈ A.regions := by
    by_contra hn
    rw [hc y hn] at hy
    omega
  rw [A.flattenState_mem_addr hd _ hr hy,
    Memory.storeBlock_active_cell pid base y N B seed offset lower upper f j hj]
  rfl

theorem storeBlock_frame (A : FlatAlloc) (pid : Nat) (base : BlockState)
    (y : RegionName) (N B : Nat) (seed offset : Int) (lower upper : ℝ)
    (r : RegionName) (o : Nat) (ho : Outside A pid y N B r o) :
    (A.flattenState (Memory.storeBlock pid base y N B seed offset lower upper)).mem r o =
      (A.flattenState base).mem r o := by
  by_cases hr : r = A.flat
  · subst r
    simp only [FlatAlloc.flattenState]
    unfold FlatAlloc.readFlat
    cases hc : A.decode o with
    | none => rfl
    | some pair =>
      obtain ⟨q, z⟩ := pair
      obtain ⟨_, hz, _⟩ := A.decode_sound hc
      apply congrArg A.trCell
      apply Memory.storeBlock_frame
      intro f j hj ⟨hq, hoz⟩
      subst q
      rcases ho with hn | hn
      · exact hn rfl
      · exact hn f j hj (by rw [hoz]; exact hz)
  · simp only [FlatAlloc.flattenState, if_neg hr]

theorem storeBlock_readback (A : FlatAlloc) (hd : A.Disjoint)
    (hc : ∀ r, r ∉ A.regions → A.extent r = 0)
    (pid : Nat) (base : BlockState) (y : RegionName) (N B : Nat)
    (seed offset : Int) (lower upper : ℝ)
    (hb : Bounds y N B pid A.extent) (f : Fin 4) (j : Fin B)
    (hj : SourceDomain.outputIndex pid B f j < N) :
    (A.flattenState (Memory.storeBlock pid base y N B seed offset lower upper)).readMem A.flat
      (A.addr y (SourceDomain.outputIndex pid B f j)) =
        Formula.expected B seed offset lower upper (SourceDomain.outputIndex pid B f j) := by
  have h := storeBlock_active_cell A hd hc pid base y N B seed offset lower upper hb f j hj
  simp only [BlockState.readMem, h]
  rfl

#axiomsClean bounds_of_extent
#axiomsClean storeBlock_active_cell
#axiomsClean storeBlock_frame
#axiomsClean storeBlock_readback

end
end VeriTile.Bench.TritonBenchG.UniformSampling.FlatMemory
