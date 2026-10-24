import bench.repairs.tritonbench_g.chunked_cumsum_fwd.Proofs.FlatStorage

/-! Nonempty same-device witnesses for the new ordinary allocator interface.
One caller allocation (with a partial trailing word), two fresh outputs and
two aliased input descriptors inhabit the interface for arbitrary word values. -/
namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.AllocatorWitness
open VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd RawBoundary Storage
set_option autoImplicit false

def allocation (n : Nat) : Allocation where
  address := 4096 + 4096*(n%3)
  bytes := 7
  device := 0
  endBound := by have := Nat.mod_lt n (by decide : 0 < 3); omega
  aligned := by omega

def live (n : Nat) : Prop := n < 3

theorem separated : Separated allocation live := by
  intro a b i j ha hb hi hj _ eq
  simp only [allocation] at hi hj eq
  have aa : a%3 = a := Nat.mod_eq_of_lt ha
  have bb : b%3 = b := Nat.mod_eq_of_lt hb
  rw [aa,bb] at eq
  omega

def tensor (owner object : Nat) : Storage.Tensor allocation where
  object := object
  shape := [1]
  stride := fun _ => 1
  dtype := .fp32
  fp32 := rfl
  view := ⟨owner,0⟩
  dataPtr := (allocation owner).address
  device := 0
  sameDevice := rfl
  pointer := by intro off _; simp

theorem aliased_inputs : (tensor 0 10).view = (tensor 0 11).view ∧
    (tensor 0 10).dataPtr = (tensor 0 11).dataPtr ∧
    (tensor 0 10).object ≠ (tensor 0 11).object := by decide

theorem fresh_outputs : (tensor 1 20).view.allocation ≠ (tensor 2 21).view.allocation ∧
    (tensor 1 20).dataPtr ≠ (tensor 0 10).dataPtr ∧
    (tensor 2 21).dataPtr ≠ (tensor 0 11).dataPtr ∧
    (tensor 1 20).dataPtr ≠ (tensor 2 21).dataPtr := by decide

theorem all_words_representable (memory : PhysicalMemory Word) :
    ∃ flat, Represents allocation live memory flat := representation_exists allocation live separated memory

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.AllocatorWitness
