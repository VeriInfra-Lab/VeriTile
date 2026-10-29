import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Contract
import bench.repairs.tritonbench_g.adam_update_triton.Kernels
import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Index
import VeriTile.Torch.FP32
namespace VeriTile.Bench.Repairs.AdamUpdateTriton.InputBinding
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32

/-- All three admitted contiguous views read their original logical elements,
including nonzero storage offsets. No allocation or wrapper run is modeled. -/
theorem logical_reads (read : Nat → Word) (p g m : Tensor)
    (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata) :
    (∀ j, j < p.metadata.numel →
      read ((dataView p lp h.contiguous.1).address j) = logicalValue read p lp j) ∧
    (∀ j, j < p.metadata.numel →
      read ((dataView g lg h.contiguous.2.1).address j) = logicalValue read g lg j) ∧
    (∀ j, j < p.metadata.numel →
      read ((dataView m lm h.contiguous.2.2).address j) = logicalValue read m lm j) :=
  ⟨fun j hj => direct_read_eq_logical read p lp h.contiguous.1 h.noNeg.1 j hj,
   fun j hj => direct_read_eq_logical read g lg h.contiguous.2.1 h.noNeg.2.1 j
     (lt_of_lt_of_le hj h.enough.1),
   fun j hj => direct_read_eq_logical read m lm h.contiguous.2.2 h.noNeg.2.2 j
     (lt_of_lt_of_le hj h.enough.2)⟩

/-- Bind actual physical FP32 reads as entry observations for either the Real
or native graph interpretation. This does not assume any kernel output. -/
def bindReadViews {α : Type} [Inhabited α] (read : Nat → Word) (decode : Word → α)
    (p g m : Tensor) (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata) (scratch : State α) : State α :=
  { scratch with mem := fun r j =>
      if r = "P" then .mk .real (decode (read ((dataView p lp h.contiguous.1).address j)))
      else if r = "G" then .mk .real (decode (read ((dataView g lg h.contiguous.2.1).address j)))
      else if r = "M" then .mk .real (decode (read ((dataView m lm h.contiguous.2.2).address j)))
      else scratch.mem r j }

theorem bindReadViews_logical {α : Type} [Inhabited α] (read : Nat → Word) (decode : Word → α)
    (p g m : Tensor) (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata) (scratch : State α) :
    let s := bindReadViews read decode p g m lp lg lm h scratch
    (∀ j, j < p.metadata.numel → (s.mem "P" j).read .real = decode (logicalValue read p lp j)) ∧
    (∀ j, j < p.metadata.numel → (s.mem "G" j).read .real = decode (logicalValue read g lg j)) ∧
    (∀ j, j < p.metadata.numel → (s.mem "M" j).read .real = decode (logicalValue read m lm j)) := by
  obtain ⟨hp, hg, hm⟩ := logical_reads read p g m lp lg lm h
  exact ⟨fun j hj => congrArg decode (hp j hj),
    fun j hj => congrArg decode (hg j hj), fun j hj => congrArg decode (hm j hj)⟩

/-- Launch views retain the pointers checked by Python, including offsets. -/
theorem actual_pointers (p g m : Tensor) (lp : TensorLaws p) (lg : TensorLaws g) (lm : TensorLaws m)
    (h : InputContract p.metadata g.metadata m.metadata) (hn : 0 < p.metadata.numel) :
    BitVec.ofNat 64 ((dataView p lp h.contiguous.1).address 0) = p.dataPtr ∧
    BitVec.ofNat 64 ((dataView g lg h.contiguous.2.1).address 0) = g.dataPtr ∧
    BitVec.ofNat 64 ((dataView m lm h.contiguous.2.2).address 0) = m.dataPtr :=
  ⟨actual_pointer p lp h.contiguous.1 hn,
    actual_pointer g lg h.contiguous.2.1 (lt_of_lt_of_le hn h.enough.1),
    actual_pointer m lm h.contiguous.2.2 (lt_of_lt_of_le hn h.enough.2)⟩

/-- Reuse the existing per-chunk signed32 refinement and derive the active
native pointer directly from this shared Tensor view. No unused masked pointer
is required to be backed, and no new total-input size cap is introduced. -/
theorem native_address (t : Tensor) (laws : TensorLaws t)
    (hc : t.metadata.isContiguous = Bool.true) (n : Nat)
    (config : VeriTile.Bench.TritonBenchG.AdamUpdateTriton.HostAutotune.Config)
    (pid lane : Nat) (extent : n ≤ t.metadata.numel) (bounded : n ≤ 8388480)
    (hp : pid < Index.grid n config) (hl : lane < config.block)
    (active : pid * config.block + lane < n) :
    (Index.pointer (dataView t laws hc).base 0 pid config.block lane).toNat =
      (dataView t laws hc).address (pid * config.block + lane) := by
  rw [Index.pointer_modular _ _ n config pid lane bounded hp hl]
  have fits := (dataView t laws hc).spanFits
  change (dataView t laws hc).base + 4 * t.metadata.numel ≤ 2^64 at fits
  simp only [DataView.address, Nat.zero_add]
  apply Nat.mod_eq_of_lt
  omega

#axiomsClean logical_reads
#axiomsClean bindReadViews_logical
#axiomsClean actual_pointers
#axiomsClean native_address
end VeriTile.Bench.Repairs.AdamUpdateTriton.InputBinding
