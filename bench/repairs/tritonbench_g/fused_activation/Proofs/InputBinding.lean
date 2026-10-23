import bench.repairs.tritonbench_g.fused_activation.Proofs.Contract
import VeriTile.Torch.FP32
import VeriTile.Triton.Semantics.State

/-! Input metadata has actual Tensor/storage meaning. This binding does not
execute allocation or a wrapper. Bias and input may share read-only storage. -/
namespace VeriTile.Bench.Repairs.FusedActivation.InputBinding
open VeriTile.Triton
open VeriTile.Torch.FP32

/-- All three raw contiguous reads equal the corresponding logical elements.
The caller may bind the two read-only regions to overlapping physical views. -/
theorem logical_reads (read : Nat → Word) (x b i : Tensor)
    (lx : TensorLaws x) (lb : TensorLaws b) (li : TensorLaws i)
    (h : InputContract x.metadata b.metadata i.metadata) :
    (∀ j, j < x.metadata.numel →
      read ((dataView x lx h.2.2.2.2.1.1).address j) = logicalValue read x lx j) ∧
    (∀ j, j < b.metadata.numel →
      read ((dataView b lb h.2.2.2.2.1.2.1).address j) = logicalValue read b lb j) ∧
    (∀ j, j < x.metadata.numel →
      read ((dataView i li h.2.2.2.2.1.2.2).address j) = logicalValue read i li j) := by
  rcases h with ⟨_, _, _, _, ⟨xc,bc,ic⟩, ⟨xn,bn,ineg⟩, extent, _⟩
  exact ⟨fun j hj => direct_read_eq_logical read x lx xc xn j hj,
    fun j hj => direct_read_eq_logical read b lb bc bn j hj,
    fun j hj => direct_read_eq_logical read i li ic ineg j (lt_of_lt_of_le hj extent)⟩

/-- Entry observations of the three checked views through one physical reader.
The two read-only views may overlap, including different storage offsets.
`decode` supplies their mathematical interpretation; this snapshot neither
executes a wrapper nor asserts an exact native floating-point result. -/
noncomputable def bindReadViews (read : Nat → Word) (decode : Word → ℝ)
    (x b i : Tensor) (lx : TensorLaws x) (lb : TensorLaws b) (li : TensorLaws i)
    (h : InputContract x.metadata b.metadata i.metadata) (scratch : BlockState) : BlockState :=
  { scratch with mem := fun r o =>
      if r = "x" then MemCell.real (decode (read ((dataView x lx h.2.2.2.2.1.1).address o)))
      else if r = "bias" then
        MemCell.real (decode (read ((dataView b lb h.2.2.2.2.1.2.1).address o)))
      else if r = "input" then
        MemCell.real (decode (read ((dataView i li h.2.2.2.2.1.2.2).address o)))
      else scratch.mem r o }

/-- All kernel entry reads of the snapshot are the checked logical Tensor
values. There is no read-only separation premise or output premise. -/
theorem bindReadViews_logical (read : Nat → Word) (decode : Word → ℝ)
    (x b i : Tensor) (lx : TensorLaws x) (lb : TensorLaws b) (li : TensorLaws i)
    (h : InputContract x.metadata b.metadata i.metadata) (scratch : BlockState) :
    let s := bindReadViews read decode x b i lx lb li h scratch
    (∀ j, j < x.metadata.numel → s.readMem "x" j = decode (logicalValue read x lx j)) ∧
    (∀ j, j < b.metadata.numel → s.readMem "bias" j = decode (logicalValue read b lb j)) ∧
    (∀ j, j < x.metadata.numel → s.readMem "input" j = decode (logicalValue read i li j)) := by
  obtain ⟨hx, hb, hi⟩ := logical_reads read x b i lx lb li h
  refine ⟨fun j hj => ?_, fun j hj => ?_, fun j hj => ?_⟩
  · exact congrArg decode (hx j hj)
  · exact congrArg decode (hb j hj)
  · exact congrArg decode (hi j hj)

/-- The three views passed to nonempty kernel launches use the pointers
observed by Python; storage offsets are already included in those pointers. -/
theorem actual_pointers (x b i : Tensor)
    (lx : TensorLaws x) (lb : TensorLaws b) (li : TensorLaws i)
    (h : InputContract x.metadata b.metadata i.metadata) (hn : 0 < x.metadata.numel) :
    BitVec.ofNat 64 ((dataView x lx h.2.2.2.2.1.1).address 0) = x.metadata.dataPtr ∧
    BitVec.ofNat 64 ((dataView b lb h.2.2.2.2.1.2.1).address 0) = b.metadata.dataPtr ∧
    BitVec.ofNat 64 ((dataView i li h.2.2.2.2.1.2.2).address 0) = i.metadata.dataPtr := by
  rcases h with ⟨_, _, _, _, ⟨xc,bc,ic⟩, _, extent, _, _, live⟩
  exact ⟨actual_pointer x lx xc hn, actual_pointer b lb bc (live hn).1,
    actual_pointer i li ic (lt_of_lt_of_le hn extent)⟩

/-- Every used view has scalar-FP32 alignment from exactly the checked pointer
conditions. An empty x imposes no unnecessary alignment on unused operands. -/
theorem access_alignment (x b i : Tensor)
    (lx : TensorLaws x) (lb : TensorLaws b) (li : TensorLaws i)
    (h : InputContract x.metadata b.metadata i.metadata) (hn : 0 < x.metadata.numel) :
    (dataView x lx h.2.2.2.2.1.1).base % 4 = 0 ∧
    (dataView b lb h.2.2.2.2.1.2.1).base % 4 = 0 ∧
    (dataView i li h.2.2.2.2.1.2.2).base % 4 = 0 := by
  rcases h with ⟨_, _, _, _, ⟨xc,bc,ic⟩, _, _, _, _, live⟩
  obtain ⟨_, ⟨xa,ba,ia⟩, _⟩ := live hn
  exact ⟨dataView_aligned x lx xc (fun _ => xa),
    dataView_aligned b lb bc (fun _ => ba), dataView_aligned i li ic (fun _ => ia)⟩

#axiomsClean logical_reads
#axiomsClean bindReadViews_logical
#axiomsClean actual_pointers
#axiomsClean access_alignment
end VeriTile.Bench.Repairs.FusedActivation.InputBinding
