import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.Contract
import VeriTile.Torch.FP32

/-! Kernel entry binding for the source's existing contiguous materializations.
Copy preservation is an ordinary Torch API boundary, never a kernel-result
premise. Original input and weight views may overlap or be noncontiguous. -/
namespace VeriTile.Bench.Repairs.RmsNormTriton.InputBinding
open VeriTile.Triton FP.Structural VeriTile.Torch.FP32
open VeriTile.Bench.TritonBenchG.RmsNormTriton

def info (t : Tensor) (conjugated : Bool) : TensorInfo := ⟨t.metadata,conjugated⟩

/-- A source-required contiguous view/copy, observed before kernel entry.
The result may share storage with the original; no freshness is required. -/
structure Materialized (entry ready : Nat → Word) (original : Tensor) (laws : TensorLaws original) where
  tensor : Tensor
  binding : TensorLaws tensor
  shape : tensor.metadata.shape = original.metadata.shape
  device : tensor.metadata.device = original.metadata.device
  contiguous : tensor.metadata.isContiguous = Bool.true
  noNeg : tensor.metadata.isNeg = Bool.false
  values : ∀ j, j < original.metadata.numel →
    logicalValue ready tensor binding j = logicalValue entry original laws j

theorem Materialized.read_eq {entry ready : Nat → Word} {original : Tensor} {laws : TensorLaws original}
    (copy : Materialized entry ready original laws) (j : Nat) (hj : j < original.metadata.numel) :
    ready ((dataView copy.tensor copy.binding copy.contiguous).address j) =
      logicalValue entry original laws j := by
  have bound : j < copy.tensor.metadata.numel := by
    simpa only [VeriTile.Torch.TensorMetadata.numel, copy.shape] using hj
  exact (direct_read_eq_logical ready copy.tensor copy.binding copy.contiguous copy.noNeg j bound).trans
    (copy.values j hj)

/-- The copied view includes its actual storage offset and launch pointer. -/
theorem Materialized.actual_pointer {entry ready : Nat → Word} {original : Tensor} {laws : TensorLaws original}
    (copy : Materialized entry ready original laws) (hn : 0 < original.metadata.numel) :
    BitVec.ofNat 64 ((dataView copy.tensor copy.binding copy.contiguous).address 0) = copy.tensor.dataPtr := by
  apply VeriTile.Torch.FP32.actual_pointer
  simpa only [VeriTile.Torch.TensorMetadata.numel, copy.shape] using hn

def layout (x : TensorInfo) (n : List Nat) : Whole.Layout :=
  ⟨"X","W","Y",rows x n,columns n,block (columns n),
    by rw [block_eq]; exact Whole.nextPower_covers _⟩

theorem shape_product (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (h : InputContract x w n eps) :
    x.numel = (layout x n.dimensions).M * (layout x n.dimensions).N := by
  have split := congrArg List.prod (List.take_append_drop (dim x n.dimensions) x.shape)
  simpa only [List.prod_append,h.suffix,VeriTile.Torch.TensorMetadata.numel,
    layout,rows,columns] using split.symm

theorem weight_size (x w : TensorInfo) (n : NormalizedInfo) (eps : EpsilonInfo)
    (h : InputContract x w n eps) : w.numel = (layout x n.dimensions).N := by
  simp only [VeriTile.Torch.TensorMetadata.numel,h.weightShape,layout,columns]

/-- Bind the two actual materialized views through the same physical reader.
Only entry reads are constructed; output and other scratch cells stay arbitrary. -/
def bindReadViews {α : Type} [Inhabited α] (entry ready : Nat → Word) (decode : Word → α)
    (x w : Tensor) (lx : TensorLaws x) (lw : TensorLaws w)
    (xc : Materialized entry ready x lx) (wc : Materialized entry ready w lw)
    (scratch : State α) : State α :=
  { scratch with mem := fun r j =>
      if r = "X" then .mk .real (decode (ready ((dataView xc.tensor xc.binding xc.contiguous).address j)))
      else if r = "W" then .mk .real (decode (ready ((dataView wc.tensor wc.binding wc.contiguous).address j)))
      else scratch.mem r j }

theorem bindReadViews_logical {α : Type} [Inhabited α] (entry ready : Nat → Word) (decode : Word → α)
    (x w : Tensor) (lx : TensorLaws x) (lw : TensorLaws w)
    (xc : Materialized entry ready x lx) (wc : Materialized entry ready w lw)
    (scratch : State α) :
    let s := bindReadViews entry ready decode x w lx lw xc wc scratch
    (∀ j, j < x.metadata.numel → (s.mem "X" j).read .real = decode (logicalValue entry x lx j)) ∧
    (∀ j, j < w.metadata.numel → (s.mem "W" j).read .real = decode (logicalValue entry w lw j)) := by
  exact ⟨fun j hj => congrArg decode (xc.read_eq j hj),
    fun j hj => congrArg decode (wc.read_eq j hj)⟩

#axiomsClean Materialized.read_eq
#axiomsClean Materialized.actual_pointer
#axiomsClean shape_product
#axiomsClean weight_size
#axiomsClean bindReadViews_logical
end VeriTile.Bench.Repairs.RmsNormTriton.InputBinding
