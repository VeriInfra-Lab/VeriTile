import bench.tritonbench_g.chunk_retention_ops.Sources.Kernels
import bench.tritonbench_g.chunk_retention_ops.Sources.Geometry
import bench.tritonbench_g.chunk_retention.Sources.ForwardOComplete
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
namespace Shared
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelModel (DotPrimitive withDot)
end Shared
namespace O
export VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
  (written value active outputAddress Index)
end O

def shape (c : Shape) : VeriTile.Bench.Repairs.ChunkRetention.Whole001.Shape :=
  ⟨c.B,c.H,c.T,c.K,c.V,c.BK,c.BV,c.keyPositive,c.valuePositive,c.keyTile,c.valueTile⟩

def kernel (c : Shape) (scale : ℝ) : ComputeKernel :=
  Source.forwardO "q" "k" "v" "h" "o"
    (c.T*c.K) c.K 1 (c.T*c.V) c.V 1 (c.chunks*c.K*c.V) c.V
    scale c.H c.T c.K c.V 64 c.BK c.BV

/-- The selected source bodies are definitionally identical. This equality reuses
an execution proof; it is not a compiler-graph identity assumption. -/
theorem source_identity (c : Shape) (scale : ℝ) : kernel c scale=
    VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.kernel (shape c) scale := rfl

variable {α : Type} [Inhabited α]
/-- The complete selected output quotation executes, with its independent output
value and scatter memory. The total dot argument is a local primitive model. -/
theorem program (M : Algebra α) (D : Shared.DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec (engine (Shared.withDot M D)) (kernel c scale) s=some t ∧
      t.mem=(O.written M D (shape c) scale s).mem := by
  rw [source_identity]
  exact VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.source_memory M D (shape c) scale s

theorem readback (M : Algebra α) (D : Shared.DotPrimitive α) (c : Shape) (scale : ℝ)
    (s : State α) (i : O.Index (shape c)) (hi : O.active (shape c) s i)
    (hw : ∀ i j,O.active (shape c) s i → O.active (shape c) s j →
      O.outputAddress (shape c) s i=O.outputAddress (shape c) s j → i=j) :
    (O.written M D (shape c) scale s).mem "o" (O.outputAddress (shape c) s i)=
      .mk .real (O.value M D (shape c) scale s i) :=
  VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.readback M D (shape c) scale s i hi hw

theorem frame (M : Algebra α) (D : Shared.DotPrimitive α) (c : Shape) (scale : ℝ)
    (s : State α) (r : RegionName) (off : Nat)
    (h : r≠"o" ∨ ∀ i,O.active (shape c) s i → off≠O.outputAddress (shape c) s i) :
    (O.written M D (shape c) scale s).mem r off=s.mem r off :=
  VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource.frame M D (shape c) scale s r off h

#axiomsClean source_identity
#axiomsClean program
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOutput
