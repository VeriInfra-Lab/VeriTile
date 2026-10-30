import bench.tritonbench_g.chunk_linear_attn.LibrarySum
import bench.tritonbench_g.chunk_linear_attn.GridReadback
import bench.tritonbench_g.chunk_delta_fwd.Sources.RealModel
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

def valueCount (p : Parameters) : Nat := p.B*p.H*p.T*p.V

/-- The ordinary library selection boundary supplies only a finite operation
tree and leaf-coverage certificate for each output. There is no answer field. -/
structure LibrarySelection (p : Parameters) where
  plans : Fin (valueCount p) → LibrarySum.Plan (keyBlocks p)
  profile : String
  device : Nat

def sumValue (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (m : Memory α) (o : Fin (valueCount p)) : α :=
  LibrarySum.evaluate M (selection.plans o)
    (fun k => (m "valuePartials" (k.val*valueCount p+o.val)).read .real)

def hostSumMemory (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (input current : Memory α) : Memory α :=
  scatter (TileShape.allIndices [valueCount p]) "valueGradient" (fun _ => True)
    (fun i => i.1.val) (fun i => .mk .real (cast M (sumValue M p selection input i.1))) current

theorem host_sum_read (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (input current : Memory α) (o : Fin (valueCount p)) :
    hostSumMemory M p selection input current "valueGradient" o.val=
      .mk .real (cast M (sumValue M p selection input o)) := by
  unfold hostSumMemory
  apply VeriTile.Bench.Repairs.ChunkDeltaFwd.scatter_readback (shape := [valueCount p])
    "valueGradient" (fun i => i.1.val) (fun i => .mk .real (cast M (sumValue M p selection input i.1))) (fun _ => True) current (o,()) trivial
  intro a b ha hb he
  exact Prod.ext (Fin.ext he) (Subsingleton.elim _ _)

theorem host_sum_frame (M : Algebra α) (p : Parameters) (selection : LibrarySelection p)
    (input current : Memory α) (r : RegionName) (o : Nat)
    (h : r≠"valueGradient" ∨ valueCount p≤o) :
    hostSumMemory M p selection input current r o=current r o := by
  apply scatter_frame
  rcases h with hr|ho
  · exact Or.inl hr
  · right
    intro i _ _ he
    have hi := i.1.isLt
    omega

theorem host_sum_real (p : Parameters) (selection : LibrarySelection p)
    (m : Memory ℝ) (o : Fin (valueCount p)) :
    sumValue VeriTile.Bench.Repairs.ChunkDeltaFwd.Real.algebra p selection m o=
      ∑ k : Fin (keyBlocks p), (m "valuePartials" (k.val*valueCount p+o.val)).read .real := by
  exact LibrarySum.real_complete (selection.plans o) _

/-- The host consumes the actual memory produced by all four grids. -/
def completeMemory (M : Algebra α)
    (D : VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.DotPrimitive α)
    (forward backward : Parameters) (scale : ℝ) (seed final terminal : Bool)
    (selection : LibrarySelection backward) (m : Memory α) : Memory α :=
  let kernels := modelFour M D forward backward scale seed final terminal
    (grid .forwardState forward) (grid .forwardOutput forward)
    (grid .backwardState backward) (grid .backwardGradient backward) m
  hostSumMemory M backward selection kernels kernels

theorem complete_value_read (M : Algebra α)
    (D : VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput.DotPrimitive α)
    (forward backward : Parameters) (scale : ℝ) (seed final terminal : Bool)
    (selection : LibrarySelection backward) (m : Memory α) (o : Fin (valueCount backward)) :
    completeMemory M D forward backward scale seed final terminal selection m "valueGradient" o.val=
      .mk .real (cast M (sumValue M backward selection
        (modelFour M D forward backward scale seed final terminal
          (grid .forwardState forward) (grid .forwardOutput forward)
          (grid .backwardState backward) (grid .backwardGradient backward) m) o)) :=
  host_sum_read M backward selection _ _ o

#axiomsClean host_sum_read
#axiomsClean host_sum_frame
#axiomsClean host_sum_real
#axiomsClean complete_value_read
#specNonCircular LibrarySelection avoiding [sumValue,hostSumMemory,LibrarySum.evaluate]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn
