import bench.tritonbench_g.matrix_vector_multip.Proofs.SourcePreludeInvariant
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.MatrixVectorMultip.Source
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α]

/-- A lossless swap of the two FP32 payload channel tags. Numerical values,
all discrete cells and other float storage channels are untouched. -/
def storageView : Cell α → Cell α
  | .mk .real value => .mk .fp32 value
  | .mk .fp32 value => .mk .real value
  | .mk dtype value => .mk dtype value

def viewMemory (memory : RegionName → Nat → Cell α) : RegionName → Nat → Cell α :=
  fun region offset => storageView (memory region offset)

def viewState (s : State α) : State α := {s with mem := viewMemory s.mem}

theorem storageView_read (cell : Cell α) :
    (storageView cell).read .fp32=cell.read .real := by
  cases cell with
  | mk dtype value => cases dtype <;> rfl

theorem storageView_involutive (cell : Cell α) : storageView (storageView cell)=cell := by
  cases cell with
  | mk dtype value => cases dtype <;> rfl

#axiomsClean storageView_read
#axiomsClean storageView_involutive
end
end VeriTile.Bench.Repairs.MatrixVectorMultip.Source
