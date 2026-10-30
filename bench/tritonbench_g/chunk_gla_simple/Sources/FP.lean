import bench.tritonbench_g.chunk_gla_simple.Sources.LogicalView
import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_HostFP

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.FPProof
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Host

/-- FP32 ComputeOp loads use the algorithm real cell channel and carry
precision separately. Half/BF16 loads retain their storage tags. -/
def storageType : Storage → FloatDType
  | .fp32 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

def inputType (dot value gate : Storage) (r : RegionName) : FloatDType :=
  if r = "v" then storageType value
  else if r = "g" then storageType gate
  else storageType dot

def bind {α : Type} (types : RegionName → FloatDType) (s : State α) (read : RegionName → Nat → Option α) : State α :=
  {s with mem := fun r i => match read r i with
    | some x => Cell.mk (types r).toTileDType (ofFloat (types r) x)
    | none => s.mem r i}

theorem bind_read {α : Type} [Inhabited α] (types : RegionName → FloatDType)
    (s : State α) (read : RegionName → Nat → Option α) (r : RegionName) (i : Nat)
    (x : α) (hx : read r i = some x) :
    ((bind types s read).mem r i).read (types r).toTileDType = ofFloat (types r) x := by
  simp [bind,hx]

/-- The source-default FP32 loader really receives each supplied logical cell,
not the default value from a mismatched storage tag. -/
theorem fp32_bind_read {α : Type} [Inhabited α] (s : State α)
    (read : RegionName → Nat → Option α) (r : RegionName) (i : Nat)
    (x : α) (hx : read r i = some x) :
    ((bind (inputType .fp32 .fp32 .fp32) s read).mem r i).read .real = x := by
  have h : inputType .fp32 .fp32 .fp32 r = .real := by
    unfold inputType storageType
    split_ifs <;> rfl
  simp only [bind,hx]
  rw [h]
  rfl

theorem prepared_state {α : Type} (types : RegionName → FloatDType) (s : State α) (a : Logical.Inputs α)
    (p : Logical.PreparedInputs a) : bind types s p.read = bind types s a.read := by
  rw [Logical.prepared_read a p]

/-- Every optional full execution state agrees after materialization. This is
an exact logical-input equality theorem for the actual candidate, for every
retained production precision. It assumes no numerical identity or successful
execution; both success and failure are preserved. -/
theorem execution_on_logical_values {α : Type} [Inhabited α] (M : Algebra α)
    (c : Shape) (dot value gate : Storage) (out : FloatDType) (scale : ℝ)
    (s : State α) (a : Logical.Inputs α) (p : Logical.PreparedInputs a)
    (idx : GridIndex (grid c)) :
    FP.Structural.exec (FP.ProfiledRewrite.engine M) (kernel c dot value gate out scale)
      (HostFP.atProgram c (bind (inputType dot value gate) s p.read) idx) =
    FP.Structural.exec (FP.ProfiledRewrite.engine M) (kernel c dot value gate out scale)
      (HostFP.atProgram c (bind (inputType dot value gate) s a.read) idx) := by
  rw [prepared_state (inputType dot value gate) s a p]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaSimple.FPProof
