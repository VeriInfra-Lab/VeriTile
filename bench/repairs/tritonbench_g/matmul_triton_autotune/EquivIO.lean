import bench.repairs.tritonbench_g.matmul_triton_autotune.IO
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelNative

/-! The existing original and safe-address matmul sources, under exactly the
same FP32 scalar, live-carry instruction, FP16 output and launch configuration.
The source definitions remain in the two task Kernels modules. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.FloatingIO
open VeriTile.Triton FP.Structural Whole
noncomputable section
local instance : Inhabited Floating.Word := ⟨FP.Value.zero .fp32⟩

structure Program where
  instructions : Floating.Instructions
  geometry : Geometry
  ids : List Nat
  complete : ids.Perm (List.range (programs geometry.request))
  safeAddresses : Bool

def Program.kernel (p : Program) : ComputeKernel :=
  if p.safeAddresses then sourceKernel p.geometry.request
  else FPValue.source p.geometry.request.parameters .single

/-- Run the actual source selected by the view, over its entire launch list.
Both alternatives retain the original dot carry, activation and half store. -/
def Program.run (p : Program) (initial : State Floating.Word) : Option (State Floating.Word) :=
  if p.safeAddresses then
    Repair.runPrograms Floating.scalar (Floating.dot p.instructions p.geometry.config)
      p.geometry.request p.ids initial
  else
    Whole.runPrograms Floating.scalar (Floating.dot p.instructions p.geometry.config)
      p.geometry.request p.ids initial

/-- Both source schedules terminate and agree on every typed memory cell.
The relation preserves arbitrary FP32 payloads and the entire exterior frame;
input/output channels and strides are retained in the geometry signature. -/
def Equivalent (lhs rhs : Program) : Prop :=
  ∀ initial, ∃ left right,
    lhs.run initial = some left ∧ rhs.run initial = some right ∧ left.mem = right.mem

instance : Spec.ProgramSyntax Program where
  Statement := ComputeStmt
  Signature := Floating.Instructions × Geometry × List Nat
  signature p := (p.instructions, p.geometry, p.ids)
  body p := p.kernel.surfaceBody
  structural := some Equivalent

def original (instructions : Floating.Instructions) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) : Program :=
  ⟨instructions, g, ids, order, Bool.false⟩

def safe (instructions : Floating.Instructions) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) : Program :=
  ⟨instructions, g, ids, order, Bool.true⟩

end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel.FloatingIO
