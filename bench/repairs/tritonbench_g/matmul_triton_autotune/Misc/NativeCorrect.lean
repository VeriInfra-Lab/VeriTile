import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Interface
import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.KernelNative

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section
local instance nativeWordInhabited : Inhabited Floating.Word := ⟨FP.Value.zero .fp32⟩

/-- The entire selected grid; FP32 input words, live-carry MMA/MFMA, scalar
FP32 activation and mandatory FP16 output all retain their source precision. -/
def nativeIO (instructions : Floating.Instructions) (g : Geometry) (ids : List Nat) :
    ProgramIO Floating.Word Floating.Word :=
  Interface.programIO Floating.scalar (Floating.dot instructions g.config) g ids

/-- Independent padded instruction recurrence on logical input matrices.
Fixed instruction meanings remain the existing model boundary. This result
adds no numerical rewrite law and admits every FP32 input word. -/
def floating (instructions : Floating.Instructions) (g : Geometry)
    (values : Interface.Inputs Floating.Word g) : Interface.Outputs Floating.Word g :=
  Interface.result g (Whole.logicalReference (Floating.numerics instructions g.config)
    g.request (Interface.left values) (Interface.right values))

theorem floating_kernel (instructions : Floating.Instructions) (g : Geometry)
    (ids : List Nat) (order : ids.Perm (List.range (programs g.request)))
    (sepA : g.C ≠ g.A) (sepB : g.C ≠ g.B) :
    nativeIO instructions g ids ⊨ floating instructions g :=
  Interface.implements Floating.scalar (Floating.dot instructions g.config)
    (Floating.numerics instructions g.config) (Floating.primitives instructions g.config)
    g ids order sepA sepB

#specNonCircular floating avoiding [Source.typed, Repair.runPrograms, FP.Structural.exec, FP.Structural.run]
#axiomsClean floating_kernel
#stmtSurfaceSubset floating_kernel ⊆
  [ProgramIO.Implements, nativeIO, floating, Floating.Instructions, Floating.Word, nativeWordInhabited,
   Geometry, Geometry.request, Geometry.A, Geometry.B, Geometry.C, programs, RegionName]
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
