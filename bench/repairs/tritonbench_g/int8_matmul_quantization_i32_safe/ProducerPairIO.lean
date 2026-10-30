import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIndices

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerIndicesIO
open VeriTile Triton FP.Structural
open KernelProducerGrid (Geometry)
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry) (r : Fin p.rows) : Invocation α :=
  { PublicIO.Producer.invocation M integer8 p r with kernel:=ProducerIndices.candidate p }

def variant (M : Algebra α) (integer8 : α→BitVec 8) (p : Geometry) (order : Producer.Schedule p) :
    ProgramIO.Variant (PublicIO.Producer.io M integer8 p order) where
  schedule:=fun _=>order.order.map (invocation M integer8 p)
  observe:=fun _ _=>id

local instance : Inhabited Word32:=⟨FP.Value.zero .fp32⟩

def original (f : KernelNativeModel.FloatingInstructions) (p : Geometry) (selection : Producer.Selection p)
    (rest : Algebra Word32) (order : Producer.Schedule p) :=
  ProgramIO.Implementation.original (PublicIO.Producer.nativeIO f p selection rest order) (fun _=>True)
def optimized (f : KernelNativeModel.FloatingInstructions) (p : Geometry) (selection : Producer.Selection p)
    (rest : Algebra Word32) (order : Producer.Schedule p) :=
  ProgramIO.Implementation.ofVariant
    (variant (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8 p order) (fun _=>True)

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.ProducerIndicesIO
