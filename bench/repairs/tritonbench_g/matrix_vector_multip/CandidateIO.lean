import bench.repairs.tritonbench_g.matrix_vector_multip.IO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelFP
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeSource WholeNative
noncomputable section

/-- The existing native source model already interprets each product/add pair
as a single FMA carry. This explicit primitive constructs that same carry;
there is no equation between a separately rounded multiply/add and FMA. -/
def algebra (native : Scalars) (g : Graph BN BM) : Algebra Payload :=
  { WholeNative.algebra native g with
    fma := fun _ => some (fun a b c => .carry c a b) }

def originalIO (native : Scalars) (N M BN BM pid : Nat) (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (g : Graph BN BM) :=
  IO.io (WholeNative.algebra native g) N M BN BM pid rowShape columnShape Payload.word Payload.bits

def invocation (native : Scalars) (N M BN BM pid : Nat) (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (g : Graph BN BM) : Invocation Payload :=
  { IO.invocation (WholeNative.algebra native g) N M BN BM pid rowShape columnShape with
    kernel := Kernels.explicitFMA N M BN BM rowShape columnShape
    backend := algebra native g }

def variant (native : Scalars) (N M BN BM pid : Nat) (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (g : Graph BN BM) :
    ProgramIO.Variant (originalIO native N M BN BM pid rowShape columnShape g) where
  schedule := fun _ => [invocation native N M BN BM pid rowShape columnShape g]
  observe := fun _ _ => Payload.bits

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.GridPair
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section

def io (native : Scalars) (N M BN BM : Nat) (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (g : Graph BN BM) (order : List Nat) :
    ProgramIO (BitVec 32) Payload where
  inputs := IO.channels N M
  outputs := [⟨.fp32,[N],"C"⟩]
  buffers := ["A","B","C"]
  scratch := []
  read := IO.inputRead N M
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => order.map (fun pid => IO.invocation (WholeNative.algebra native g) N M BN BM pid rowShape columnShape)
  encode := Payload.word
  observe := fun _ _ => Payload.bits

def variant (native : Scalars) (N M BN BM : Nat) (rowShape : Random.ArangeSizeDomain BN) (columnShape : Random.ArangeSizeDomain BM) (g : Graph BN BM) (order : List Nat) :
    ProgramIO.Variant (io native N M BN BM rowShape columnShape g order) where
  schedule := fun _ => order.map (fun pid => ExplicitFMA.invocation native N M BN BM pid rowShape columnShape g)
  observe := fun _ _ => Payload.bits

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.GridPair
