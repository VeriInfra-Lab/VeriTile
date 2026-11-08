import bench.tritonbench_g.matrix_vector_multip.IO
import bench.tritonbench_g.matrix_vector_multip.Proofs.KernelFP
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeSource WholeNative
noncomputable section

def originalIO (native : Scalars) (N M BN BM pid : Nat) (g : Graph BN BM) :=
  IO.io (WholeNative.algebra native g) N M BN BM pid Payload.word Payload.bits

def invocation (native : Scalars) (N M BN BM pid : Nat) (g : Graph BN BM) : Invocation Payload :=
  { IO.invocation (WholeNative.algebra native g) N M BN BM pid with
    kernel := Kernels.explicitFMA N M BN BM
    backend := algebra native g }

def variant (native : Scalars) (N M BN BM pid : Nat) (g : Graph BN BM) :
    ProgramIO.Variant (originalIO native N M BN BM pid g) where
  schedule := fun _ => [invocation native N M BN BM pid g]
  observe := fun _ _ => Payload.bits

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.ExplicitFMA

namespace VeriTile.Bench.Repairs.MatrixVectorMultip.GridPair
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.MatrixVectorMultip WholeNative
noncomputable section

def io (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM) (order : List Nat) :
    ProgramIO (BitVec 32) Payload where
  inputs := IO.channels N M
  outputs := [⟨.real,[N],"C"⟩]
  buffers := ["A","B","C"]
  scratch := []
  read := IO.inputRead N M
  inputMask := fun _ _ _ => True
  write := Fin.cases (fun _ j => j.1.val) (fun i => i.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => i.elim0
  scratchMask := fun i => i.elim0
  schedule := fun _ => order.map (IO.invocation (WholeNative.algebra native g) N M BN BM)
  encode := Payload.word
  observe := fun _ _ => Payload.bits

def variant (native : Scalars) (N M BN BM : Nat) (g : Graph BN BM) (order : List Nat) :
    ProgramIO.Variant (io native N M BN BM g order) where
  schedule := fun _ => order.map (ExplicitFMA.invocation native N M BN BM · g)
  observe := fun _ _ => Payload.bits

end
end VeriTile.Bench.Repairs.MatrixVectorMultip.GridPair
