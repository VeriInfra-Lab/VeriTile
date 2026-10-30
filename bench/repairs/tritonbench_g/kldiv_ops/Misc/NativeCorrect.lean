import bench.repairs.tritonbench_g.kldiv_ops.Proofs.IONative

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.FPEquiv
open VeriTile Triton FP.Structural
open KldivTriton.Native (Word Node atom)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The actual full FP32 forward source grid, its runtime epsilon ABI value,
and explicit native observation. IO channels and masks are shared with Correct. -/
def forwardIO (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (red : Nat) (eps : BitVec 64) : ProgramIO Word Node :=
  Interface.forwardIO (Contiguous.Native.algebra ports plans) p flag red
    (atom (ports.bindEpsilon eps)) atom Node.value

def backwardIO (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) : ProgramIO Word Node :=
  Interface.backwardIO (Contiguous.Native.algebra ports plans) p flag atom Node.value

/-- Independent ordered FP32 scalar/reduction graph. Plans select the original
per-block addition trees; sequential chunk carries and scalar ABI are retained. -/
abbrev forwardFormula := NativeIO.forwardFormula
abbrev backwardFormula := NativeIO.backwardFormula

/-- All FP32 words are admitted. This is a source-to-native observation theorem;
primitive/compiler meanings are model parameters, with no floating rewrite or
numerical experiment added. -/
theorem forward (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (red : Nat) (eps : BitVec 64)
    (block : p.block=Host.blockSize p.cols) (predictionSeparate : p.Y≠p.O)
    (targetSeparate : p.T≠p.O) :
    forwardIO ports plans p flag red eps ⊨ fun values =>
      Interface.result (forwardFormula ports plans p flag red eps values) := by
  exact NativeIO.forward ports plans p flag red eps block predictionSeparate targetSeparate

theorem backward (ports : Contiguous.Native.Ports) (plans : Contiguous.Native.Plans)
    (p : Parameters) (flag : Bool) (block : 0<p.block) (targetSeparate : p.T≠p.O) :
    backwardIO ports plans p flag ⊨ fun values =>
      Interface.result (backwardFormula ports p flag values) := by
  exact NativeIO.backward ports plans p flag block targetSeparate

#stmtSurfaceSubset forward ⊆
  [Contiguous.Native.Ports,Contiguous.Native.Plans,Parameters,Parameters.block,Parameters.cols,
   Parameters.Y,Parameters.T,Parameters.O,Host.blockSize,Word,Node,
   RegionName,Interface.outputLength,forwardIO,ProgramIO.Implements,ProgramIO.Inputs,Interface.result,forwardFormula,
   KldivTriton.Native.instInhabitedNode]
#stmtSurfaceSubset backward ⊆
  [Contiguous.Native.Ports,Contiguous.Native.Plans,Parameters,Parameters.block,
   Parameters.T,Parameters.O,Word,Node,
   RegionName,Parameters.rows,Parameters.cols,backwardIO,ProgramIO.Implements,ProgramIO.Inputs,Interface.result,backwardFormula,
   KldivTriton.Native.instInhabitedNode]
#specNonCircular forwardFormula avoiding [KldivOps.forward,Program.grid,FP.Structural.exec]
#specNonCircular backwardFormula avoiding [KldivOps.backward,Backward.grid,FP.Structural.exec]
#axiomsClean forward
#axiomsClean backward
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Kernel.FPEquiv
