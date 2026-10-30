import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.IONative

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.FPEquiv
open VeriTile VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware (Word)
noncomputable section

/-- The retained compiled graph, as shared Structural phase invocations:
original prelude, then every K-block's original prefix / FP16 FMA / live-carry
MMA suffix, then the original FP16 store epilogue, for the complete grid.
Half payloads use the low 16 bits of Word; FP32 accumulator words use all 32.
The source input/output channels keep their actual dtype tags and strides. -/
def nativeIO (actual : Native.Instructions) (c : RawKernel.Layout) (i : Fin 17) :
    ProgramIO Word Word where
  inputs := [⟨.fp16,[c.M,c.K],"A"⟩,⟨.int,[c.K/8,c.N],"Q"⟩,
    ⟨.fp16,[c.K/c.group,c.N],"S"⟩,⟨.int,[c.K/c.group,c.N/8],"Z"⟩]
  outputs := [⟨.fp16,[c.M,c.N],"C"⟩]
  scratch := []
  buffers := ["A","Q","S","Z","C"]
  read := fun port _=>IOData.readWindow c port
  inputMask := fun _ _ _=>True
  write := fun port _=>IOData.writeWindow c port
  outputMask := fun _ _ _=>True
  scratchWrite := fun port=>port.elim0
  scratchMask := fun port=>port.elim0
  schedule := fun _=>IONativeGraph.jobs actual c i
  encode := id
  observe := fun _ _=>id

/-- Independent coordinate target: rounded half zero product, sign-bit negate,
one half FMA, source-ordered fixed K16 MMA instructions with live FP32 carry,
and final half narrowing. No finite-only or special-value restriction. -/
def nativeFormula (mma : Opcode → ISA.MMA16) (c : RawKernel.Layout) (i : Fin 17) :
    ChannelValues Word (IOData.inputs c) → ChannelValues Word (IOData.outputs c) :=
  fun x=>IOData.expected c (IONativeMath.value (mma (tunings i).opcode) c i (IOData.arrays c x))

/-- The existing four local primitive boundaries are retained: half multiply,
half FMA, FP32-to-half narrowing and a fixed K16 instruction lane. This is a
source-to-native observation, with no new floating transformation rule. -/
theorem fp_equiv (actual : Native.Instructions) (mma : Opcode → ISA.MMA16)
    (ports : Native.Obligations actual mma) (c : RawKernel.Layout) (valid : c.Valid) (i : Fin 17) :
    (nativeIO actual c i).Implements (nativeFormula mma c i) :=
  IONative.implements actual mma ports c valid i

#specNonCircular nativeFormula avoiding [Native.exec,Native.grid,Physical.source,
  FP.Structural.exec,FP.Structural.runSchedule,IONativeGraph.jobs]
#axiomsClean fp_equiv
#stmtSurfaceSubset fp_equiv ⊆ [nativeIO,ProgramIO.Implements,nativeFormula,
  Native.Instructions,Native.Obligations,Opcode,ISA.MMA16,RawKernel.Layout,RawKernel.Layout.Valid,
  Word,VeriTile.Bench.TritonBenchG.Int4Matmul.FPSoftware.instInhabitedWord]
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.FPEquiv
