import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOData

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- The mathematical forward value/normalizer and backward value-shard
partials, using logical inputs only and the actual runtime scale. -/
def realExpected (shape : Shape) (scale : ℝ) (xs : ChannelValues ℝ (inputPorts shape)) :
    ChannelValues ℝ (outputPorts shape) := fun out=>match out with
  | ⟨0,_⟩ => fun j=>outputSpec shape.T shape.K scale (headInputs (tensorValues shape xs) j.1.val) (row j) (col j)
  | ⟨1,_⟩ => fun j=>normalizerSpec shape.T shape.K scale (headInputs (tensorValues shape xs) j.1.val)
      (j.2.1.val*128+j.2.2.2.1.val)
  | ⟨2,_⟩ => fun j=>queryPartial shape scale (tensorValues shape xs) j.1.val j.2.2.1.val (row j) j.2.2.2.2.1.val
  | ⟨3,_⟩ => fun j=>keyPartial shape scale (tensorValues shape xs) j.1.val j.2.2.1.val (row j) j.2.2.2.2.1.val
  | ⟨4,_⟩ => fun j=>valueGradient shape scale (tensorValues shape xs) j.1.val (row j) (col j)

/-- Independent numerical graph for every original source phase. The input
carrier is unrestricted symbolic expressions, including the actual entry
images of all raw word cells and their mismatched-tag defaults. -/
def nativeExpected (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (xs : ChannelValues NativeStage.Expr (inputPorts d.shape)) :
    ChannelValues NativeStage.Word (outputPorts d.shape) := fun out=>match out with
  | ⟨0,_⟩ => fun j=>
      let p:=program d.shape 128 d.shape.BV j
      NativeStage.evaluate a .forward d p.tile
        ((IOLogical.forwardValues (NativeStage.algebra .forward d) (NativeStage.dots .forward d.shape)
          (forwardConfig d.shape scale) (pid p) (inputValues d.shape xs)).output ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩)
  | ⟨1,_⟩ => fun j=>
      let p:=normProgram d.shape j
      NativeStage.evaluate a .forward d p.tile
        ((IOLogical.forwardValues (NativeStage.algebra .forward d) (NativeStage.dots .forward d.shape)
          (forwardConfig d.shape scale) (pid p) (inputValues d.shape xs)).normalizer ⟨j.2.2.2.1,()⟩)
  | ⟨2,_⟩ => fun j=>
      let p:=program d.shape 64 d.shape.BK j
      NativeStage.evaluate a .query d p.tile
        (IOLogical.queryValues (NativeStage.algebra .query d) (NativeStage.dots .query d.shape)
          (dqConfig d.shape scale p.tile) (inputValues d.shape xs) ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩)
  | ⟨3,_⟩ => fun j=>
      let p:=program d.shape 64 d.shape.BK j
      NativeStage.evaluate a .keyValue d p.tile
        ((IOLogical.backwardValues (NativeStage.algebra .keyValue d) (NativeStage.dots .keyValue d.shape)
          (backwardConfig d.shape scale p.tile) (inputValues d.shape xs)).key ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩)
  | ⟨4,_⟩ => fun j=>
      let p:=program d.shape 64 d.shape.BV j
      NativeStage.evaluate a .keyValue d p.tile
        ((IOLogical.backwardValues (NativeStage.algebra .keyValue d) (NativeStage.dots .keyValue d.shape)
          (backwardConfig d.shape scale p.tile) (inputValues d.shape xs)).value ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩)

def nativeObserver (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) :
    (out : Fin (outputPorts d.shape).length) → TileIndex (outputPorts d.shape)[out].shape →
      NativeStage.Expr → NativeStage.Word
  | ⟨0,_⟩ => fun j=>NativeStage.evaluate a .forward d (program d.shape 128 d.shape.BV j).tile
  | ⟨1,_⟩ => fun j=>NativeStage.evaluate a .forward d (normProgram d.shape j).tile
  | ⟨2,_⟩ => fun j=>NativeStage.evaluate a .query d (program d.shape 64 d.shape.BK j).tile
  | ⟨3,_⟩ => fun j=>NativeStage.evaluate a .keyValue d (program d.shape 64 d.shape.BK j).tile
  | ⟨4,_⟩ => fun j=>NativeStage.evaluate a .keyValue d (program d.shape 64 d.shape.BV j).tile

#specNonCircular realExpected avoiding [State,FP.Structural.exec,runPrograms]
#specNonCircular nativeExpected avoiding [State,FP.Structural.exec,runPrograms,NativeStage.forwardRun,NativeStage.queryRun,NativeStage.keyValueRun]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
