import bench.tritonbench_g.parallel_attention.NativeMemory

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Independent source-dataflow recurrence under the fixed staged native
interpretation. It reads initial cells and has no executor or store dependency. -/
def queryTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile)
    (s : State Word) (r : Fin 64) (e : Fin d.shape.BK) : Word :=
  evaluate a .query d tile (queryValues (algebra .query d) (dots .query d.shape)
    (dqConfig d.shape σ tile) (entry s) ⟨r,e,()⟩)

def keyTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile)
    (s : State Word) (r : Fin 64) (e : Fin d.shape.BK) : Word :=
  evaluate a .keyValue d tile ((backwardValues (algebra .keyValue d) (dots .keyValue d.shape)
    (backwardConfig d.shape σ tile) (entry s)).key ⟨r,e,()⟩)

def valueTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile)
    (s : State Word) (r : Fin 64) (v : Fin d.shape.BV) : Word :=
  evaluate a .keyValue d tile ((backwardValues (algebra .keyValue d) (dots .keyValue d.shape)
    (backwardConfig d.shape σ tile) (entry s)).value ⟨r,v,()⟩)

def outputTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (s : State Word) (r : Fin 128) (v : Fin d.shape.BV) : Word :=
  evaluate a .forward d (launchTile d.shape s) ((forwardValues (algebra .forward d) (dots .forward d.shape)
    (forwardConfig d.shape σ) (entry s)).output ⟨r,v,()⟩)

def normalizerTarget (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ)
    (s : State Word) (r : Fin 128) : Word :=
  evaluate a .forward d (launchTile d.shape s) ((forwardValues (algebra .forward d) (dots .forward d.shape)
    (forwardConfig d.shape σ) (entry s)).normalizer ⟨r,()⟩)

def queryRun (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile) (s : State Word) : Option (State Word) :=
  (FP.Structural.exec (withDot (algebra .query d) (dots .query d.shape))
    (queryKernel (dqConfig d.shape σ tile)) (entry s)).map (finish a .query d tile)
def keyValueRun (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (tile : Tile) (s : State Word) : Option (State Word) :=
  (FP.Structural.exec (withDot (algebra .keyValue d) (dots .keyValue d.shape))
    (backwardKernel (backwardConfig d.shape σ tile)) (entry s)).map (finish a .keyValue d tile)
def forwardRun (a : Scalars Word) (d : NativePlans.Descriptor) (σ : ℝ) (s : State Word) : Option (State Word) :=
  (FP.Structural.exec (withDot (algebra .forward d) (dots .forward d.shape))
    (forwardKernel (forwardConfig d.shape σ)) (entry s)).map (finish a .forward d (launchTile d.shape s))

#specNonCircular queryTarget avoiding [queryRun,FP.Structural.exec,FP.Structural.run,queryOutputMemory,queryKernel]
#specNonCircular keyTarget avoiding [keyValueRun,FP.Structural.exec,FP.Structural.run,backwardOutputMemory,backwardKernel]
#specNonCircular valueTarget avoiding [keyValueRun,FP.Structural.exec,FP.Structural.run,backwardOutputMemory,backwardKernel]
#specNonCircular outputTarget avoiding [forwardRun,FP.Structural.exec,FP.Structural.run,outputMemory,forwardKernel]
#specNonCircular normalizerTarget avoiding [forwardRun,FP.Structural.exec,FP.Structural.run,outputMemory,forwardKernel]
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.NativeStage
