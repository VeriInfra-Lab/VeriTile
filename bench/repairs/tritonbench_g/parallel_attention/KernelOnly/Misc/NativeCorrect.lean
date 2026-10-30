/- Auxiliary single-source native correctness. No program-pair equivalence is claimed. -/
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IONative

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.FPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
open IOData
noncomputable section

/-- Complete FP32 source schedule with the existing source-specific compiler
algebras. Inputs include every symbolic expression and every raw entry image.
The native descriptor has the original unrestricted shape/flag domain; the
Real ShapeContract is not added to this native observation. -/
abbrev attentionIO (direction : Direction) (d : NativePlans.Descriptor) (scale : ℝ) :
    ProgramIO NativeStage.Expr NativeStage.Expr :=
  dataIO NativeStage.Expr direction d.shape (nativeJobs direction d scale)

/-- State-free source dataflow with exact dot/reduction topology, source casts,
source masks and native FP32 scalar interpretation. -/
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

theorem forward_grid (actual reference : Scalars NativeStage.Word)
    (laws : NativeStage.PrimitiveLaws actual reference) (d : NativePlans.Descriptor) (scale : ℝ) :
    (attentionIO .forward d scale).Observes (nativeObserver actual d) (nativeExpected reference d scale) := by
  have same:=laws.eq
  subst reference
  exact IOData.native_forward actual d scale

theorem backward_grid (actual reference : Scalars NativeStage.Word)
    (laws : NativeStage.PrimitiveLaws actual reference) (d : NativePlans.Descriptor) (scale : ℝ) :
    (attentionIO .backward d scale).Observes (nativeObserver actual d) (nativeExpected reference d scale) := by
  have same:=laws.eq
  subst reference
  exact IOData.native_backward actual d scale

#axiomsClean forward_grid
#axiomsClean backward_grid
#stmtSurfaceSubset forward_grid ⊆
  [Scalars,NativeStage.Word,NativeStage.PrimitiveLaws,NativePlans.Descriptor,attentionIO,
   Direction.forward,ProgramIO.Observes,nativeObserver,nativeExpected,NativeStage.Expr,NativeStage.instInhabitedExpr]
#stmtSurfaceSubset backward_grid ⊆
  [Scalars,NativeStage.Word,NativeStage.PrimitiveLaws,NativePlans.Descriptor,attentionIO,
   Direction.backward,ProgramIO.Observes,nativeObserver,nativeExpected,NativeStage.Expr,NativeStage.instInhabitedExpr]
#specNonCircular nativeExpected avoiding [State,FP.Structural.exec,runPrograms,forwardValues,queryValues,backwardValues]
/- PrimitiveLaws is the existing scalar-instruction model boundary. These
statements compare source observations, not two optimized kernels; no floating
transformation rule or new numerical assumption is introduced. -/
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.FPEquiv
