import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealBackend
import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Memory.LogicalIO
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax
import VeriTile.Triton.Float.ArrayProgramIO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeExpr
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Grid
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Ownership
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Support.KernelEntryAdapters

/-! Finite six-port layouts for the complete ordinary grid. Channel coordinates retain program, row and feature dimensions; repeated reads are pinned simultaneously. The explicit Real backend and native graph backend execute the same repaired source AST. M/L/O carry ports alias their outputs; END preserves L. Logical coordinate equivalences retain all input and output data. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section

local instance finiteTile (shape : TileShape) : Fintype (TileIndex shape) :=
  Fintype.ofList (TileShape.allIndices shape) (TileShape.mem_allIndices shape)

def inputShape (c : Call) := RealIO.inputShape c.BM c.D c.NK
def outputShape (c : Call) := RealIO.outputShape c.BM c.D
abbrev InputPoint (c : Call) (i : Fin 6) := GridIndex c.grid × TileIndex (inputShape c i)
abbrev OutputPoint (c : Call) (o : Fin 3) := GridIndex c.grid × TileIndex (outputShape c o)
def inputArity (c : Call) (i : Fin 6) := Fintype.card (InputPoint c i)
def outputArity (c : Call) (o : Fin 3) := Fintype.card (OutputPoint c o)
def inputCoordinates (c : Call) (i : Fin 6) : InputPoint c i ≃ Fin (inputArity c i) :=
  Fintype.equivFin _
def outputCoordinates (c : Call) (o : Fin 3) : OutputPoint c o ≃ Fin (outputArity c o) :=
  Fintype.equivFin _

abbrev Inputs (c : Call) := (i : Fin 6) → Fin (inputArity c i) → ℝ
abbrev Outputs (c : Call) := (o : Fin 3) → Fin (outputArity c o) → ℝ

def inputRegion : Fin 6 → RegionName := !["Q","K","V","M","L","O"]
def outputRegion : Fin 3 → RegionName := !["M","L","O"]

def inputAddress (c : Call) (i : Fin 6) (j : InputPoint c i) : Nat :=
  c.io.inputWindow i (c.tile j.1) (c.hz j.1) 0 j.2
def inputActive (c : Call) (i : Fin 6) (j : InputPoint c i) : Prop :=
  c.io.inputMask i (c.tile j.1) (c.hz j.1) 0 j.2
def outputAddress (c : Call) (o : Fin 3) (j : OutputPoint c o) : Nat :=
  c.io.outputWindowFor o (c.tile j.1) (c.hz j.1) 0 j.2
def outputActive (c : Call) (o : Fin 3) : Prop := c.io.outputMask o

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

/-! Native IO keeps the original six arbitrary decoded input arrays and a
finite, source-free access footprint. Every FP32 storage/operation/cast stays
in the selected quotation. The raw FP32 scale slot is bound by the existing
ScaleBinding interpretation, including signed zero and NaN payloads.
M/L/O inputs alias the corresponding outputs. M/O include padding rows;
L is written only for continuation. There is no private scratch buffer.
-/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Ownership
noncomputable section
attribute [local instance] Classical.propDecidable

def inputPorts : List ArrayIOChannel :=
  [⟨.real,"Q"⟩,⟨.real,"K"⟩,⟨.real,"V"⟩,⟨.real,"M"⟩,⟨.real,"L"⟩,⟨.real,"O"⟩]
def outputPorts (c : Call) : List IOChannel := List.ofFn (fun o : Fin 3 =>
  ⟨.real,[RealInterface.outputArity c o],RealInterface.outputRegion o⟩)

@[simp] theorem input_port (i : Fin 6) :
    inputPorts[i]=⟨.real,RealInterface.inputRegion i⟩ := by fin_cases i <;> rfl
@[simp] theorem output_port (c : Call) (o : Fin 3) :
    (outputPorts c)[o]=⟨.real,[RealInterface.outputArity c o],RealInterface.outputRegion o⟩ := by
  fin_cases o <;> rfl

abbrev Inputs := ArrayValues Native.Expr inputPorts

def arrayValues (v : Inputs) (i : Fin 6) (off : Nat) : Native.Expr :=
  cast (congrArg (Value Native.Expr) (congrArg ArrayIOChannel.dtype (input_port i))) (v i off)
def outputIndex (c : Call) (o : Fin 3) (j : TileIndex (outputPorts c)[o].shape) :
    Fin (RealInterface.outputArity c o) :=
  (cast (congrArg TileIndex (congrArg IOChannel.shape (output_port c o))) j).1
def outputPoint (c : Call) (o : Fin 3) (j : TileIndex (outputPorts c)[o].shape) :
    RealInterface.OutputPoint c o := (RealInterface.outputCoordinates c o).symm (outputIndex c o j)


def outputEncoded (c : Call) (o : Fin 3) (j : RealInterface.OutputPoint c o) :
    TileIndex (outputPorts c)[o].shape :=
  cast (congrArg TileIndex (congrArg IOChannel.shape (output_port c o))).symm
    (RealInterface.outputCoordinates c o j,PUnit.unit)

@[simp] theorem outputPoint_encoded (c : Call) (o : Fin 3) (j : RealInterface.OutputPoint c o) :
    outputPoint c o (outputEncoded c o j)=j := by
  fin_cases o <;> simp [outputPoint,outputIndex,outputEncoded] <;> exact Equiv.symm_apply_apply _ _

/-- Explicit finite Q/K/V and optional carry windows, with the same affine
strides and masks as the Real ports. Unused array suffixes need no allocation. -/
def reads (c : Call) (i : Fin 6) : List Nat :=
  ((List.finRange (RealInterface.inputArity c i)).filter fun j =>
    decide (RealInterface.inputActive c i ((RealInterface.inputCoordinates c i).symm j))).map
      fun j => RealInterface.inputAddress c i ((RealInterface.inputCoordinates c i).symm j)

def job (c : Call) (idx : GridIndex c.grid) : Invocation Native.Expr where
  kernel := Kernels.source (ScaleBinding.template c)
  backend := engine Native.backend.algebra
  clearRegisters := Bool.true
  pids := idx.toPids
  numPids := some c.grid.toNumPids

def programIO (c : Call) : ArrayProgramIO Native.Expr Native.Expr where
  inputs := inputPorts
  outputs := outputPorts c
  scratch := []
  buffers := ["Q","K","V","M","L","O"]
  reads := fun i _ => reads c i
  write := fun o _ j => RealInterface.outputAddress c o (outputPoint c o j)
  outputMask := fun o _ _ => RealInterface.outputActive c o
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => (Reference.Composition.allPrograms c.grid).map (job c)
  encode := id
  observe := fun _ _ => id

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.NativeInterface

/-! Candidate IO reuses the complete canonical native layout. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.CSEInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership NativeInterface
noncomputable section

def job (c : Call) (idx : GridIndex c.grid) : Invocation Native.Expr :=
  {NativeInterface.job c idx with kernel := Kernels.keyMaskCSE (ScaleBinding.template c)}

def programIO (c : Call) : ArrayProgramIO Native.Expr Native.Expr :=
  {NativeInterface.programIO c with
    schedule := fun _ => (Reference.Composition.allPrograms c.grid).map (job c)}

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.CSEInterface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host Ownership
noncomputable section

def localInputs (c : Call) (v : Inputs c) (idx : GridIndex c.grid) :
    RealInputs.Inputs c.BM c.D c.NK where
  query := fun r d => v 0 (inputCoordinates c 0 (idx,r,d,PUnit.unit))
  key := fun k d => v 1 (inputCoordinates c 1 (idx,k,d,PUnit.unit))
  value := fun k d => v 2 (inputCoordinates c 2 (idx,k,d,PUnit.unit))
  center := fun r => v 3 (inputCoordinates c 3 (idx,r,PUnit.unit))
  denominator := fun r => v 4 (inputCoordinates c 4 (idx,r,PUnit.unit))
  output := fun r d => v 5 (inputCoordinates c 5 (idx,r,d,PUnit.unit))

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host
noncomputable section

def modelInputPorts (c : Call) : List IOChannel :=
  [⟨.real,[inputArity c 0],inputRegion 0⟩,⟨.real,[inputArity c 1],inputRegion 1⟩,⟨.real,[inputArity c 2],inputRegion 2⟩,⟨.real,[inputArity c 3],inputRegion 3⟩,⟨.real,[inputArity c 4],inputRegion 4⟩,⟨.real,[inputArity c 5],inputRegion 5⟩]

def modelOutputPorts (c : Call) : List IOChannel := NativeInterface.outputPorts c

@[simp] theorem model_input_port (c : Call) (i : Fin 6) :
    (modelInputPorts c)[i]=⟨.real,[inputArity c i],inputRegion i⟩ := by fin_cases i <;> rfl

@[simp] theorem model_output_port (c : Call) (i : Fin 3) :
    (modelOutputPorts c)[i]=⟨.real,[outputArity c i],outputRegion i⟩ := by fin_cases i <;> rfl

def packInputs (c : Call) (values : Inputs c) : ChannelValues ℝ (modelInputPorts c) :=
  Fin.cases (fun j => values 0 j.1) (Fin.cases (fun j => values 1 j.1) (Fin.cases (fun j => values 2 j.1) (Fin.cases (fun j => values 3 j.1) (Fin.cases (fun j => values 4 j.1) (Fin.cases (fun j => values 5 j.1) ((fun i => Fin.elim0 i)))))))

def unpackInputs (c : Call) (raw : ChannelValues ℝ (modelInputPorts c)) : Inputs c :=
  Fin.cases (fun j => raw (0 : Fin 6) (j,PUnit.unit)) (Fin.cases (fun j => raw (1 : Fin 6) (j,PUnit.unit)) (Fin.cases (fun j => raw (2 : Fin 6) (j,PUnit.unit)) (Fin.cases (fun j => raw (3 : Fin 6) (j,PUnit.unit)) (Fin.cases (fun j => raw (4 : Fin 6) (j,PUnit.unit)) (Fin.cases (fun j => raw (5 : Fin 6) (j,PUnit.unit)) ((fun i => Fin.elim0 i)))))))

def modelInputsEquiv (c : Call) : ChannelValues ℝ (modelInputPorts c) ≃ Inputs c where
  toFun:=unpackInputs c
  invFun:=packInputs c
  left_inv:=by
    intro raw
    funext i j
    fin_cases i <;> rcases j with ⟨r,tail⟩ <;> cases tail <;> rfl
  right_inv:=by
    intro values
    funext i j
    fin_cases i <;> rfl

def packOutputs (c : Call) (values : Outputs c) : ChannelValues ℝ (modelOutputPorts c) :=
  Fin.cases (fun j => values 0 j.1) (Fin.cases (fun j => values 1 j.1) (Fin.cases (fun j => values 2 j.1) ((fun i => Fin.elim0 i))))

def unpackOutputs (c : Call) (raw : ChannelValues ℝ (modelOutputPorts c)) : Outputs c :=
  Fin.cases (fun j => raw (0 : Fin 3) (j,PUnit.unit)) (Fin.cases (fun j => raw (1 : Fin 3) (j,PUnit.unit)) (Fin.cases (fun j => raw (2 : Fin 3) (j,PUnit.unit)) ((fun i => Fin.elim0 i))))

def modelOutputsEquiv (c : Call) : ChannelValues ℝ (modelOutputPorts c) ≃ Outputs c where
  toFun:=unpackOutputs c
  invFun:=packOutputs c
  left_inv:=by
    intro raw
    funext i j
    fin_cases i <;> rcases j with ⟨r,tail⟩ <;> cases tail <;> rfl
  right_inv:=by
    intro values
    funext i j
    fin_cases i <;> rfl

def modelInputIndex (c : Call) (i : Fin 6) (j : TileIndex (modelInputPorts c)[i].shape) :
    Fin (inputArity c i) :=
  (cast (congrArg TileIndex (congrArg IOChannel.shape (model_input_port c i))) j).1

def modelJob (c : Call) (idx : GridIndex c.grid) : Invocation Real.R where
  kernel:=Kernels.source c
  backend:=FP.ProfiledRewrite.engine Real.ops.algebra
  clearRegisters:=Bool.true
  pids:=idx.toPids
  numPids:=some c.grid.toNumPids

def modelCore (c : Call) : ProgramIO ℝ Real.R where
  inputs:=modelInputPorts c
  outputs:=modelOutputPorts c
  scratch:=[]
  buffers:=["Q","K","V","M","L","O"]
  read:=fun i _ j=>inputAddress c i ((inputCoordinates c i).symm (modelInputIndex c i j))
  inputMask:=fun i _ j=>inputActive c i ((inputCoordinates c i).symm (modelInputIndex c i j))
  write:=fun o _ j=>outputAddress c o (NativeInterface.outputPoint c o j)
  outputMask:=fun o _ _=>outputActive c o
  scratchWrite:=fun i=>Fin.elim0 i
  scratchMask:=fun i=>Fin.elim0 i
  schedule:=fun _=>(Reference.Composition.allPrograms c.grid).map (modelJob c)
  encode:=fun x=>some x
  observe:=fun _ _ x=>x.unbotD 0

def modelIO (c : Call) : LogicalProgramIO ℝ Real.R (Inputs c) (Outputs c) where
  core:=modelCore c
  inputs:=modelInputsEquiv c
  outputs:=modelOutputsEquiv c

theorem model_uses_source (c : Call) (idx : GridIndex c.grid) :
    (modelJob c idx).kernel=Kernels.source c := rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.RealInterface

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalIO
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host
noncomputable section

/-- Six logical arrays at each program coordinate, tied to every original
input window. Carry aliases, padded rows and all input masks are retained. -/
structure Inputs (c : Call) where
  programs : GridIndex c.grid → RealInputs.Inputs c.BM c.D c.NK
  backing : RealInterface.Inputs c
  connection : programs = RealInterface.localInputs c backing

structure Outputs (c : Call) where
  center : GridIndex c.grid → Fin c.BM → ℝ
  denominator : GridIndex c.grid → Fin c.BM → ℝ
  output : GridIndex c.grid → Fin c.BM → Fin c.D → ℝ

def inputsEquiv (c : Call) : RealInterface.Inputs c ≃ Inputs c where
  toFun := fun raw => ⟨RealInterface.localInputs c raw,raw,rfl⟩
  invFun := Inputs.backing
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨programs,backing,connection⟩
    cases connection
    rfl

def outputsEquiv (c : Call) : RealInterface.Outputs c ≃ Outputs c where
  toFun := fun values => {
    center := fun program row => values 0 (RealInterface.outputCoordinates c 0 (program,row,()))
    denominator := fun program row => values 1 (RealInterface.outputCoordinates c 1 (program,row,()))
    output := fun program row col => values 2 (RealInterface.outputCoordinates c 2 (program,row,col,())) }
  invFun := fun values => Fin.cases (fun j =>
    let point := (RealInterface.outputCoordinates c 0).symm j
    values.center point.1 point.2.1)
    (Fin.cases (fun j =>
      let point := (RealInterface.outputCoordinates c 1).symm j
      values.denominator point.1 point.2.1)
      (Fin.cases (fun j =>
        let point := (RealInterface.outputCoordinates c 2).symm j
        values.output point.1 point.2.1 point.2.2.1) (fun i => Fin.elim0 i)))
  left_inv := by
    intro values
    funext o j
    fin_cases o
    · change values 0 ((RealInterface.outputCoordinates c 0)
        ((RealInterface.outputCoordinates c 0).symm j)) = values 0 j
      rw [Equiv.apply_symm_apply]
    · change values 1 ((RealInterface.outputCoordinates c 1)
        ((RealInterface.outputCoordinates c 1).symm j)) = values 1 j
      rw [Equiv.apply_symm_apply]
    · change values 2 ((RealInterface.outputCoordinates c 2)
        ((RealInterface.outputCoordinates c 2).symm j)) = values 2 j
      rw [Equiv.apply_symm_apply]
  right_inv := by
    rintro ⟨center,denominator,output⟩
    dsimp only
    congr 1
    · funext program row
      change center (((RealInterface.outputCoordinates c 0).symm
        (RealInterface.outputCoordinates c 0 (program,row,()))).1)
        (((RealInterface.outputCoordinates c 0).symm
        (RealInterface.outputCoordinates c 0 (program,row,()))).2.1) = center program row
      rw [Equiv.symm_apply_apply]
    · funext program row
      change denominator (((RealInterface.outputCoordinates c 1).symm
        (RealInterface.outputCoordinates c 1 (program,row,()))).1)
        (((RealInterface.outputCoordinates c 1).symm
        (RealInterface.outputCoordinates c 1 (program,row,()))).2.1) = denominator program row
      rw [Equiv.symm_apply_apply]
    · funext program row col
      change output (((RealInterface.outputCoordinates c 2).symm
        (RealInterface.outputCoordinates c 2 (program,row,col,()))).1)
        (((RealInterface.outputCoordinates c 2).symm
        (RealInterface.outputCoordinates c 2 (program,row,col,()))).2.1)
        (((RealInterface.outputCoordinates c 2).symm
        (RealInterface.outputCoordinates c 2 (program,row,col,()))).2.2.1) = output program row col
      rw [Equiv.symm_apply_apply]

def io (c : Call) :
    LogicalIO (RealInterface.Inputs c) (RealInterface.Outputs c)
      (Inputs c) (Outputs c) (RealInterface.modelIO c).Implements where
  inputs := inputsEquiv c
  outputs := outputsEquiv c

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalIO
