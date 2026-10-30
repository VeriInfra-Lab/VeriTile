import VeriTile.Triton.Memory.LogicalIO
import VeriTile.Triton.Float.ArrayProgramEquivalenceSyntax
import VeriTile.Triton.Float.ArrayProgramIO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.NativeProjection
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Whole
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Erasure
import bench.tritonbench_g.attention_fwd_triton3.Kernels

/-! Finite six-port IO for the complete ordinary grid. The channel coordinates
retain the program, row and feature dimensions; repeated read coordinates are
bound simultaneously. The mathematical source explicitly erases the original
FP32 quotation. Carry ports alias their own outputs, and L is preserved at END.
The separate existing flat-memory and Tensor transport remains in Auxiliary.
-/
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

theorem source_erasure (c : Call) : (Kernels.source c).eraseDType=c.io.kernel := by
  exact FP32Profile.erases "Q" "K" "V" "M" "O" "L" c.scale
    c.q.batch c.q.head c.q.row c.q.dim
    c.k.batch c.k.head c.k.row c.k.dim
    c.v.batch c.v.head c.v.row c.v.dim
    c.o.batch c.o.head c.o.row c.o.dim
    c.Z c.H c.HK c.NQ c.rounded c.NK
    c.params.evenM c.params.evenN c.BM c.D c.BN c.finish c.init c.window c.complement c.offset c.size

def selectedSource (c : Call) : AlgorithmSource where
  kernel := (Kernels.source c).eraseDType
  projection := by
    rw [source_erasure]
    exact c.io.projection

def programIO (c : Call) : AlgorithmProgramIO where
  nIn := 6
  nOut := 3
  nScr := 0
  bufs := ["Q","K","V","M","L","O"]
  ity := fun _ => .float
  iarity := inputArity c
  ibuf := inputRegion
  oty := fun _ => .float
  oarity := outputArity c
  obuf := outputRegion
  obuf_mem := by intro o; fin_cases o <;> simp [outputRegion]
  sarity := Fin.elim0
  sbuf := Fin.elim0
  iwin := fun i _ _ _ _ j => inputAddress c i ((inputCoordinates c i).symm j)
  imask := fun i _ _ _ _ j => inputActive c i ((inputCoordinates c i).symm j)
  owin := fun o _ _ _ _ j => outputAddress c o ((outputCoordinates c o).symm j)
  omask := fun o _ _ _ _ _ => outputActive c o
  swin := fun i => Fin.elim0 i
  smask := fun i => Fin.elim0 i
  initialUndef := .zero
  stages := fun _ => [.ordinary (selectedSource c) c.grid]

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
      (Inputs c) (Outputs c) (RealInterface.programIO c).Implements where
  inputs := inputsEquiv c
  outputs := outputsEquiv c

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalIO
