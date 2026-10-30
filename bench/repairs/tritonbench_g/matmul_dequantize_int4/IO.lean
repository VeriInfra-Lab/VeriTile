import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.RawKernel
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.Bounds
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Triton.Float.LogicalProgramIO


/-! Canonical typed tensor ports, strided windows, logical bindings and launch data. -/
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOData
open VeriTile.Triton FP.Structural
noncomputable section

abbrev inputs (c : RawKernel.Layout) : List IOChannel :=
  [⟨.fp16,[c.M,c.K],"A"⟩,⟨.int,[c.K/8,c.N],"Q"⟩,
   ⟨.fp16,[c.K/c.group,c.N],"S"⟩,⟨.int,[c.K/c.group,c.N/8],"Z"⟩]
abbrev outputs (c : RawKernel.Layout) : List IOChannel := [⟨.fp16,[c.M,c.N],"C"⟩]

def readWindow (c : RawKernel.Layout) :
    (port : Fin (inputs c).length) → TileIndex (inputs c)[port].shape → Nat
  | ⟨0,_⟩ => fun j => c.A.addr j.1.val j.2.1.val
  | ⟨1,_⟩ => fun j => c.Q.addr j.1.val j.2.1.val
  | ⟨2,_⟩ => fun j => c.S.addr j.1.val j.2.1.val
  | ⟨3,_⟩ => fun j => c.Z.addr j.1.val j.2.1.val

def writeWindow (c : RawKernel.Layout) :
    (port : Fin (outputs c).length) → TileIndex (outputs c)[port].shape → Nat
  | ⟨0,_⟩ => fun j => c.C.addr j.1.val j.2.1.val

abbrev memory (α : Type) (c : RawKernel.Layout) : MemoryIO α where
  inputs := inputs c
  outputs := outputs c
  scratch := []
  buffers := ["A","Q","S","Z","C"]
  read := fun port _ => readWindow c port
  inputMask := fun _ _ _ => True
  write := fun port _ => writeWindow c port
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0

abbrev program (α : Type) (c : RawKernel.Layout) (jobs : List (Invocation α)) :
    ProgramIO α α :=
  { memory α c with schedule := fun _ => jobs, encode := id, observe := fun _ _ => id }

/-- Coordinate arrays have exactly the public finite shapes. Values outside
those shapes are unused extensions, not extra memory bindings. -/
structure Arrays (α : Type) where
  A : Nat → Nat → α
  Q : Nat → Nat → Int
  S : Nat → Nat → α
  Z : Nat → Nat → Int

def arrays {α : Type} [Inhabited α] (c : RawKernel.Layout)
    (x : ChannelValues α (inputs c)) : Arrays α where
  A := fun r k => if h : r<c.M ∧ k<c.K then x 0 (⟨r,h.1⟩,⟨k,h.2⟩,()) else default
  Q := fun k n => if h : k<c.K/8 ∧ n<c.N then x 1 (⟨k,h.1⟩,⟨n,h.2⟩,()) else 0
  S := fun g n => if h : g<c.K/c.group ∧ n<c.N then x 2 (⟨g,h.1⟩,⟨n,h.2⟩,()) else default
  Z := fun g n => if h : g<c.K/c.group ∧ n<c.N/8 then x 3 (⟨g,h.1⟩,⟨n,h.2⟩,()) else 0

def expected {α : Type} (c : RawKernel.Layout) (f : Nat → Nat → α) :
    ChannelValues α (outputs c)
  | ⟨0,_⟩ => fun j => f j.1.val j.2.1.val

/-- The signed-int32 ABI constrains only the logical packed Q and Z inputs. -/
def packedInt32 {α : Type} (c : RawKernel.Layout) (x : ChannelValues α (inputs c)) : Prop :=
  (∀ j : TileIndex [c.K/8,c.N], ∃ w : BitVec 32, x 1 j=w.toInt) ∧
  (∀ j : TileIndex [c.K/c.group,c.N/8], ∃ w : BitVec 32, x 3 j=w.toInt)

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOData

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOExecution
open VeriTile.Triton FP.Structural
noncomputable section

def launch {α : Type} (source : ComputeKernel) (model : Algebra α) (pid : Nat) : Invocation α where
  kernel := source
  backend := model
  pids := fun ax=>if ax=0 then pid else 0

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOExecution

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOReal
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
noncomputable section

def jobs (c : RawKernel.Layout) (i : Fin 17) : List (Invocation ℝ) :=
  (List.range (KernelSource.count c.call i)).map
    (IOExecution.launch (KernelSource.kernel c.call i) (backend KernelReal.scalar KernelReal.dot))

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOReal


namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOMath
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG
open MatmulDequantizeInt4.FPTyped
open MatmulKernel.RealSum
noncomputable section

def input (c : RawKernel.Layout) (x : ChannelValues ℝ (IOData.inputs c)) : Real.Input where
  A := (IOData.arrays c x).A
  Q := fun k n => BitVec.ofInt 32 ((IOData.arrays c x).Q k n)
  S := (IOData.arrays c x).S
  Z := fun g n => BitVec.ofInt 32 ((IOData.arrays c x).Z g n)

end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.IOMath

namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.ReaderIO
open VeriTile Triton FP.Structural
noncomputable section

/-- Actual logical packed matrix operands, connected to their complete typed
backing. The graph retains every signed raw Q/Z cell and its original domain. -/
structure Inputs (c : RawKernel.Layout) where
  backing : ChannelValues ℝ (IOData.inputs c)
  arrays : Real.Input
  connected : arrays = IOMath.input c backing

def inputs (c : RawKernel.Layout) : ChannelValues ℝ (IOData.inputs c) ≃ Inputs c where
  toFun := fun raw => ⟨raw,IOMath.input c raw,rfl⟩
  invFun := Inputs.backing
  left_inv := fun _ => rfl
  right_inv := by
    rintro ⟨raw,arrays,connected⟩
    cases connected
    rfl

def packedInt32 (c : RawKernel.Layout) (xs : Inputs c) : Prop := IOData.packedInt32 c xs.backing

/-- Full original strided grid, all 17 tunings and exact typed exterior frame. -/
def view (c : RawKernel.Layout) (i : Fin 17) :
    LogicalProgramIO ℝ ℝ (Inputs c) (Fin c.M → Fin c.N → ℝ) where
  core := IOData.program ℝ c (IOReal.jobs c i)
  inputs := inputs c
  outputs := (ChannelValues.singleEquiv ℝ _).trans (Values.matrixEquiv ℝ .fp16 c.M c.N)
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.ReaderIO
