import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileComplete
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.TileValue
import bench.repairs.tritonbench_g.int8_matmul_kernel_modular_checked.Proofs.StorageProfile
import bench.tritonbench_g.int8_matmul_kernel.Whole001.Proofs.Grid

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section
set_option maxHeartbeats 6000000

def OutputLayout (layout : Layout) (s : Shape) : Prop :=
  layout.outputRowStride = max s.n 1 ∧ layout.outputColumnStride = 1

def matrixWordTarget (layout : Layout) (s : Shape) (state : State Word32) (m n : Nat) : Word32 :=
  denseWordTarget s.k (inputWords layout state m) (packedWords layout state n)

def outputRow (p : Parameters) (state : State Word32) (i : CIndex p) : Nat :=
  rowBlock p state*p.blockRows+i.1.val
def outputColumn (p : Parameters) (state : State Word32) (i : CIndex p) : Nat :=
  columnBlock p state*p.blockColumns+i.2.1.val

theorem output_coordinates_source (layout : Layout) (s : Shape) (c : Fin 15)
    (state : State Word32) (i : CIndex (parameters layout s c)) (hout : OutputLayout layout s) :
    outputOffset (parameters layout s c) state i =
      Whole001.outputOffset s (outputRow (parameters layout s c) state i)
        (outputColumn (parameters layout s c) state i) := by
  simp [outputOffset, Whole001.outputOffset, outputRow, outputColumn, parameters, hout.1, hout.2,
    Nat.mul_comm]

theorem output_injective_source (layout : Layout) (s : Shape) (c : Fin 15)
    (state : State Word32) (hout : OutputLayout layout s)
    (i j : CIndex (parameters layout s c))
    (hi : active (parameters layout s c) state i) (hj : active (parameters layout s c) state j)
    (he : outputOffset (parameters layout s c) state i = outputOffset (parameters layout s c) state j) :
    i = j := by
  rw [output_coordinates_source _ _ _ _ _ hout, output_coordinates_source _ _ _ _ _ hout] at he
  have coords := Whole001.output_injective s _ _ _ _ hi.2 hj.2 he
  rcases i with ⟨ir,ic,iu⟩
  rcases j with ⟨jr,jc,ju⟩
  cases iu
  cases ju
  have hr : ir = jr := Fin.ext (Nat.add_left_cancel coords.1)
  have hc : ic = jc := Fin.ext (Nat.add_left_cancel coords.2)
  simp [hr,hc]

theorem active_value (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (state : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s state) (bytes : UInt8Domain layout s state)
    (i : CIndex (parameters layout s c)) (hi : active (parameters layout s c) state i) :
    outerValues cast (parameters layout s c) state 4 i =
      (matrixWordTarget layout s state (outputRow (parameters layout s c) state i)
        (outputColumn (parameters layout s c) state i)).toInt := by
  rw [outerValues_low32 cast castLeaf layout s c legal state mpos npos defined bytes i]
  unfold matrixWordTarget inputRow inputColumn outputRow outputColumn
  rw [Nat.mod_eq_of_lt hi.1, Nat.mod_eq_of_lt hi.2]

/-- Complete projected quotation execution, with an independent mathematical
word target on every active store and exact preservation of all other cells. -/
theorem tile_low32 (cast : Word32 → Byte)
    (castLeaf : ∀ word, DefinedCast word → cast word = referenceCast word)
    (rest : Algebra Word32) (layout : Layout) (s : Shape) (c : Fin 15) (legal : s.legal c)
    (initial : State Word32) (mpos : 0 < s.m) (npos : 0 < s.n)
    (defined : FP32Domain layout s initial) (bytes : UInt8Domain layout s initial)
    (hout : OutputLayout layout s) :
    ∃ final, FP.Structural.exec (backend cast rest) (source (parameters layout s c)) initial = some final ∧
      (∀ i, active (parameters layout s c) initial i →
        final.mem layout.output (outputOffset (parameters layout s c) initial i) =
          int32Cell (matrixWordTarget layout s initial
            (outputRow (parameters layout s c) initial i) (outputColumn (parameters layout s c) initial i))) ∧
      (∀ r o, (r ≠ layout.output ∨ ∀ i, active (parameters layout s c) initial i →
        o ≠ outputOffset (parameters layout s c) initial i) → final.mem r o = initial.mem r o) := by
  obtain ⟨final, run, post⟩ := correctness cast rest (parameters layout s c) initial
    (all_config_assertions layout s c legal) (admitted_tile_casts layout s c legal initial mpos defined)
    (output_injective_source layout s c initial hout)
  refine ⟨final,run,?_,post.2⟩
  intro i hi
  erw [post.1 i hi]
  rw [active_value cast castLeaf layout s c legal initial mpos npos defined bytes i hi]
  rfl

#axiomsClean output_coordinates_source
#axiomsClean output_injective_source
#axiomsClean active_value
#axiomsClean tile_low32
#specNonCircular matrixWordTarget avoiding [source, outerValues, innerValues, modularAccumulation]
end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked.TileModel
