import bench.tritonbench_g.matmul_dequantize_int4.SoftwareCorrect
import bench.tritonbench_g.matmul_dequantize_int4.CompiledSoftwareCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- An actual original autotune member, two complete K blocks and one full
64x128 output tile. NO_GROUPS agrees with groupsize=K. -/
def parameters (mode : Bool) : Parameters :=
  ⟨"A","C","S","B","Z",64,128,64,64,1,128,1,128,1,128,1,16,1,
    (if mode then 64 else 32),64,128,32,8,mode⟩

def halfWord (n : Nat) : Word := QuantizeKvTransform.MixedBits.embed
  (MatmulDequantInt4.Binary16.Value.ofBits n)

def initial : State Word where
  mem := fun r off => if r=("B" : RegionName) then .mk .int (0x33333333 : BitVec 32).toInt
    else if r=("Z" : RegionName) then .mk .int (0x22222222 : BitVec 32).toInt
    else if r=("A" : RegionName) then .mk .fp16 (halfWord 0x3c00)
    else if r=("S" : RegionName) then .mk .fp16 (halfWord (if off<128 then 0x3c01 else 0x3c00))
    else .mk .fp16 (halfWord 0x7e55)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def scaleBits (mode : Bool) (step : Nat) : Nat := if mode || step=0 then 0x3c01 else 0x3c00

theorem packed_nibble_three (j : Nat) : MatmulDequantInt4.FPTyped.nibble
    (0x33333333 : BitVec 32).toInt j = 3 := by
  unfold MatmulDequantInt4.FPTyped.nibble
  have h : j%8<8 := Nat.mod_lt _ (by decide)
  generalize hx : j%8=n at *
  interval_cases n <;> decide +kernel

theorem packed_nibble_two (j : Nat) : MatmulDequantInt4.FPTyped.nibble
    (0x22222222 : BitVec 32).toInt j = 2 := by
  unfold MatmulDequantInt4.FPTyped.nibble
  have h : j%8<8 := Nat.mod_lt _ (by decide)
  generalize hx : j%8=n at *
  interval_cases n <;> decide +kernel

theorem packed_inputs (mode : Bool) : PackedInputs (parameters mode) initial := by
  intro step hs i
  exact ⟨⟨0x33333333,by simp [initial,parameters]⟩,⟨0x22222222,by simp [initial,parameters]⟩⟩

theorem weight_uniform (mode : Bool) (step : Nat) (i : BIndex (parameters mode)) :
    weightNibble (parameters mode) initial step i = 3 := by
  simp [weightNibble,parameters,initial]
  exact packed_nibble_three _

theorem zero_uniform (mode : Bool) (step : Nat) (j : TileIndex [(parameters mode).blockColumns]) :
    zeroNibble (parameters mode) initial step j = 2 := by
  simp [zeroNibble,parameters,initial]
  exact packed_nibble_two _

theorem scale_uniform (mode : Bool) (step : Nat) (hs : step<2)
    (j : TileIndex [(parameters mode).blockColumns]) :
    scaleValue (parameters mode) initial step j = halfWord (scaleBits mode step) := by
  have hj : j.1.val<128 := j.1.isLt
  cases mode <;> interval_cases step <;>
    simp [scaleValue,scaleOffset,inputColumn,columnBlock,rowsInGroup,firstRow,blocksM,blocksN,
      groupIndex,parameters,initial,scaleBits,show j.1.val<128 from hj,
      show ¬j.1.val+128<128 from by omega]

theorem a_uniform (mode : Bool) (step : Nat) (i : AIndex (parameters mode)) :
    FPSoftware.machineA (parameters mode) initial step i = halfWord 0x3c00 := by
  have hi : i.1.val<64 := i.1.isLt
  simp [FPSoftware.machineA,inputRow,rowBlock,rowsInGroup,firstRow,blocksM,blocksN,parameters,initial,hi]

#axiomsClean packed_nibble_three
#axiomsClean packed_nibble_two
#axiomsClean packed_inputs
#axiomsClean weight_uniform
#axiomsClean zero_uniform
#axiomsClean scale_uniform
#axiomsClean a_uniform
end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.Characterization
