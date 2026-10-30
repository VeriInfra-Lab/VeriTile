import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.NativeScalar
import bench.repairs.tritonbench_g.sgmv_expand_slice.Proofs.SourceMemory

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 20000

@[simp] theorem pack_zero (f : Native.Format) : pack f (0 : Native.Word f) = 0 := by
  cases f <;> rfl

def inputWords (abi : SourceABI) (input : Native.Inputs (profile abi)) : Nat → Nat → Raw :=
  fun r k => pack (scalar abi.input) (input.x r k)
def weightWords (abi : SourceABI) (input : Native.Inputs (profile abi)) : Nat → Nat → Nat → Raw :=
  fun a c k => pack (weight abi.weight).format (input.weight a c k)

@[simp] theorem scalar_weight (f : LowFormat) : scalar f.scalar = (weight f).format := by cases f <;> rfl

theorem a_matches (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (s : State Raw)
    (memory : SourceMemory.Inputs abi p shape s (inputWords abi input) (weightWords abi input))
    (fits : SegmentFits shape segment) (live : segment.active) (pid block : Nat)
    (hb : block < (p.inner+15)/16) (i : TileIndex [32,16]) :
    SourceBody.aValues (algebra rest ops (profile abi)) abi p
      (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s block i =
      pack (weight abi.weight).format
        (Native.aTile ops (profile abi) shape segment input (pid/tiles p.columns) block i.1 i.2.1) := by
  rw [SourceMemory.a_values _ _ _ _ _ _ _ memory _
    (SourceMemory.coordinates_rows p shape segment fits live pid) block hb i]
  rw [typed_zero rest ops (profile abi) zeros abi.input]
  simp only [inputWords]
  by_cases hk : block*16+i.2.1.val < shape.inner
  all_goals simp only [hk,if_true,if_false]
  all_goals rw [cast_input]
  all_goals simp only [Native.aTile,profile,SourceInit.coordinates,SourceSetup.rowIndex,
    Nat.add_comm i.1.val,hk,if_true,if_false]
  all_goals rfl

theorem b_matches (rest : Algebra Raw) (ops : Native.Primitives) (zeros : ZeroConversions ops)
    (abi : SourceABI) (p : KernelArgs) (shape : Shape) (segment : Segment)
    (input : Native.Inputs (profile abi)) (s : State Raw)
    (memory : SourceMemory.Inputs abi p shape s (inputWords abi input) (weightWords abi input))
    (fits : SegmentFits shape segment) (live : segment.active) (positive : 0 < shape.columns)
    (pid block : Nat) (hb : block < (p.inner+15)/16) (i : TileIndex [16,32]) :
    SourceBody.bValues (algebra rest ops (profile abi)) abi p
      (SourceInit.coordinates p segment.start (segment.adapter.getD 0) segment.length pid) s block i =
      pack (weight abi.weight).format
        (Native.bTile (profile abi) shape segment input (pid%tiles p.columns) block i.1 i.2.1) := by
  rw [SourceMemory.b_values _ _ _ _ _ _ _ memory _
    (SourceMemory.coordinates_adapter shape segment fits live)
    (SourceMemory.coordinates_columns p shape segment memory.columns positive pid) block hb i]
  rw [typed_zero rest ops (profile abi) zeros abi.weight.scalar]
  simp only [weightWords,scalar_weight,Native.bTile,profile,SourceInit.coordinates,
    SourceSetup.columnIndex,memory.columns,Nat.add_comm i.2.1.val]
  split_ifs
  · rfl
  · change pack (scalar abi.weight.scalar) (0 : Native.Word (scalar abi.weight.scalar)) =
      pack (weight abi.weight).format (0 : Native.Word (weight abi.weight).format)
    rw [pack_zero,pack_zero]

#axiomsClean a_matches
#axiomsClean b_matches
end
end VeriTile.Bench.Repairs.SgmvExpandSlice.NativeBackend
