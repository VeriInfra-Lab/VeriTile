import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeRealPrefix
import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_RealInit
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def linearIndex (p : Parameters) (initial : BlockState) := initial.pids 0+p.base

def featureIndex (p : Parameters) (initial : BlockState) := linearIndex p initial % (p.S/p.BS)
def chunkIndex (p : Parameters) (initial : BlockState) := (linearIndex p initial / (p.S/p.BS)) % (p.T/p.BT)
def headIndex (p : Parameters) (initial : BlockState) := linearIndex p initial / ((p.S/p.BS)*(p.T/p.BT))
def tileBase (p : Parameters) (initial : BlockState) :=
  headIndex p initial*p.headStride + chunkIndex p initial*p.BT*p.timeStride +
    featureIndex p initial*p.BS*p.featureStride

theorem active_input_pointer (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.regs .blockPtr [p.BT,p.BS] "p_s"=
      some (⟨fun _=>⟨s,tileBase p initial,[p.BT,p.BS],[p.BT,p.BS],
        [p.timeStride,p.featureStride],[0,0]⟩⟩ : Tile .blockPtr [p.BT,p.BS]) := by
  by_cases h:p.swapped
  all_goals
    simp [-Tile.scalar_eta,activePreludeExecution,h,preludeExecution,Tile.scalar,Tile.bop,Tile.cop,
      Tile.select,headIndex,chunkIndex,featureIndex,linearIndex,tileBase,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      Int.natCast_add,Int.natCast_mul,
      Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg _)]
    simp only [←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv]
    simp only [Int.natCast_nonneg,not_lt.mpr (Int.natCast_nonneg _),↓reduceIte,
      ←Int.emod_def,←Int.natCast_emod,←Int.natCast_add,←Int.natCast_mul,
      Int.toNat_natCast]
    have remCast (a b : Nat) : (a : Int) - (b*(a/b) : Nat) = (a%b : Nat) := by
      rw [Int.natCast_mul,Int.natCast_ediv,←Int.emod_def,←Int.natCast_emod]
    simp only [remCast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
    try simp only [Nat.mul_comm,Nat.mul_assoc,Nat.mul_left_comm]

theorem active_output_pointer (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.regs .blockPtr [p.BT,p.BS] "p_o"=
      some (⟨fun _=>⟨o,tileBase p initial,[p.BT,p.BS],[p.BT,p.BS],
        [p.timeStride,p.featureStride],[0,0]⟩⟩ : Tile .blockPtr [p.BT,p.BS]) := by
  by_cases h:p.swapped <;>
    simpa [activePreludeExecution,h,preludeExecution] using active_input_pointer p o s initial

theorem active_triangular (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.regs .real [p.BT,p.BT] "m_s"=
      some (ChunkCumsumVector.lowerTriTile p.BT) := by
  by_cases h:p.swapped
  all_goals
    simp [-Tile.scalar_eta,activePreludeExecution,h,preludeExecution]
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [Tile.cop,Tile.expandDim,TileShape.dropInsertedIndex,Broadcast.leftIndex,
      Broadcast.rightIndex,ComparableDType.ge,ChunkCumsumVector.lowerTriTile,Tile.select]

theorem active_undef (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.undef=initial.undef := by
  by_cases h:p.swapped <;> simp [activePreludeExecution,h] <;> rfl

theorem active_pids (p : Parameters) (s o : RegionName) (initial : BlockState) :
    (activePreludeExecution p s o initial).val.pids=initial.pids := by
  by_cases h:p.swapped <;> simp [activePreludeExecution,h] <;> rfl

#axiomsClean active_input_pointer
#axiomsClean active_output_pointer
#axiomsClean active_triangular
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource.Real
