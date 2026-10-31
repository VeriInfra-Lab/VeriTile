import bench.tritonbench_g.chunk_gated_attention.Proofs.HRealProgram
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention
open VeriTile.Bench.TritonBenchG.ChunkGatedAttention
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 600000
set_option linter.unusedSimpArgs false

def linearIndex (p : Parameters) (initial : BlockState) := initial.pids 0+p.base
def valueIndex (p : Parameters) (initial : BlockState) := linearIndex p initial % (p.V/p.BV)
def keyIndex (p : Parameters) (initial : BlockState) :=
  (linearIndex p initial / (p.V/p.BV)) % (p.K/p.BK)
def headIndex (p : Parameters) (initial : BlockState) :=
  linearIndex p initial / ((p.V/p.BV)*(p.K/p.BK))
def decoded (p : Parameters) (initial : BlockState) : RealHMath.Launch :=
  ⟨valueIndex p initial,keyIndex p initial,headIndex p initial⟩

private theorem remCast (a b : Nat) : (a : Int) - (b*(a/b) : Nat) = (a%b : Nat) := by
  rw [Int.natCast_mul,Int.natCast_ediv,←Int.emod_def,←Int.natCast_emod]

local macro "normalize_source_index" : tactic => `(tactic|
  (simp [-Tile.scalar_eta,preludeExecution,Tile.scalar,Tile.bop,Tile.cop,Tile.select,
    NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
    ComparableDType.lt,linearIndex,valueIndex,keyIndex,headIndex,Int.natCast_add,Int.natCast_mul]
   simp only [←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv]
   simp only [Int.natCast_nonneg,not_lt.mpr (Int.natCast_nonneg _),↓reduceIte,
    ←Int.emod_def,←Int.natCast_emod,←Int.natCast_add,←Int.natCast_mul,
    remCast,Int.toNat_natCast]
   try rfl))

theorem prelude_value (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    (preludeExecution p r initial).val.regs .int [] "i_v"=
      some (Tile.scalar (valueIndex p initial : Int)) := by
  normalize_source_index

theorem prelude_key (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    (preludeExecution p r initial).val.regs .int [] "i_k"=
      some (Tile.scalar (keyIndex p initial : Int)) := by
  normalize_source_index

theorem prelude_head (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    (preludeExecution p r initial).val.regs .int [] "i_bh"=
      some (Tile.scalar (headIndex p initial : Int)) := by
  normalize_source_index

def StrideRegs (p : Parameters) (u : BlockState) : Prop :=
  u.regs .int [] "s_k_h"=some (Tile.scalar (p.keyHead : Int)) ∧
  u.regs .int [] "s_k_t"=some (Tile.scalar (p.keyTime : Int)) ∧
  u.regs .int [] "s_k_d"=some (Tile.scalar (p.keyFeature : Int)) ∧
  u.regs .int [] "s_v_h"=some (Tile.scalar (p.valueHead : Int)) ∧
  u.regs .int [] "s_v_t"=some (Tile.scalar (p.valueTime : Int)) ∧
  u.regs .int [] "s_v_d"=some (Tile.scalar (p.valueFeature : Int)) ∧
  u.regs .int [] "s_h_h"=some (Tile.scalar (p.historyHead : Int)) ∧
  u.regs .int [] "s_h_t"=some (Tile.scalar (p.historyKey : Int)) ∧
  u.regs .int [] "s_h_d"=some (Tile.scalar (p.historyValue : Int))

theorem prelude_strides (p : Parameters) (r : RealHProgram.Regions) (initial : BlockState) :
    StrideRegs p (preludeExecution p r initial).val := by
  simp [StrideRegs,preludeExecution]

#axiomsClean prelude_value
#axiomsClean prelude_key
#axiomsClean prelude_head
#axiomsClean prelude_strides
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.HSource.Real
