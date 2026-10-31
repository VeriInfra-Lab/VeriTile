import bench.tritonbench_g.chunk_gated_attention.Proofs.CumulativeSource
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.ChunkGatedAttention FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

variable {α : Type} [Inhabited α]

def linearIndex (p : Parameters) (initial : State α) := initial.pids 0+p.base

def featureIndex (p : Parameters) (initial : State α) := linearIndex p initial % (p.S/p.BS)
def chunkIndex (p : Parameters) (initial : State α) := (linearIndex p initial / (p.S/p.BS)) % (p.T/p.BT)
def headIndex (p : Parameters) (initial : State α) := linearIndex p initial / ((p.S/p.BS)*(p.T/p.BT))
def tileBase (p : Parameters) (initial : State α) :=
  headIndex p initial*p.headStride + chunkIndex p initial*p.BT*p.timeStride +
    featureIndex p initial*p.BS*p.featureStride

 theorem active_input_pointer (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    (activePreludeExecution model p dtype out_dtype s o initial).val.regs .blockPtr [p.BT,p.BS] "p_s"=
      some (fun _=>⟨s,tileBase p initial,[p.BT,p.BS],[p.BT,p.BS],
        [p.timeStride,p.featureStride],[0,0]⟩) := by
  simp [activePreludeExecution,preludeExecution,FP.Structural.bop,FP.Structural.integralFloorDiv,
    FP.Structural.integralMod,FP.Structural.numericLt,FP.Structural.numericEq,FP.Structural.natLt,
    headIndex,chunkIndex,featureIndex,linearIndex,tileBase,Int.natCast_add,Int.natCast_mul,
    Int.tmod_eq_emod_of_nonneg (Int.natCast_nonneg _)]
  simp only [←Int.natCast_add,←Int.natCast_mul,←Int.natCast_ediv]
  simp only [Int.natCast_nonneg,not_lt.mpr (Int.natCast_nonneg _),↓reduceIte,
    ←Int.emod_def,←Int.natCast_emod,←Int.natCast_add,←Int.natCast_mul,
    Int.toNat_natCast]
  have remCast (a b : Nat) : (a : Int) - (b*(a/b) : Nat) = (a%b : Nat) := by
    rw [Int.natCast_mul,Int.natCast_ediv,←Int.emod_def,←Int.natCast_emod]
  simp only [remCast,←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
  simp only [Nat.mul_comm,Nat.mul_assoc,Nat.mul_left_comm]

theorem active_output_pointer (model : Algebra α) (p : Parameters) (dtype out_dtype : FloatDType)
    (s o : RegionName) (initial : State α) :
    (activePreludeExecution model p dtype out_dtype s o initial).val.regs .blockPtr [p.BT,p.BS] "p_o"=
      some (fun _=>⟨o,tileBase p initial,[p.BT,p.BS],[p.BT,p.BS],
        [p.timeStride,p.featureStride],[0,0]⟩) := by
  exact active_input_pointer model p dtype out_dtype o s initial

#axiomsClean active_input_pointer
#axiomsClean active_output_pointer
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.CumulativeSource
