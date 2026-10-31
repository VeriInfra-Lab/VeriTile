import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealLoopFormula

/-! The diagonal helper's six setup statements. The first helper advances
its local copies; the second resets from the caller's preserved pointer bases.
No state left in the first helper's advanced pointers is used by the second. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetup
open VeriTile.Triton RealLoads RealMemory RealIteration
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def lowerName (second : Bool) : RegName := if second then "second_lo" else "first_lo"
def upperName (second : Bool) : RegName := if second then "second_hi" else "first_hi"
def keyBase (second : Bool) : RegName := if second then "K_ptrs" else "K_ptrs_first"
def scaleBase (second : Bool) : RegName := if second then "K_scale_ptr" else "K_scale_ptr_first"
def valueBase (second : Bool) : RegName := if second then "V_ptrs" else "V_ptrs_first"

def lowerExpr (BM : Nat) : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)
def upperExpr (BM : Nat) : Op .nat [] :=
  .mul .nat .nil (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat BM)

def writes (second : Bool) : List RegName :=
  [lowerName second,upperName second,scaleName second,keyName second,valueName second]

theorem key_shift (s : BlockState) (c : Config) (BN BD start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [BD,BN] ptr = some (keyPointer c BD BN 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BD,BN] ptr)
      (.mul .nat .nil (.constNat c.headDim) (.ref .nat [] low))) s = some (keyPointer c BD BN start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1

theorem value_shift (s : BlockState) (c : Config) (BN BD start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [BN,BD] ptr = some (valuePointer c BN BD 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BN,BD] ptr)
      (.mul .nat .nil (.constNat c.headDim) (.ref .nat [] low))) s = some (valuePointer c BN BD start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1

theorem scale_shift (s : BlockState) (c : Config) (BN start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [] ptr = some (keyScalePointer c BN 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .nil (.ref .ptr [] ptr)
      (.floorDiv .nat .nil (.ref .nat [] low) (.constNat BN))) s = some (keyScalePointer c BN start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [keyScalePointer, Tile.scalar, IntegralDType.floorDiv, -Tile.scalar_eta]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetup
