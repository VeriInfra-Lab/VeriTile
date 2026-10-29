import bench.tritonbench_g.bgmv_expand_slice.Proofs.NativeFP
open VeriTile Triton FP.Structural FP.Guarded
open VeriTile.Bench.TritonBenchG.BgmvExpandSlice
open NativeFP
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000
example {α : Type} (M : Algebra α) (D : Domain α) (dtype : FloatDType) :
    ScalarAdditionCommutes M D none dtype ↔
      ScalarAdditionCommutes M D (some .fp32) dtype := Iff.rfl
example {n : Nat} (dtype output : FloatDType)
    (a b : Op dtype.toTileDType [n]) :
    additionOperands? (.castFloat dtype output
      (.withPrecision dtype .fp64 (.withPrecision dtype .fp32
        (.add dtype.numericDType (.consSame .nil) a b)))) =
      some ⟨dtype, a, b, some .fp32⟩ := by
  cases dtype <;> rfl
#axiomsClean NativeFP.output_first_execution
-- CONFIG fp16/fp16/false/real
#reduce (requirementsFor (original .fp16 .fp16 Bool.false .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/false/fp32
#reduce (requirementsFor (original .fp16 .fp16 Bool.false .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/false/fp16
#reduce (requirementsFor (original .fp16 .fp16 Bool.false .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/false/bf16
#reduce (requirementsFor (original .fp16 .fp16 Bool.false .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/false/f8e4
#reduce (requirementsFor (original .fp16 .fp16 Bool.false .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/false/f8e5
#reduce (requirementsFor (original .fp16 .fp16 Bool.false .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/true/real
#reduce (requirementsFor (original .fp16 .fp16 Bool.true .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/true/fp32
#reduce (requirementsFor (original .fp16 .fp16 Bool.true .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/true/fp16
#reduce (requirementsFor (original .fp16 .fp16 Bool.true .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/true/bf16
#reduce (requirementsFor (original .fp16 .fp16 Bool.true .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/true/f8e4
#reduce (requirementsFor (original .fp16 .fp16 Bool.true .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/fp16/true/f8e5
#reduce (requirementsFor (original .fp16 .fp16 Bool.true .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/false/real
#reduce (requirementsFor (original .fp16 .bf16 Bool.false .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/false/fp32
#reduce (requirementsFor (original .fp16 .bf16 Bool.false .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/false/fp16
#reduce (requirementsFor (original .fp16 .bf16 Bool.false .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/false/bf16
#reduce (requirementsFor (original .fp16 .bf16 Bool.false .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/false/f8e4
#reduce (requirementsFor (original .fp16 .bf16 Bool.false .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/false/f8e5
#reduce (requirementsFor (original .fp16 .bf16 Bool.false .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/true/real
#reduce (requirementsFor (original .fp16 .bf16 Bool.true .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/true/fp32
#reduce (requirementsFor (original .fp16 .bf16 Bool.true .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/true/fp16
#reduce (requirementsFor (original .fp16 .bf16 Bool.true .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/true/bf16
#reduce (requirementsFor (original .fp16 .bf16 Bool.true .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/true/f8e4
#reduce (requirementsFor (original .fp16 .bf16 Bool.true .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp16/bf16/true/f8e5
#reduce (requirementsFor (original .fp16 .bf16 Bool.true .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/false/real
#reduce (requirementsFor (original .bf16 .fp16 Bool.false .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/false/fp32
#reduce (requirementsFor (original .bf16 .fp16 Bool.false .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/false/fp16
#reduce (requirementsFor (original .bf16 .fp16 Bool.false .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/false/bf16
#reduce (requirementsFor (original .bf16 .fp16 Bool.false .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/false/f8e4
#reduce (requirementsFor (original .bf16 .fp16 Bool.false .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/false/f8e5
#reduce (requirementsFor (original .bf16 .fp16 Bool.false .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/true/real
#reduce (requirementsFor (original .bf16 .fp16 Bool.true .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/true/fp32
#reduce (requirementsFor (original .bf16 .fp16 Bool.true .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/true/fp16
#reduce (requirementsFor (original .bf16 .fp16 Bool.true .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/true/bf16
#reduce (requirementsFor (original .bf16 .fp16 Bool.true .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/true/f8e4
#reduce (requirementsFor (original .bf16 .fp16 Bool.true .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/fp16/true/f8e5
#reduce (requirementsFor (original .bf16 .fp16 Bool.true .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/false/real
#reduce (requirementsFor (original .bf16 .bf16 Bool.false .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/false/fp32
#reduce (requirementsFor (original .bf16 .bf16 Bool.false .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/false/fp16
#reduce (requirementsFor (original .bf16 .bf16 Bool.false .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/false/bf16
#reduce (requirementsFor (original .bf16 .bf16 Bool.false .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/false/f8e4
#reduce (requirementsFor (original .bf16 .bf16 Bool.false .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/false/f8e5
#reduce (requirementsFor (original .bf16 .bf16 Bool.false .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/true/real
#reduce (requirementsFor (original .bf16 .bf16 Bool.true .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/true/fp32
#reduce (requirementsFor (original .bf16 .bf16 Bool.true .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/true/fp16
#reduce (requirementsFor (original .bf16 .bf16 Bool.true .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/true/bf16
#reduce (requirementsFor (original .bf16 .bf16 Bool.true .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/true/f8e4
#reduce (requirementsFor (original .bf16 .bf16 Bool.true .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG bf16/bf16/true/f8e5
#reduce (requirementsFor (original .bf16 .bf16 Bool.true .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/false/real
#reduce (requirementsFor (original .fp32 .fp16 Bool.false .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/false/fp32
#reduce (requirementsFor (original .fp32 .fp16 Bool.false .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/false/fp16
#reduce (requirementsFor (original .fp32 .fp16 Bool.false .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/false/bf16
#reduce (requirementsFor (original .fp32 .fp16 Bool.false .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/false/f8e4
#reduce (requirementsFor (original .fp32 .fp16 Bool.false .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/false/f8e5
#reduce (requirementsFor (original .fp32 .fp16 Bool.false .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/true/real
#reduce (requirementsFor (original .fp32 .fp16 Bool.true .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/true/fp32
#reduce (requirementsFor (original .fp32 .fp16 Bool.true .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/true/fp16
#reduce (requirementsFor (original .fp32 .fp16 Bool.true .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/true/bf16
#reduce (requirementsFor (original .fp32 .fp16 Bool.true .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/true/f8e4
#reduce (requirementsFor (original .fp32 .fp16 Bool.true .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/fp16/true/f8e5
#reduce (requirementsFor (original .fp32 .fp16 Bool.true .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/false/real
#reduce (requirementsFor (original .fp32 .bf16 Bool.false .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/false/fp32
#reduce (requirementsFor (original .fp32 .bf16 Bool.false .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/false/fp16
#reduce (requirementsFor (original .fp32 .bf16 Bool.false .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/false/bf16
#reduce (requirementsFor (original .fp32 .bf16 Bool.false .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/false/f8e4
#reduce (requirementsFor (original .fp32 .bf16 Bool.false .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/false/f8e5
#reduce (requirementsFor (original .fp32 .bf16 Bool.false .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/true/real
#reduce (requirementsFor (original .fp32 .bf16 Bool.true .real "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/true/fp32
#reduce (requirementsFor (original .fp32 .bf16 Bool.true .fp32 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/true/fp16
#reduce (requirementsFor (original .fp32 .bf16 Bool.true .fp16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/true/bf16
#reduce (requirementsFor (original .fp32 .bf16 Bool.true .bf16 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/true/f8e4
#reduce (requirementsFor (original .fp32 .bf16 Bool.true .f8e4 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
-- CONFIG fp32/bf16/true/f8e5
#reduce (requirementsFor (original .fp32 .bf16 Bool.true .f8e5 "X" "W" "Y" 1 1 ⟨"I"⟩ 1 1 1 1 1 1 1 0 1 1 1 Bool.false Bool.true)).map Site.profile
