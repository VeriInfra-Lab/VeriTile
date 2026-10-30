import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
open VeriTile.Triton FP.Structural KernelExecution
open KernelProducerGrid
noncomputable section

theorem correct (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (input : Gemm.Input ℝ)
    (q : Int8MatmulQuantization.Schedule s c) (initial : State ℝ)
    (reads : Gemm.Reads (Gemm.parameters s c columnMajor) rowScale input initial) :
    KernelInterface.RealResult tree rest plans rowScale output s c columnMajor input q initial :=
  KernelInterface.real tree rest plans rowScale output s c columnMajor input q initial reads

theorem producerCorrect (p : Geometry) (selection : Producer.Selection p) (covers : p.columns ≤ p.width)
    (rest : Algebra ℝ) (values : Nat → Nat → ℝ) (s : State ℝ)
    (reads : KernelProducerTargets.Reads p values s) (order : Producer.Schedule p) :
    ∃ final, KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8 p order.order s = some final ∧
      (∀ r : Fin p.rows, ∀ c : Fin p.width, c.val < p.columns →
        final.mem "q" (r.val*p.columns+c.val) =
          .mk .int (KernelProducerTargets.mathematicalByte (values r) p.columns c).toInt) ∧
      (∀ r : Fin p.rows, final.mem "scales" r =
        .mk .fp16 (KernelProducerTargets.mathematicalScale (values r) p.columns)) ∧
      (∀ (region : RegionName) offset, (region ≠ "q" ∨ p.rows*p.columns ≤ offset) →
        (region ≠ "scales" ∨ p.rows ≤ offset) → final.mem region offset = s.mem region offset) :=
  Producer.mathematical_complete p selection covers rest values s reads order

theorem composedCorrect (s : Shape) (c : Config) (columnMajor : Bool) (width : Nat)
    (positive : 0<width) (covers : s.K≤width) (selection : Producer.Selection (Composed.geometry s width positive))
    (rest : Algebra ℝ) (plans : Plans) (output : Precision) (values : Nat → Nat → ℝ)
    (B : Nat → Nat → Byte) (bs : Nat → ℝ) (state : State ℝ)
    (reads : KernelProducerTargets.Reads (Composed.geometry s width positive) values state)
    (weights : Composed.WeightReads s c columnMajor B bs state)
    (order : Producer.Schedule (Composed.geometry s width positive)) (schedule : Int8MatmulQuantization.Schedule s c) :
    ∃ produced,
      KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
        (Composed.geometry s width positive) order.order state=some produced ∧
      (∀ r : Fin s.M, ∀ k : Fin width, k.val<s.K → produced.mem "q" (r.val*s.K+k.val)=
        .mk .int (KernelProducerTargets.mathematicalByte (values r) s.K k).toInt) ∧
      (∀ r : Fin s.M, produced.mem "scales" r=.mk .fp16 (KernelProducerTargets.mathematicalScale (values r) s.K)) ∧
      (∀ (region : RegionName) offset, (region≠"q" ∨ s.M*s.K≤offset) →
        (region≠"scales" ∨ s.M≤offset) → produced.mem region offset=state.mem region offset) ∧
      KernelInterface.RealResult selection.tree rest plans .half output s c columnMajor
        (Composed.realInput s values B bs) schedule (Composed.attach produced) := by
  obtain ⟨produced,run,data,scales,frame⟩ := producerCorrect (Composed.geometry s width positive)
    selection covers rest values state reads order
  have bound : Gemm.Reads (Gemm.parameters s c columnMajor) .half
      (Composed.realInput s values B bs) (Composed.attach produced) :=
    Composed.reads_produced s c columnMajor width covers state produced _ _ _ _ data scales weights
      (fun region i hq hs => frame region i (Or.inl hq) (Or.inl hs))
  exact ⟨produced,run,data,scales,frame,correct selection.tree rest plans .half output s c columnMajor
    (Composed.realInput s values B bs) schedule (Composed.attach produced) bound⟩

#axiomsClean correct
#axiomsClean producerCorrect
#axiomsClean composedCorrect
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
