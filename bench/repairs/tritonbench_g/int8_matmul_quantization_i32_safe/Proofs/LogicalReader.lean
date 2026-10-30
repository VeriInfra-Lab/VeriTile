import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.ProducerIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.GemmIO
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerIOResults
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.GemmIOResults

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.LogicalReader
open VeriTile.Triton FP.Structural KernelExecution
open KernelProducerGrid (Geometry)
open scoped ProgramIO
noncomputable section

/-- Maximum absolute value over exactly the logical columns. -/
def rowMaximum (values : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => max (rowMaximum values n) |values n|


/-- The producer emits signed8 toward-zero codes and max/127 row scales. -/
def producerFormula (p : Geometry) (xs : PublicIO.Producer.Inputs ℝ p) :
    PublicIO.Producer.Outputs ℝ p := fun port => match port with
  | ⟨0,_⟩ => fun j =>
      let values := PublicIO.Producer.values 0 p xs j.1.val
      let x := values j.2.1.val / (rowMaximum values p.columns / 127)
      (BitVec.ofInt 8 (if x<0 then -Int.floor (-x) else Int.floor x)).toInt
  | ⟨1,_⟩ => fun j => rowMaximum (PublicIO.Producer.values 0 p xs j.1.val) p.columns / 127

/-- Full-K dense signed-byte product, then row and column scaling. The scalar
sum is the independent finite mathematical sum `rsum`, bounded by s.K. -/
def gemmFormula (s : Shape) (rowScale output : Precision)
    (xs : PublicIO.Gemm.Inputs ℝ s rowScale) : PublicIO.Gemm.Outputs ℝ s output :=
  fun port => match port with
  | ⟨0,_⟩ => fun j =>
      let input := PublicIO.Gemm.input 0 s rowScale xs
      PublicIO.Gemm.value output
        (Int8DequantMatmul.rsum (fun k => ((input.A j.1.val k).toInt : ℝ) *
          ((input.B k j.2.1.val).toInt : ℝ)) s.K * input.aScale j.1.val * input.bScale j.2.1.val)



/-- Full producer grid: FP32 input matrix to signed INT8 data and FP16 row
scales, with independent quantization formulas, actual source safety and frame. -/
theorem producer_correct (p : Geometry) (selection : Producer.Selection p)
    (covers : p.columns≤p.width) (rest : Algebra ℝ) (order : Producer.Schedule p) :
    Spec.Real (PublicIO.Producer.realIO p selection rest order ⊨ producerFormula p) :=
  by
    have rowMaximum_eq (values : Nat → ℝ) (n : Nat) :
        rowMaximum values n = maximumAbs values n := by
      induction n with
      | zero => rfl
      | succ n ih => simp only [rowMaximum,maximumAbs,ih]
    have target : producerFormula p = PublicIO.Producer.mathematical p := by
      funext xs port
      rcases port with ⟨port,hp⟩
      have small : port<2 := hp
      interval_cases port
      · funext j
        simp only [producerFormula,PublicIO.Producer.mathematical,
          KernelProducerTargets.mathematicalByte,KernelProducerTargets.mathematicalScale,
          realInteger8,rowMaximum_eq]
      · funext j
        simp only [producerFormula,PublicIO.Producer.mathematical,
          KernelProducerTargets.mathematicalScale,rowMaximum_eq]
        rfl
    simpa only [target] using PublicIO.Producer.real_implements p selection covers rest order

/-- Dense scaled integer matrix product for every original scale/output
precision and B layout. The shared IO judgment includes typed zero reset,
complete grouped source invocations, original per-cell split orders, actual
load/store safety, every finite output and the exact exterior frame. -/
theorem gemm_correct (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    Spec.Real ((PublicIO.Gemm.realIO tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells
      (PublicIO.Gemm.SignedInputs s rowScale) (PublicIO.Gemm.initialValues (0:ℝ) s output)
      (gemmFormula s rowScale output)) :=
  PublicIO.Gemm.real_implements tree rest plans rowScale output s c columnMajor q

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.LogicalReader

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.LogicalReader
open VeriTile.Triton FP.Structural KernelExecution
open KernelProducerGrid (Geometry)
open scoped LogicalProgramIO LogicalDomainIO BigOperators
noncomputable section

def logicalProducerFormula (p : Geometry) (xs : Fin p.rows → Fin p.columns → ℝ) : PublicIO.Producer.LogicalOutputs p :=
  { codes := fun row col =>
      let x := xs row col / (rowMaximum (PublicIO.Producer.extendRow xs row) p.columns / 127)
      (BitVec.ofInt 8 (if x<0 then -Int.floor (-x) else Int.floor x)).toInt
    scales := fun row => rowMaximum (PublicIO.Producer.extendRow xs row) p.columns / 127 }

def logicalGemmFormula (s : Shape) (xs : PublicIO.Gemm.LogicalInputs s) (row : Fin s.M) (col : Fin s.N) : ℝ :=
  (∑ k : Fin s.K, (((BitVec.ofInt 8 (xs.a row k)).toInt : ℝ) *
    ((BitVec.ofInt 8 (xs.b k col)).toInt : ℝ))) * xs.rowScale row * xs.columnScale col

theorem rsum_eq_fin (f : Nat → ℝ) (n : Nat) : Int8DequantMatmul.rsum f n = ∑ k : Fin n, f k.val := by
  induction n with
  | zero => simp [Int8DequantMatmul.rsum]
  | succ n ih => simpa only [Int8DequantMatmul.rsum,Fin.sum_univ_castSucc,Fin.coe_castSucc,Fin.val_last] using congrArg (fun x => x+f n) ih

theorem producer_full_iff (p : Geometry) (selection : Producer.Selection p) (rest : Algebra ℝ) (order : Producer.Schedule p) :
    (PublicIO.Producer.logicalIO p selection rest order ⊨ logicalProducerFormula p) ↔
      (PublicIO.Producer.realIO p selection rest order).Implements (producerFormula p) := by
  have target : (PublicIO.Producer.logicalIO p selection rest order).lift (logicalProducerFormula p) = producerFormula p := by
    funext xs o
    have roweq (r : Fin p.rows) : PublicIO.Producer.extendRow (fun r c => xs (0 : Fin 1) (r,c,())) r =
        PublicIO.Producer.values 0 p xs r.val := by
      funext c
      simp only [PublicIO.Producer.extendRow,PublicIO.Producer.values,dif_pos r.isLt]
      rfl
    fin_cases o
    · funext j
      change (logicalProducerFormula p (fun r c => xs (0 : Fin 1) (r,c,()))).codes j.1 j.2.1 = _
      dsimp only [logicalProducerFormula]
      rw [roweq]
      simp only [producerFormula,PublicIO.Producer.values,dif_pos j.1.isLt,dif_pos j.2.1.isLt]
      rfl
    · funext j
      change (logicalProducerFormula p (fun r c => xs (0 : Fin 1) (r,c,()))).scales j.1 = _
      dsimp only [logicalProducerFormula]
      rw [roweq]
      rfl
  change (PublicIO.Producer.realIO p selection rest order).Implements _ ↔ _
  simpa only [target]

theorem producer_with_max (p : Geometry) (selection : Producer.Selection p) (covers : p.columns≤p.width)
    (rest : Algebra ℝ) (order : Producer.Schedule p) (maximum : (Nat → ℝ) → Nat → ℝ)
    (zero : ∀ v, maximum v 0=0)
    (step : ∀ v n, maximum v (n+1)=max (maximum v n) |v n|) :
    PublicIO.Producer.logicalIO p selection rest order ⊨ (fun xs =>
      PublicIO.Producer.LogicalOutputs.mk
        (fun row col =>
          let x := xs row col / (maximum (PublicIO.Producer.extendRow xs row) p.columns / 127)
          (BitVec.ofInt 8 (if x<0 then -Int.floor (-x) else Int.floor x)).toInt)
        (fun row => maximum (PublicIO.Producer.extendRow xs row) p.columns / 127)) := by
  have eq : maximum = rowMaximum := by
    funext v n
    induction n with
    | zero => exact zero v
    | succ n ih => rw [step,rowMaximum,ih]
  subst maximum
  exact (producer_full_iff p selection rest order).mpr (producer_correct p selection covers rest order)

theorem gemm_full_iff (tree : KernelScalarProfile.Tree) (rest : Algebra ℝ) (plans : Plans)
    (rowScale output : Precision) (s : Shape) (c : Config) (columnMajor : Bool) (q : Schedule s c) :
    (PublicIO.Gemm.logicalIO tree rest plans rowScale output s c columnMajor q (PublicIO.Gemm.initialValues (0:ℝ) s output)
      ⊨{PublicIO.Gemm.logicalSignedInputs s rowScale} logicalGemmFormula s) ↔
      (PublicIO.Gemm.realIO tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells
        (PublicIO.Gemm.SignedInputs s rowScale) (PublicIO.Gemm.initialValues (0:ℝ) s output) (gemmFormula s rowScale output) := by
  have domain : (fun raw => PublicIO.Gemm.logicalSignedInputs s rowScale ((PublicIO.Gemm.logicalInputs s rowScale) raw)) =
      PublicIO.Gemm.SignedInputs s rowScale := by
    funext raw
    simp only [PublicIO.Gemm.logicalSignedInputs,Equiv.symm_apply_apply]
  have target : (PublicIO.Gemm.logicalIO tree rest plans rowScale output s c columnMajor q
      (PublicIO.Gemm.initialValues (0:ℝ) s output)).lift (logicalGemmFormula s) = gemmFormula s rowScale output := by
    funext xs o; fin_cases o; funext j
    change PublicIO.Gemm.value output
      (logicalGemmFormula s ((PublicIO.Gemm.logicalInputs s rowScale) xs) j.1 j.2.1) = _
    have inputA (k : Fin s.K) : (PublicIO.Gemm.input 0 s rowScale xs).A j.1 k =
        BitVec.ofInt 8 (((PublicIO.Gemm.logicalInputs s rowScale) xs).a j.1 k) := by
      simp only [PublicIO.Gemm.input,dif_pos j.1.isLt,dif_pos k.isLt]; rfl
    have inputB (k : Fin s.K) : (PublicIO.Gemm.input 0 s rowScale xs).B k j.2.1 =
        BitVec.ofInt 8 (((PublicIO.Gemm.logicalInputs s rowScale) xs).b k j.2.1) := by
      simp only [PublicIO.Gemm.input,dif_pos j.2.1.isLt,dif_pos k.isLt]; rfl
    have scaleA : (PublicIO.Gemm.input 0 s rowScale xs).aScale j.1 =
        ((PublicIO.Gemm.logicalInputs s rowScale) xs).rowScale j.1 := by
      simp only [PublicIO.Gemm.input,dif_pos j.1.isLt]; rfl
    have scaleB : (PublicIO.Gemm.input 0 s rowScale xs).bScale j.2.1 =
        ((PublicIO.Gemm.logicalInputs s rowScale) xs).columnScale j.2.1 := by
      simp only [PublicIO.Gemm.input,dif_pos j.2.1.isLt]; rfl
    simp only [gemmFormula,rsum_eq_fin,inputA,inputB,scaleA,scaleB,logicalGemmFormula]
  change (PublicIO.Gemm.realIO tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells
    (fun raw => PublicIO.Gemm.logicalSignedInputs s rowScale ((PublicIO.Gemm.logicalInputs s rowScale) raw))
    (PublicIO.Gemm.initialValues (0:ℝ) s output)
    ((PublicIO.Gemm.logicalIO tree rest plans rowScale output s c columnMajor q (PublicIO.Gemm.initialValues (0:ℝ) s output)).lift (logicalGemmFormula s)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun d f => (PublicIO.Gemm.realIO tree rest plans rowScale output s c columnMajor q).ImplementsResetOnCells d (PublicIO.Gemm.initialValues (0:ℝ) s output) f) domain target)

end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.KernelOnly.LogicalReader
