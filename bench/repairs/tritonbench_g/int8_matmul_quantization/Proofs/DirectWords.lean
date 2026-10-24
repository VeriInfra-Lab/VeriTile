import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectGemmReal
import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectRow

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section
set_option maxHeartbeats 1500000

structure LogicalGemmWords (F : Type) where
  A : Nat → Nat → Byte
  B : Nat → Nat → Byte
  aScale : Nat → F
  bScale : Nat → F
def originalGemmWords (a : Ops I F H) (r : GemmRequest) (m : Memory F H) : LogicalGemmWords F where
  A := fun row k=>read8 m (logicalAddress r.a (row*r.shape.K+k))
  B := fun k n=>read8 m (logicalAddress r.b (k*r.shape.N+n))
  aScale := fun row=>loadScale a (scalePrecision (some r.aScale.info) Bool.false) m (logicalAddress r.aScale row)
  bScale := fun n=>loadScale a .single m (logicalAddress r.bScale n)
def logicalAccumulation (a : Ops I F H) (s : Shape) (c : Config) (selected : Selection s c)
    (input : LogicalGemmWords F) (row n z : Nat) : Nat → I
  | 0=>a.izero
  | t+1=>(selected.plan row n z t).eval a.toOps
      (fun l=>if sourceK c t z l.val<s.K then input.A row (sourceK c t z l.val) else 0)
      (fun l=>if sourceK c t z l.val<s.K then input.B (sourceK c t z l.val) n else 0)
      (logicalAccumulation a s c selected input row n z t)
def logicalPartial (a : Ops I F H) (s : Shape) (c : Config) (selected : Selection s c)
    (input : LogicalGemmWords F) (row n z : Nat) : H :=
  a.narrow (a.mul (a.mul (a.toFloat (logicalAccumulation a s c selected input row n z (steps s c)))
    (input.aScale row)) (input.bScale n))
def logicalWordReference (a : Ops I F H) (s : Shape) (c : Config) (precision : Precision)
    (selected : Selection s c) (q : Schedule s c) (input : LogicalGemmWords F) (cell : MatrixCell s) : Stored F H :=
  cellFold a c precision (fun z=>logicalPartial a s c selected input cell.1 cell.2 z)
    (projection cell q.events) (outputZero a precision)

theorem gemm_accumulation_original (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (a : Ops I F H) (c : Config) (selected : Selection r.shape c) (m : Memory F H)
    (out : Nat) (cell : MatrixCell r.shape) (z t : Nat) :
    accumulation a r.shape c (r.buffers out) selected m cell.1 cell.2 z t=
      logicalAccumulation a r.shape c selected (originalGemmWords a r m) cell.1 cell.2 z t := by
  have ha : ∀ t,operandA r.shape c (r.buffers out) m cell.1 t z=
      (fun l=>if sourceK c t z l.val<r.shape.K then (originalGemmWords a r m).A cell.1 (sourceK c t z l.val) else 0) := by
    intro t;funext l;unfold operandA;split_ifs with hk
    · rw [gemm_a_address base r laws out _ _ cell.1.isLt hk];rfl
    · rfl
  have hb : ∀ t,operandB r.shape c (r.buffers out) m cell.2 t z=
      (fun l=>if sourceK c t z l.val<r.shape.K then (originalGemmWords a r m).B (sourceK c t z l.val) cell.2 else 0) := by
    intro t;funext l;unfold operandB;split_ifs with hk
    · rw [gemm_b_address base r laws out _ _ hk cell.2.isLt];rfl
    · rfl
  induction t with
  | zero=>rfl
  | succ t ih=>simp only [accumulation,logicalAccumulation,ha,hb,ih]
theorem gemm_partial_original (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (a : Ops I F H) (c : Config) (selected : Selection r.shape c) (m : Memory F H)
    (out : Nat) (cell : MatrixCell r.shape) (z : Nat) :
    partialValue a r.shape c (r.buffers out) selected m cell.1 cell.2 z=
      logicalPartial a r.shape c selected (originalGemmWords a r m) cell.1 cell.2 z := by
  obtain ⟨ha,hb⟩:=gemm_scale_addresses base r laws out cell
  unfold partialValue logicalPartial
  rw [gemm_accumulation_original base r laws,ha,hb]
  rfl
theorem gemm_word_original (base : Nat → Nat) (r : GemmRequest) (laws : GemmLaws base r)
    (a : Ops I F H) (c : Config) (selected : Selection r.shape c) (q : Schedule r.shape c)
    (m : Memory F H) (out : Nat) (cell : MatrixCell r.shape) :
    wordReference a r.shape c (r.buffers out) selected q m cell=
      logicalWordReference a r.shape c (r.buffers out).outPrecision selected q (originalGemmWords a r m) cell := by
  unfold wordReference logicalWordReference
  have h : (fun z : Fin c.S=>partialValue a r.shape c (r.buffers out) selected m cell.1 cell.2 z)=
      (fun z : Fin c.S=>logicalPartial a r.shape c selected (originalGemmWords a r m) cell.1 cell.2 z) := by
    funext z;exact gemm_partial_original base r laws a c selected m out cell z
  rw [h]

def rowMachineScale (a : Ops I F H) (r : RowRequest) (plan : Fin r.G → ReductionPlan r.K)
    (m : Memory F H) (g : Fin r.G) : F :=
  a.scale127 (a.maximum a.fzero ((plan g).tree.eval a (fun k=>originalRow a r m g k)))
def rowMachineValue (a : Ops I F H) (r : RowRequest) (plan : Fin r.G → ReductionPlan r.K)
    (m : Memory F H) (cell : QuantCell r.G r.K) : Byte :=
  a.integer8 (a.divide (originalRow a r m cell.1 cell.2) (rowMachineScale a r plan m cell.1))
def rowMachineExpected (a : Ops I F H) (r : RowRequest) (plan : Fin r.G → ReductionPlan r.K)
    (m : Memory F H) : QuantIndex r.G r.K → Cell F H
  | .inl cell=>.inl (rowMachineValue a r plan m cell)
  | .inr g=>.inr (.inr (a.narrow (rowMachineScale a r plan m g)))
theorem row_machine_scale_original (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (a : Ops I F H) (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory F H) (g : Fin r.G) :
    quantScale a (r.buffers data scales) plan m g=rowMachineScale a r plan m g := by
  have h : (fun k : Fin r.K=>read32 a.toOps m ((r.buffers data scales).inputAddress g k))=
      (fun k : Fin r.K=>originalRow a r m g k) := by
    funext k;rw [row_input_address base r laws data scales g k g.isLt k.isLt];rfl
  unfold quantScale rowMachineScale
  rw [h]
  rfl
theorem row_machine_expected_original (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (a : Ops I F H) (data scales : Nat) (plan : Fin r.G → ReductionPlan r.K) (m : Memory F H) (idx : QuantIndex r.G r.K) :
    quantExpected a (r.buffers data scales) plan m idx=rowMachineExpected a r plan m idx := by
  cases idx with
  | inl cell=>
    simp only [quantExpected,rowMachineExpected,quantValue,rowMachineValue]
    rw [row_machine_scale_original base r laws,row_input_address base r laws data scales cell.1 cell.2 cell.1.isLt cell.2.isLt]
    rfl
  | inr g=>
    simp only [quantExpected,rowMachineExpected,row_machine_scale_original base r laws]
    rfl

#axiomsClean gemm_word_original
#axiomsClean row_machine_expected_original
#specNonCircular logicalWordReference avoiding [runGemm,wrapper,wordReference,partialValue]
#specNonCircular rowMachineExpected avoiding [runRow,quantWrapper,quantExpected,quantScale,quantValue]
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
