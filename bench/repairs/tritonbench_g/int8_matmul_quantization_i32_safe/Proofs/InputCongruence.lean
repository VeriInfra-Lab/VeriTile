import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.Endpoints

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
open VeriTile.Triton FP.Structural KernelExecution
noncomputable section
set_option maxHeartbeats 6000000
variable {I F H : Type} [Inhabited F]

structure Agree (s : Shape) (r n : Nat) (left right : Input F) : Prop where
  A : ∀ k, k<s.K → left.A r k=right.A r k
  B : ∀ k, k<s.K → left.B k n=right.B k n
  aScale : left.aScale r=right.aScale r
  bScale : left.bScale n=right.bScale n

theorem accumulator_congr (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (s : Shape) (c : Config) (columnMajor : Bool) (left right : Input F)
    (r n z t : Nat) (same : Agree s r n left right) :
    accumulator carrier ops plans (parameters s c columnMajor) left r n z t=
      accumulator carrier ops plans (parameters s c columnMajor) right r n z t := by
  have ha : ∀ t, (fun lane : Fin c.BK => if inner (parameters s c columnMajor) t z lane<s.K then
      left.A r (inner (parameters s c columnMajor) t z lane) else 0) =
      (fun lane : Fin c.BK => if inner (parameters s c columnMajor) t z lane<s.K then
      right.A r (inner (parameters s c columnMajor) t z lane) else 0) := by
    intro t; funext lane; split_ifs with h
    · exact same.A _ h
    · rfl
  have hb : ∀ t, (fun lane : Fin c.BK => if inner (parameters s c columnMajor) t z lane<s.K then
      left.B (inner (parameters s c columnMajor) t z lane) n else 0) =
      (fun lane : Fin c.BK => if inner (parameters s c columnMajor) t z lane<s.K then
      right.B (inner (parameters s c columnMajor) t z lane) n else 0) := by
    intro t; funext lane; split_ifs with h
    · exact same.B _ h
    · rfl
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [accumulator, ih]
    simp only [parameters] at ha hb ⊢
    rw [ha,hb]

theorem partial_congr (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (rowScale : Precision) (s : Shape) (c : Config) (columnMajor : Bool)
    (left right : Input F) (r n z : Nat) (same : Agree s r n left right) :
    partialWord M carrier ops plans rowScale (parameters s c columnMajor) left r n z=
      partialWord M carrier ops plans rowScale (parameters s c columnMajor) right r n z := by
  unfold partialWord
  rw [accumulator_congr carrier ops plans s c columnMajor left right r n z _ same, same.bScale]
  cases rowScale <;> simp [rowWord, same.aScale]

theorem cell_fold_congr (M : Algebra F) (carrier : IntegerCarrier I) (ops : Int8DequantMatmul.Ops I F H)
    (plans : Plans) (atomicHalf atomicSingle : F → F → F) (rowScale output : Precision)
    (s : Shape) (c : Config) (columnMajor : Bool) (left right : Input F) (cell : MatrixCell s)
    (same : Agree s cell.1 cell.2 left right) (zs : List (Fin c.S)) (old : FP.Structural.Cell F) :
    zs.foldl (fun old z => Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor left (cell,z) old) old =
      zs.foldl (fun old z => Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor right (cell,z) old) old := by
  have act : ∀ z old, Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor left (cell,z) old =
      Grid.action M carrier ops plans atomicHalf atomicSingle rowScale output s c columnMajor right (cell,z) old := by
    intro z old
    unfold Grid.action
    rw [partial_congr M carrier ops plans rowScale s c columnMajor left right cell.1 cell.2 z same]
  induction zs generalizing old with
  | nil => rfl
  | cons z zs ih => rw [List.foldl_cons, act, ih]; rfl

#axiomsClean accumulator_congr
#axiomsClean partial_congr
#axiomsClean cell_fold_congr
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Gemm
