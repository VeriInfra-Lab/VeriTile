import bench.tritonbench_g.int8_matmul_quantization.Proofs.Reduction
import bench.tritonbench_g.int8_matmul_quantization.Proofs.Writes

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

abbrev QuantCell (G K : Nat) := Fin G × Fin K
abbrev QuantIndex (G K : Nat) := Sum (QuantCell G K) (Fin G)
structure QuantBuffers where
  input : Nat
  output : Nat
  scales : Nat
  inputOffset : Nat
  outputOffset : Nat
  scaleOffset : Nat
  inputGroupStride : Nat
  inputElementStride : Nat
  transposeOutput : Bool
  scalePrecision : Precision
  initialMaximumZero : Bool
structure CompositionLink (q : QuantBuffers) (b : Buffers) : Prop where
  aStorage : b.a=q.output
  aOffset : b.aOffset=q.outputOffset
  scaleStorage : b.aScale=q.scales
  scaleOffset : b.aScaleOffset=q.scaleOffset
  scaleType : b.aScalePrecision=.half
  rowLayout : q.transposeOutput=Bool.false
  rowScaleType : q.scalePrecision=.half
def QuantBuffers.Separate (q : QuantBuffers) : Prop :=
  q.input≠q.output ∧ q.input≠q.scales ∧ q.output≠q.scales
def QuantBuffers.inputAddress (q : QuantBuffers) (g k : Nat) : Address :=
  (q.input,q.inputOffset+g*q.inputGroupStride+k*q.inputElementStride)
def QuantBuffers.dataAddress (q : QuantBuffers) (x : QuantCell G K) : Address :=
  (q.output,q.outputOffset+if q.transposeOutput then x.2.val*G+x.1.val else x.1.val*K+x.2.val)
def QuantBuffers.scaleAddress (q : QuantBuffers) (g : Fin G) : Address :=
  (q.scales,q.scaleOffset+g.val)
def QuantBuffers.writeAddress (q : QuantBuffers) : QuantIndex G K → Address
  | .inl x => q.dataAddress x
  | .inr g => q.scaleAddress g

theorem linearIndex_injective (x y : Fin G × Fin K)
    (equal : x.1.val*K+x.2.val=y.1.val*K+y.2.val) : x=y := by
  have hx := x.2.isLt
  have hy := y.2.isLt
  have positive : 0<K := by omega
  have h := congrArg (fun n => n/K) equal
  simp only [Nat.mul_comm _ K,Nat.mul_add_div positive,Nat.div_eq_of_lt hx,Nat.div_eq_of_lt hy,Nat.add_zero] at h
  apply Prod.ext
  · exact Fin.ext h
  · apply Fin.ext
    rw [h] at equal
    omega
theorem QuantBuffers.data_injective (q : QuantBuffers) : Function.Injective (q.dataAddress (G:=G) (K:=K)) := by
  intro x y equal
  have h := congrArg Prod.snd equal
  by_cases trans : q.transposeOutput
  · simp only [dataAddress,trans,ite_true,Prod.snd] at h
    have swapped : x.swap=y.swap := linearIndex_injective x.swap y.swap
      (by change x.2.val*G+x.1.val=y.2.val*G+y.1.val; omega)
    exact congrArg Prod.swap swapped
  · simp only [dataAddress,trans,Bool.false_eq_true,ite_false,Prod.snd] at h
    exact linearIndex_injective x y (by omega)
theorem QuantBuffers.scale_injective (q : QuantBuffers) : Function.Injective (q.scaleAddress (G:=G)) := by
  intro x y equal
  have h := congrArg Prod.snd equal
  exact Fin.ext (Nat.add_left_cancel h)
theorem QuantBuffers.write_injective (q : QuantBuffers) (sep : q.Separate) :
    Function.Injective (q.writeAddress (G:=G) (K:=K)) := by
  intro x y same
  cases x with
  | inl x =>
    cases y with
    | inl y => exact congrArg Sum.inl (q.data_injective same)
    | inr y => exact False.elim (sep.2.2 (congrArg Prod.fst same))
  | inr x =>
    cases y with
    | inl y => exact False.elim (sep.2.2 (congrArg Prod.fst same).symm)
    | inr y => exact congrArg Sum.inr (q.scale_injective same)

def QuantInputsAgree (q : QuantBuffers) (m before : Memory F H) : Prop :=
  ∀ p,p.1=q.input → m p=before p
def quantScale (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (m : Memory F H) (g : Fin G) : F :=
  let maximum := (plan g).tree.eval a (fun k => read32 a.toOps m (q.inputAddress g k))
  a.scale127 (if q.initialMaximumZero then a.maximum a.fzero maximum else maximum)
def quantValue (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (m : Memory F H) (x : QuantCell G K) : Byte :=
  a.integer8 (a.divide (read32 a.toOps m (q.inputAddress x.1 x.2)) (quantScale a q plan m x.1))
def quantExpected (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (m : Memory F H) : QuantIndex G K → Cell F H
  | .inl x => .inl (quantValue a q plan m x)
  | .inr g => .inr (match q.scalePrecision with
      | .half => .inr (a.narrow (quantScale a q plan m g))
      | .single => .inl (quantScale a q plan m g))

theorem quantScale_current (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (m before : Memory F H) (same : QuantInputsAgree q m before) (g : Fin G) :
    quantScale a q plan m g=quantScale a q plan before g := by
  have values : (fun k : Fin K => read32 a.toOps m (q.inputAddress g k))=
      (fun k : Fin K => read32 a.toOps before (q.inputAddress g k)) := by
    funext k
    unfold read32 Int8DequantMatmul.read32
    rw [same _ rfl]
  unfold quantScale
  rw [values]
theorem quantValue_current (a : Ops I F H) (q : QuantBuffers) (plan : Fin G → ReductionPlan K)
    (m before : Memory F H) (same : QuantInputsAgree q m before) (x : QuantCell G K) :
    quantValue a q plan m x=quantValue a q plan before x := by
  unfold quantValue read32 Int8DequantMatmul.read32
  rw [quantScale_current a q plan m before same,same _ rfl]

theorem quant_put_input (q : QuantBuffers) (sep : q.Separate) (x : QuantIndex G K)
    (value : Cell F H) (m before : Memory F H) (same : QuantInputsAgree q m before) :
    QuantInputsAgree q (put m (q.writeAddress x) value) before := by
  intro p input
  have different : p≠q.writeAddress x := by
    intro equal
    have h := congrArg Prod.fst equal
    cases x with
    | inl x => exact sep.1 (input.symm.trans h)
    | inr g => exact sep.2.1 (input.symm.trans h)
  rw [put,if_neg different,same p input]

#axiomsClean QuantBuffers.write_injective
#axiomsClean quantScale_current
#axiomsClean quantValue_current
#axiomsClean quant_put_input
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization
