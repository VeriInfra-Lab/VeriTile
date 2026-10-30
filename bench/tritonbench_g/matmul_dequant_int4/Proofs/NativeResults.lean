import bench.tritonbench_g.matmul_dequant_int4.Proofs.KernelInterface
import bench.tritonbench_g.matmul_dequant_int4.Spec

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Native
open VeriTile.Triton FP.Structural
open MatmulDequantInt4.Direct
noncomputable section
abbrev Word := BitVec 32
abbrev Instructions := MatmulDequantInt4.Native.Instructions

/-- Width-preserving native operations. The structural quotation materializes
an exact integer promotion through FP32 before the selected-width cast. -/
def scalar (I : Instructions) : Algebra Word where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ d op x y => if op=.mul then
    match d with
    | .fp16 => (I.mul16 (x.truncate 16) (y.truncate 16)).zeroExtend 32
    | .fp32 => I.mul32 x y
    | _ => 0
    else 0
  unary := fun _ _ x => x
  cast := fun _ _ dst x => if dst=.fp16 then (I.narrow x).zeroExtend 32 else x
  fromNat := fun _ n => I.convert32 .signed n
  fromInt := fun _ n => I.convert32 .signed n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ _ => 0

/-- A local conversion port on the 31 possible signed nibble differences. It
contains no dimensions, loads, dot products, execution or output hypotheses.
The existing concrete FP16/FP32 conversion-exactness proofs justify the finite
promotion path for matching native primitive implementations. -/
def Conversion (I : Instructions) : Prop :=
  ∀ n : Int,-15≤n → n≤15 → I.narrow (I.convert32 .signed n)=I.convert16 .signed n

/-- Independent selected-width native target over raw packed/input cells. -/
def kernel_target (f : Source.Profile) (I : Instructions) (p : Source.Parameters)
    (initial : State Word) (x : KernelGrid.Coordinate p) : Word :=
  match f with
  | .half => (I.mul16 (I.convert16 .signed (KernelInterface.difference p initial x))
      ((KernelInterface.scale f p initial x).truncate 16)).zeroExtend 32
  | .single => I.mul32 (I.convert32 .signed (KernelInterface.difference p initial x))
      (KernelInterface.scale f p initial x)

theorem kernel_result_eq (f : Source.Profile) (I : Instructions)
    (conversion : f=.half → Conversion I) (p : Source.Parameters) (initial : State Word)
    (x : KernelGrid.Coordinate p) :
    KernelGrid.result f (scalar I) p initial x=kernel_target f I p initial x := by
  have bounds := KernelInterface.difference_bounds p initial x
  cases f with
  | half =>
    have converted := conversion rfl _ bounds.1 bounds.2
    dsimp only [KernelInterface.difference] at converted
    simp [KernelGrid.result, scalar, kernel_target, Source.Profile.dtype,
      KernelInterface.difference, KernelInterface.scale, converted]
  | single => rfl

theorem kernel_native (f : Source.Profile) (I : Instructions)
    (conversion : f=.half → Conversion I) (p : Source.Parameters)
    (valid : KernelGrid.Valid p) (initial : State Word) :
    ∃ final,KernelGrid.run f (scalar I) p initial=some final ∧
      (∀ x : KernelGrid.Coordinate p,
        final.mem p.output (KernelGrid.address p x)=Source.cell f (kernel_target f I p initial x)) ∧
      (∀ r o,(r≠p.output ∨ ∀ x : KernelGrid.Coordinate p,o≠KernelGrid.address p x) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := KernelGrid.whole f (scalar I) p valid initial
  refine ⟨final,ran,?_,?_⟩
  · intro x
    rw [values x trivial]
    exact congrArg (Source.cell f) (kernel_result_eq f I conversion p initial x)
  · intro r o outside
    exact frame r o (outside.imp_right (fun h x _ => h x))

#specNonCircular kernel_target avoiding [KernelGrid.run,Source.source,Source.execution,scalar,KernelGrid.result]
#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Native
