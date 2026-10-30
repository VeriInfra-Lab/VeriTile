import bench.tritonbench_g.uniform_sampling.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.UniformSampling.KernelMath
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.UniformSampling
noncomputable section

/-- Mathematical projection of an ABI word. Exceptional words have no finite
Real denotation; their operational behavior remains covered by the FP theorem. -/
def argument (word : Float32Bits) : ℝ :=
  (((FP.Value.mk word.bits : FP.Value .fp32).toRat?).getD 0 : ℝ)

/-- Ordinary finite mathematical denotation of an extracted scalar argument.
This interpretation domain does not add a Python input guard. -/
def Denotes (word : Float32Bits) (value : ℝ) : Prop :=
  ((FP.Value.mk word.bits : FP.Value .fp32).toRat?).map (fun q => (q : ℝ)) = some value

theorem argument_eq (word : Float32Bits) (value : ℝ) (h : Denotes word value) :
    argument word = value := by
  unfold Denotes at h
  unfold argument
  cases hr : (FP.Value.mk word.bits : FP.Value .fp32).toRat? with
  | none => simp [hr] at h
  | some q => simpa [hr] using h

/-- Independent input-only mathematical formula. Philox10 and the canonical
rounded uniform decoder are retained; only the surrounding arithmetic is Real. -/
def expected (seed offset : Int) (lower upper : ℝ) (pid B : Nat) (f : Fin 4) (i : Fin B) : ℝ :=
  lower + Random.uniformReal (ScalarSpec.word seed offset (pid*B+i.val) (KernelExecution.fieldOf f)) * (upper-lower)

def realAlgebra : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op x y => match op with
    | .add => x+y | .sub => x-y | .mul => x*y | .div => x/y | _ => 0
  fma := fun _ => some (fun a b c => a*b+c)
  unary := fun _ _ x => x
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := argument
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ => fun _ => 0
  reduceSum := fun _ {_} _ _ _ => fun _ => 0

theorem scalar_real (seed offset : Int) (lower upper : Float32Bits)
    (B pid : Nat) (s : State ℝ) (f : Fin 4) (i : Fin B) :
    KernelExecution.values realAlgebra (fun a b c => a*b+c) seed offset lower upper B
      (KernelExecution.entry pid s) f (i,PUnit.unit) =
        expected seed offset (argument lower) (argument upper) pid B f i := by
  have random (w : Random.Word) : argument ⟨(Random.uniform w).bits⟩ = Random.uniformReal w := by
    change (((Random.uniform w).toRat?).getD 0 : ℝ) = _
    rw [Random.uniform_toRat_uniformRat]
    rfl
  simp [KernelExecution.values,KernelAffine.values,KernelExecution.words,KernelPrefix.indices,
    KernelExecution.entry,realAlgebra,random,expected,add_comm]

#specNonCircular expected avoiding [kernel,indexCommutedKernel,KernelExecution.result]
#axiomsClean scalar_real
#axiomsClean argument_eq
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.UniformSampling.KernelMath
