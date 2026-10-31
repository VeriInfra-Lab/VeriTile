import bench.tritonbench_g.matmul_tma.Sources.Tensor

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.Native
open VeriTile.Triton FP.Structural FPValue Host TensorABI
noncomputable section
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

inductive Profile where
  | mmaSync | wgmma128 | wgmma64
  deriving DecidableEq, Repr

/-- These are the independently compiled SM90 schedules in the evidence.
This applicability boundary is not an extra Python guard. -/
def ObservedSchedule (h : Shape) (f : Flags) (cfg : Config) : Profile → Prop
  | .mmaSync => h = ⟨16,32,64⟩ ∧ cfg = ⟨1,4⟩
  | .wgmma128 => h = ⟨128,128,128⟩ ∧ cfg = ⟨1,4⟩ ∧ f.transA = Bool.false ∧ f.transB = Bool.false
  | .wgmma64 => h = ⟨128,128,128⟩ ∧ cfg = ⟨4,4⟩ ∧ f.transA = Bool.false ∧ f.transB = Bool.false

abbrev Lanes := Fin 16 → Word
/-- One output lane of a fixed K16 instruction. `first` has no incoming
accumulator (MMA zero seed; WGMMA scaleD=false). Subsequent instructions carry
the live FP32 word. Tensor-core internal arithmetic remains a primitive. -/
structure Instructions where
  first : Profile → Lanes → Lanes → Word
  carry : Profile → Lanes → Lanes → Word → Word
  castHalf : Word → BitVec 16

def cast (I : Instructions) (half : Bool) (w : Word) : Word :=
  if half then ⟨(I.castHalf w).setWidth 32⟩ else w
def lane {n : Nat} (a : Fin n → Word) (group : Nat) (j : Fin 16) : Word :=
  if h : group*16+j.val < n then a ⟨group*16+j.val,h⟩ else FP.Value.zero .fp32
def dot (I : Instructions) (profile : Profile) (n : Nat) (a b : Fin n → Word) : Word :=
  if n/16 = 0 then FP.Value.zero .fp32 else
    (List.range (n/16-1)).foldl
      (fun carry group => I.carry profile (lane a (group+1)) (lane b (group+1)) carry)
      (I.first profile (lane a 0) (lane b 0))

theorem instruction_lane {n : Nat} (a : Fin n → Word) (group : Nat)
    (hg : group < n/16) (j : Fin 16) :
    ∃ bound : group*16+j.val<n, lane a group j = a ⟨group*16+j.val,bound⟩ := by
  have hj := j.isLt
  have hn := Nat.div_mul_le_self n 16
  have hb : group*16+j.val<n := by omega
  exact ⟨hb,by simp only [lane,dif_pos hb]⟩

theorem partition (h : Shape) (cfg : Config) (d : InputContract h cfg) (k : Fin h.inner) :
    k.val/16*16+k.val%16=k.val ∧ k.val%16<16 ∧ k.val/16<h.inner/16 := by
  have hk := k.isLt
  have hm := Nat.mod_eq_zero_of_dvd (inner_multiple h cfg d)
  omega

def algebra (rest : Algebra Word) (I : Instructions) : Algebra Word :=
  {rest with cast := fun p src dst x => match src,dst with
    | .fp32,.fp16 => cast I Bool.true x
    | .fp32,.fp32 => x
    | src,dst => rest.cast p src dst x}

def contraction (I : Instructions) (profile : Profile) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word) :
    TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => dot I profile k (fun j => a (i.1,j,PUnit.unit)) (fun j => b (j,i.2.1,PUnit.unit))
  | _::tail => fun i => contraction I profile (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length
def sourceDot (I : Instructions) (profile : Profile) : Dot Word :=
  ⟨fun _ _ {batch} {m} {k} {n} a b => contraction I profile a b⟩

/-- Independent initial-physical-Tensor expression, containing only the fixed
instruction sequence and the final FP16 encoding or FP32 identity. -/
def expected (I : Instructions) (profile : Profile) (read : Nat → Word) (h : Shape) (f : Flags)
    (a b : TensorABI.Tensor) (i : CIndex (params h f)) : Word :=
  cast I f.half (dot I profile h.inner
    (fun k => logical read 2 a i.1.val k.val)
    (fun k => logical read 2 b k.val i.2.1.val))

theorem output_expected (rest : Algebra Word) (I : Instructions) (profile : Profile)
    (read : Nat → Word) (h : Shape) (f : Flags) (a b c : TensorABI.Tensor)
    (abi : Binding h f a b c) (s : State Word) (input : Inputs read h a b s) :
    outputValues (algebra rest I) (sourceDot I profile) (params h f) f.half s =
      expected I profile read h f a b := by
  have inputs := source_inputs read h f a b c abi s input
  funext i
  cases hf : f.half <;>
    simp only [outputValues,sourceDot,contraction,algebra,expected,outputDType,hf,
      Bool.false_eq_true,↓reduceIte,cast,inputs.1,inputs.2] <;> rfl

/-- The independent reference exposes only fixed instruction and conversion
correspondence; it cannot inspect execution, a whole dot, or final memory. -/
structure Reference where
  first : Profile → Lanes → Lanes → Word
  carry : Profile → Lanes → Lanes → Word → Word
  castHalf : Word → BitVec 16
def referenceInstructions (R : Reference) : Instructions := ⟨R.first,R.carry,R.castHalf⟩
structure Correspondence (I : Instructions) (R : Reference) : Prop where
  first : ∀ p a b, I.first p a b=R.first p a b
  carry : ∀ p a b c, I.carry p a b c=R.carry p a b c
  castHalf : ∀ x, I.castHalf x=R.castHalf x
theorem instructions_eq (I : Instructions) (R : Reference) (law : Correspondence I R) :
    I = referenceInstructions R := by
  cases I; cases R
  simp only [referenceInstructions,Instructions.mk.injEq]
  exact ⟨funext (fun p => funext (fun a => funext (law.first p a))),
    funext (fun p => funext (fun a => funext (fun b => funext (law.carry p a b)))),funext law.castHalf⟩

#specNonCircular expected avoiding [source,FP.Structural.exec,scatter]
#axiomsClean output_expected
#axiomsClean instruction_lane
#axiomsClean partition
#axiomsClean instructions_eq
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.Native
