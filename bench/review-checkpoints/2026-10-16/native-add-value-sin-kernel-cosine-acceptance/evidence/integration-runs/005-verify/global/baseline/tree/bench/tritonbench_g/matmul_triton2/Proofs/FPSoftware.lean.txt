import bench.tritonbench_g.matmul_triton2.Proofs.FPComplete
namespace VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
open VeriTile.Triton FP.Structural FP.Equational
abbrev Word := FP.Value .fp32
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Explicit software input projections. These are not inferred from the
source's unspecified dot options. Choosing the native policy requires a target,
compiler version, fp32_default knob and conversion-instruction binding. -/
inductive InputMode where
  | fullFP32 | tf32NearestEven | tf32NearestAway
  deriving DecidableEq, Repr

/-- Quantize a binary32 magnitude to the grid with thirteen trailing fraction
bits zero. Ties select an even retained significand or the larger magnitude,
as explicitly requested. Preserve infinities, zero signs and NaN payloads;
subnormal flushing and later arithmetic NaN policy belong to cfg/FP.mul. -/
def roundedMagnitude (tiesEven : Bool) (magnitude : Nat) : Nat :=
  let retained := magnitude/8192
  let discarded := magnitude%8192
  let up := discarded>4096 ∨ (discarded=4096 ∧ (tiesEven=Bool.false ∨ retained%2=1))
  (retained+if up then 1 else 0)*8192

def tf32Projection (tiesEven : Bool) (x : Word) : Word :=
  if x.exponent=255 then x else
    FP.Value.ofBits .fp32 ((if x.sign then 2147483648 else 0)+
      roundedMagnitude tiesEven (x.raw % 2147483648))

/-- The retained magnitude is a multiple of the TF32 quantum and is at most
half a quantum from the original encoded magnitude. The definition resolves
exact midpoints with the explicitly selected even/away rule. -/
theorem roundedMagnitude_nearest (tiesEven : Bool) (magnitude : Nat) :
    roundedMagnitude tiesEven magnitude % 8192=0 ∧
    roundedMagnitude tiesEven magnitude ≤ magnitude+4096 ∧
    magnitude ≤ roundedMagnitude tiesEven magnitude+4096 := by
  have hdiv := Nat.mod_add_div magnitude 8192
  have hmod := Nat.mod_lt magnitude (by decide : 0<8192)
  unfold roundedMagnitude
  dsimp only
  split_ifs with hu <;> simp only [Nat.mul_mod_left, Nat.add_zero]
  all_goals refine ⟨trivial,?_,?_⟩
  all_goals omega

/-- Every finite input is projected onto the 10-fraction-bit TF32 grid;
overflow may become infinity. NaNs are handled separately and retain payload. -/
theorem tf32_grid (tiesEven : Bool) (x : Word) (hx : x.exponent≠255) :
    (tf32Projection tiesEven x).raw % 8192=0 := by
  simp only [tf32Projection,if_neg hx,FP.Value.raw,FP.Value.ofBits,BitVec.toNat_ofNat]
  change (((if x.sign then 2147483648 else 0)+roundedMagnitude tiesEven (x.raw%2147483648))%
    4294967296)%8192=0
  rw [Nat.mod_mod_of_dvd _ (by decide : 8192 ∣ 4294967296),Nat.add_mod]
  rw [(roundedMagnitude_nearest tiesEven (x.raw%2147483648)).1]
  split_ifs <;> decide

def inputProjection : InputMode → Word → Word
  | .fullFP32 => id
  | .tf32NearestEven => tf32Projection Bool.true
  | .tf32NearestAway => tf32Projection Bool.false

def treeValue (cfg : FP.Config) (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => FP.Value.zero .fp32
  | .add a b => FP.add cfg (treeValue cfg xs a) (treeValue cfg xs b)

/-- The explicit input callback may be full fp32, one of the concrete TF32
software rules above, or a separately bound native input projection. -/
def dotValue (project : Word → Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word) :
    TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => treeValue cfg (fun t => FP.mul cfg
      (project (a (i.1,t,PUnit.unit))) (project (b (t,i.2.1,PUnit.unit)))) (plans k).tree
  | _::tail => fun i => dotValue project cfg plans (batch:=tail) (fun j => a (i.1,j))
      (fun j => b (i.1,j)) i.2
termination_by batch.length

def softwareWithProjection (project : Word → Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) : Primitives Word where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  dot := fun _ _ {batch} {m} {k} {n} a b =>
    dotValue project cfg plans (batch:=batch) (m:=m) (k:=k) (n:=n) a b

def software (mode : InputMode) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n) : Primitives Word :=
  softwareWithProjection (inputProjection mode) cfg plans

noncomputable def expected (mode : InputMode) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) := accumulation (software mode cfg plans) p initial p.steps

theorem dot_retains_all_inputs (plan : ReductionPlan n) (k : Fin n) :
    some k ∈ plan.tree.leaves := by
  apply plan.valid.mem_iff.mpr
  simp

/-- Nonfinite bit patterns are not silently converted to finite numbers. -/
theorem tf32_nonfinite (x : Word) (hx : x.exponent=255) (tiesEven : Bool) :
    tf32Projection tiesEven x=x := by simp [tf32Projection,hx]

/-- Midpoint cases distinguish the two named software rounding rules; signed
zero, overflow and a signaling NaN boundary are included. These examples do
not establish which rule a device uses. -/
theorem projection_vectors :
    (tf32Projection Bool.true (FP.Value.ofBits .fp32 0x3f800001)).bits=BitVec.ofNat 32 0x3f800000 ∧
    (tf32Projection Bool.true (FP.Value.ofBits .fp32 0x3f801000)).bits=BitVec.ofNat 32 0x3f800000 ∧
    (tf32Projection Bool.false (FP.Value.ofBits .fp32 0x3f801000)).bits=BitVec.ofNat 32 0x3f802000 ∧
    (tf32Projection Bool.true (FP.Value.ofBits .fp32 0x3f803000)).bits=BitVec.ofNat 32 0x3f804000 ∧
    (tf32Projection Bool.true (FP.Value.ofBits .fp32 0x80000000)).bits=BitVec.ofNat 32 0x80000000 ∧
    (tf32Projection Bool.true (FP.Value.ofBits .fp32 0x7f7fffff)).bits=BitVec.ofNat 32 0x7f800000 ∧
    (tf32Projection Bool.true (FP.Value.ofBits .fp32 0x7f800001)).bits=BitVec.ofNat 32 0x7f800001 := by
  decide +kernel

#axiomsClean dot_retains_all_inputs
#axiomsClean roundedMagnitude_nearest
#axiomsClean tf32_grid
#axiomsClean tf32_nonfinite
#axiomsClean projection_vectors
#specNonCircular expected avoiding [source, FP.Structural.exec, correctness]
end VeriTile.Bench.TritonBenchG.MatmulTriton2.FPValue
