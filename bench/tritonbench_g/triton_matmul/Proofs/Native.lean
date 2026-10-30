import bench.tritonbench_g.triton_matmul.Proofs.FP
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Native
open VeriTile.Triton FP.Structural FPValue
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- One output lane of the actual fixed-K tensor-core instructions. Operands
retain the FP16/E4M3FN storage payload; seed/result are FP32. No variable-K
tile or whole-matmul primitive is a numerical port. -/
abbrev MMA (n : Nat) := (Fin n → Word) → (Fin n → Word) → Word → Word
structure Instructions where
  half16 : MMA 16
  e4m3_32 : MMA 32
  castHalf : Word → BitVec 16
  castE4M3 : Word → BitVec 8

def cast (I : Instructions) (st : Storage) (x : Word) : Word :=
  match st with
  | .half => ⟨(I.castHalf x).setWidth 32⟩
  | .e4m3 => ⟨(I.castE4M3 x).setWidth 32⟩
def width : Storage → Nat | .half => 16 | .e4m3 => 32
def instruction (I : Instructions) : (st : Storage) → MMA (width st)
  | .half => I.half16
  | .e4m3 => I.e4m3_32
def lane (I : Instructions) (st : Storage) {n : Nat} (a : Fin n → Word) (group : Nat)
    (j : Fin (width st)) : Word :=
  if h : group*width st+j.val<n then a ⟨group*width st+j.val,h⟩ else cast I st (FP.Value.zero .fp32)
def chunk (I : Instructions) (st : Storage) {n : Nat} (a b : Fin n → Word) (seed : Word) (group : Nat) : Word :=
  instruction I st (lane I st a group) (lane I st b group) seed
def dot (I : Instructions) (st : Storage) (n : Nat) (a b : Fin n → Word) (seed : Word) : Word :=
  (List.range ((n+width st-1)/width st)).foldl (chunk I st a b) seed

def algebra (rest : Algebra Word) (I : Instructions) : Algebra Word :=
  {rest with
    literal := fun p d x => if x=0 then FP.Value.zero .fp32 else rest.literal p d x
    cast := fun p src dst x => match src,dst with
      | .real,.fp16 => cast I .half x
      | .real,.f8e4 => cast I .e4m3 x
      | src,dst => rest.cast p src dst x}
def dotValue (I : Instructions) (st : Storage) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word)
    (c : TileIndex (batch++[m,n]) → Word) : TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => dot I st k (fun e => a (i.1,e,PUnit.unit)) (fun e => b (e,i.2.1,PUnit.unit)) (c i)
  | _::tail => fun i => dotValue I st (batch:=tail) (fun j => a (i.1,j))
      (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2
termination_by batch.length
def dotAcc (I : Instructions) : DotAcc Word :=
  ⟨fun _ l _ _ _ {batch} {m} {k} {n} a b c =>
    dotValue I (if l=.f8e4 then .e4m3 else .half) a b c⟩
def numerics (I : Instructions) (st : Storage) : Numerics Word where
  zero := FP.Value.zero .fp32
  loadZero := cast I st (FP.Value.zero .fp32)
  dotAcc := dot I st
  castOutput := cast I st

/-- The historical four variable-shape helper laws are discharged from the
fixed-instruction source interpretation, without assuming whole-dot equality. -/
theorem primitives (rest : Algebra Word) (I : Instructions) (st : Storage) :
    PrimitiveLaws (algebra rest I) (dotAcc I) (numerics I st) st := by
  cases st <;> constructor
  all_goals intros
  all_goals simp [algebra,dotAcc,dotValue,numerics,dtype]
  all_goals norm_num

theorem instruction_lanes (st : Storage) (group : Nat)
    (hg : group<(SourceScope.blockK st+width st-1)/width st) (j : Fin (width st)) :
    group*width st+j.val<SourceScope.blockK st := by
  have hj := j.isLt
  cases st <;> norm_num [SourceScope.blockK,width] at hg hj ⊢ <;> omega

theorem lane_partition (st : Storage) (k : Fin (SourceScope.blockK st)) :
    k.val/width st*width st+k.val%width st=k.val ∧
    k.val%width st<width st ∧ k.val/width st<(SourceScope.blockK st+width st-1)/width st := by
  have hk := k.isLt
  cases st <;> norm_num [SourceScope.blockK,width] at hk ⊢ <;> omega

/-- Independent references remain at the fixed instruction/encoding boundary;
no internal tensor-core summation order is replaced by sequential FMA. -/
structure Reference where
  half16 : MMA 16
  e4m3_32 : MMA 32
  castHalf : Word → BitVec 16
  castE4M3 : Word → BitVec 8
def referenceInstructions (R : Reference) : Instructions :=
  ⟨R.half16,R.e4m3_32,R.castHalf,R.castE4M3⟩
structure Correspondence (I : Instructions) (R : Reference) : Prop where
  half16 : ∀ a b seed,I.half16 a b seed=R.half16 a b seed
  e4m3_32 : ∀ a b seed,I.e4m3_32 a b seed=R.e4m3_32 a b seed
  castHalf : ∀ x,I.castHalf x=R.castHalf x
  castE4M3 : ∀ x,I.castE4M3 x=R.castE4M3 x

theorem instructions_eq (I : Instructions) (R : Reference) (h : Correspondence I R) :
    I=referenceInstructions R := by
  cases I; cases R
  simp only [referenceInstructions,Instructions.mk.injEq]
  exact ⟨funext (fun a => funext (fun b => funext (h.half16 a b))),
    funext (fun a => funext (fun b => funext (h.e4m3_32 a b))),funext h.castHalf,funext h.castE4M3⟩

#axiomsClean primitives
#axiomsClean instruction_lanes
#axiomsClean lane_partition
#axiomsClean instructions_eq
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.Native
