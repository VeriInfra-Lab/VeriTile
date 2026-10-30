import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Execution

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Instructions
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
noncomputable section

/-- Fixed local operations in the observed SM80 TF32 lowering. Half values use
the existing packed low-16-bit carrier between narrowing and widening. -/
abbrev MMA8 := (Fin 8 → Word) → (Fin 8 → Word) → Word → Word
structure Native where
  tf32 : Word → Word
  mma8 : MMA8
  narrow : Word → Word
  widen : Word → Word

def lane (I : Native) {n : Nat} (a : Fin n → Word) (group : Nat) (j : Fin 8) : Word :=
  I.tf32 (if h : group*8+j.val<n then a ⟨group*8+j.val,h⟩ else FP.Value.zero .fp32)
def chunk (I : Native) {n : Nat} (a b : Fin n → Word) (seed : Word) (group : Nat) : Word :=
  I.mma8 (lane I a group) (lane I b group) seed
def dot (I : Native) (n : Nat) (a b : Fin n → Word) (seed : Word) : Word :=
  (List.range ((n+7)/8)).foldl (chunk I a b) seed

def dotValue (I : Native) {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → Word) (b : TileIndex (batch++[k,n]) → Word)
    (c : TileIndex (batch++[m,n]) → Word) : TileIndex (batch++[m,n]) → Word :=
  match batch with
  | [] => fun i => dot I k (fun e => a (i.1,e,PUnit.unit)) (fun e => b (e,i.2.1,PUnit.unit)) (c i)
  | _::tail => fun i => dotValue I (batch:=tail) (fun j => a (i.1,j))
      (fun j => b (i.1,j)) (fun j => c (i.1,j)) i.2
termination_by batch.length

def dotPort (I : Native) : Dot :=
  ⟨fun _ _ _ _ _ {batch} {m} {k} {n} a b => dotValue I a b (fun _ => FP.Value.zero .fp32)⟩
def accumulatorPort (I : Native) : DotAcc :=
  ⟨fun _ _ _ _ _ {batch} {m} {k} {n} a b c => dotValue I a b c⟩

def arithmetic (cfg : FP.Config) (I : Native) (r : Request) : Execution.Arithmetic Word r where
  zero := FP.Value.zero .fp32
  add := FP.add cfg
  dot := (dotPort I).apply (some .fp32) .fp32 .fp32 .fp32 {} (batch:=[])
  dotAcc := (accumulatorPort I).apply (some .fp32) .fp32 .fp32 .real {} (batch:=[])
  narrow := I.narrow
  widen := I.widen

/-- Independent fixed-instruction reference. No variable-shape dot, source/fused
equivalence, full output, address or scheduling law is a numerical premise. -/
structure Reference where
  cfg : FP.Config
  tf32 : Word → Word
  mma8 : MMA8
def referenceInstructions (R : Reference) : Native :=
  ⟨R.tf32,R.mma8,castWord R.cfg .single .half,castWord R.cfg .half .single⟩
structure Correspondence (I : Native) (R : Reference) : Prop where
  tf32 : ∀ x,I.tf32 x=R.tf32 x
  mma8 : ∀ a b seed,I.mma8 a b seed=R.mma8 a b seed
  narrow : ∀ x,I.narrow x=castWord R.cfg .single .half x
  widen : ∀ x,I.widen x=castWord R.cfg .half .single x

theorem native_eq (I : Native) (R : Reference) (h : Correspondence I R) : I=referenceInstructions R := by
  cases I; cases R
  simp only [referenceInstructions,Native.mk.injEq]
  exact ⟨funext h.tf32,funext (fun a => funext (fun b => funext (h.mma8 a b))),funext h.narrow,funext h.widen⟩

theorem arithmetic_reference (R : Reference) (r : Request) :
    arithmetic R.cfg (referenceInstructions R) r=
      Execution.fp R.cfg (dotPort (referenceInstructions R)) (accumulatorPort (referenceInstructions R)) r := rfl

theorem lane_partition (n : Nat) (k : Fin n) :
    k.val/8*8+k.val%8=k.val ∧ k.val%8<8 ∧ k.val/8<(n+7)/8 := by
  have := k.isLt
  omega

#axiomsClean native_eq
#axiomsClean arithmetic_reference
#axiomsClean lane_partition
end
end VeriTile.Bench.Repairs.IvDependentMatmul.Instructions
