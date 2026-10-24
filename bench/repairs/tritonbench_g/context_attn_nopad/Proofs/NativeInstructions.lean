import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeProfiles

namespace VeriTile.Bench.Repairs.ContextAttnNopad.NativeInstructions
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Word := BitVec 32

/-- SM80 scalar instructions and a single output of one fixed m16n8k8 TF32
MMA instruction. An arbitrary source dot/reduction is not a numerical port. -/
structure ScalarOps where
  literal32 : ℝ → Word
  add : Word → Word → Word
  sub : Word → Word → Word
  mul : Word → Word → Word
  div : Word → Word → Word
  max : Word → Word → Word
  exp2 : Word → Word
  fmaRN : Word → Word → Word → Word
  mmaTF32 : Fin 16 → Fin 8 → (Fin 8 → Word) → (Fin 8 → Word) → Word → Word

def binary (ops : ScalarOps) : Binary → Word → Word → Word
  | .add => ops.add | .sub => ops.sub | .mul => ops.mul
  | .div => ops.div | .max => ops.max | _ => fun _ _ => 0

/-- Explicit ascending-K instruction chain, with the original zero/live seed. -/
def dotWord (ops : ScalarOps) (row column k : Nat) (a b : Fin k → Word) (seed : Word) : Word :=
  (List.range ((k+7)/8)).foldl (fun previous block =>
    ops.mmaTF32 ⟨row%16,Nat.mod_lt _ (by decide)⟩ ⟨column%8,Nat.mod_lt _ (by decide)⟩
      (fun lane => if h : block*8+lane.val < k then a ⟨block*8+lane.val,h⟩ else 0)
      (fun lane => if h : block*8+lane.val < k then b ⟨block*8+lane.val,h⟩ else 0) previous) seed

def dotAt (ops : ScalarOps) : (batch : TileShape) → {m k n : Nat} →
    (TileIndex (batch++[m,k]) → Word) → (TileIndex (batch++[k,n]) → Word) →
    Word → TileIndex (batch++[m,n]) → Word
  | [],_,k,_,a,b,seed,i => dotWord ops i.1.val i.2.1.val k
      (fun j => a ⟨i.1,j,()⟩) (fun j => b ⟨j,i.2.1,()⟩) seed
  | _::ds,_,_,_,a,b,seed,i => dotAt ops ds (fun j => a ⟨i.1,j⟩) (fun j => b ⟨i.1,j⟩) seed i.2

def operations (ops : ScalarOps) : NativeGraph.Operations where
  literal := fun _ _ => ops.literal32
  fromNat := fun _ _ => 0
  fromInt := fun _ _ => 0
  cast := fun _ source target x => if source=target then x else 0
  unary := fun _ op x => if op=.exp2 then ops.exp2 x else 0
  binary := fun _ _ => binary ops
  fma := fun _ _ => ops.fmaRN
  dotAcc := fun _ _ _ _ _ batch _ _ _ => dotAt ops batch

def evaluate (ops : ScalarOps) (plan : NativePhysical.Plan) : NativePhysical.Expr → NativePhysical.Owner → Word :=
  NativePhysical.evaluate (operations ops) plan

theorem feature_coverage (k d : Nat) (hd : d < k) :
    d/8∈List.range ((k+7)/8) ∧ d%8 < 8 ∧ d/8*8+d%8=d := by
  refine ⟨List.mem_range.mpr (by omega),Nat.mod_lt _ (by decide),?_⟩
  simpa [Nat.mul_comm] using Nat.div_add_mod d 8

theorem feature_unique (d block lane : Nat) (hl : lane < 8) (he : block*8+lane=d) :
    block=d/8 ∧ lane=d%8 := by omega

structure ScalarLaws (actual reference : ScalarOps) : Prop where
  literal32 : ∀ x,actual.literal32 x=reference.literal32 x
  add : ∀ x y,actual.add x y=reference.add x y
  sub : ∀ x y,actual.sub x y=reference.sub x y
  mul : ∀ x y,actual.mul x y=reference.mul x y
  div : ∀ x y,actual.div x y=reference.div x y
  max : ∀ x y,actual.max x y=reference.max x y
  exp2 : ∀ x,actual.exp2 x=reference.exp2 x
  fmaRN : ∀ x y z,actual.fmaRN x y z=reference.fmaRN x y z
  mmaTF32 : ∀ row column a b seed,actual.mmaTF32 row column a b seed=reference.mmaTF32 row column a b seed

theorem ScalarLaws.operations_eq (actual reference : ScalarOps) (h : ScalarLaws actual reference) : actual=reference := by
  cases actual
  cases reference
  simp only [ScalarOps.mk.injEq]
  exact ⟨funext h.literal32,
    funext (fun x => funext (h.add x)),funext (fun x => funext (h.sub x)),
    funext (fun x => funext (h.mul x)),funext (fun x => funext (h.div x)),
    funext (fun x => funext (h.max x)),funext h.exp2,
    funext (fun x => funext (fun y => funext (h.fmaRN x y))),
    funext (fun row => funext (fun column => funext (fun a => funext (fun b => funext (h.mmaTF32 row column a b)))))⟩

theorem evaluated_if_local (actual reference : ScalarOps) (h : ScalarLaws actual reference)
    (plan : NativePhysical.Plan) (e : NativePhysical.Expr) (owner : NativePhysical.Owner) :
    evaluate actual plan e owner=evaluate reference plan e owner := by
  rw [h.operations_eq actual reference]

#axiomsClean feature_coverage
#axiomsClean evaluated_if_local
end
end VeriTile.Bench.Repairs.ContextAttnNopad.NativeInstructions
