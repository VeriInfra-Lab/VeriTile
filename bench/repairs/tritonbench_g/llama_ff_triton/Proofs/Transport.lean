import bench.repairs.tritonbench_g.llama_ff_triton.Proofs.Core

namespace VeriTile.Bench.Repairs.LlamaFF
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

structure Ops.Hom (a : Ops F) (b : Ops G) (f : F → G) : Prop where
  zero : f a.zero=b.zero
  one : f a.one=b.one
  add : ∀ x y,f (a.add x y)=b.add (f x) (f y)
  mul : ∀ x y,f (a.mul x y)=b.mul (f x) (f y)
  mulInput : ∀ x y,f (a.mulInput x y)=b.mulInput (f x) (f y)
  sigmoidNeg : ∀ x,f (a.sigmoidNeg x)=b.sigmoidNeg (f x)
  neg : ∀ x,f (a.neg x)=b.neg (f x)
  fma : ∀ x y z,f (a.fma x y z)=b.fma (f x) (f y) (f z)
  mulFTZ : ∀ x y,f (a.mulFTZ x y)=b.mulFTZ (f x) (f y)
  divide : ∀ x y,f (a.divide x y)=b.divide (f x) (f y)
  rsqrt : ∀ x,f (a.rsqrt x)=b.rsqrt (f x)
  ofInt : ∀ n,f (a.ofInt n)=b.ofInt n
  mma16 : ∀ x y old,f (a.mma16 x y old)=b.mma16 (fun i => f (x i)) (fun i => f (y i)) (f old)

def Parameters.map (f : F → G) (p : Parameters F) : Parameters G :=
  ⟨p.K,f p.epsilon⟩
def RowInput.map (f : F → G) (x : RowInput F) : RowInput G :=
  ⟨fun k => f (x.x k),fun k => f (x.rms k),fun n k => f (x.weight n k),fun n k => f (x.weight3 n k)⟩

variable (a : Ops F) (b : Ops G) (f : F → G) (h : a.Hom b f)
  (p : Parameters F) (x : RowInput F)
include h
theorem masked_map (v : Nat → F) (t : Nat) (l : Fin 64) :
    f (masked a p v t l)=masked b (p.map f) (fun k => f (v k)) t l := by
  by_cases hk : t*64+l.val<p.K <;> simp [masked,Parameters.map,hk,h.zero]
theorem scaled_map (t : Nat) (l : Fin 64) :
    f (scaled a p x t l)=scaled b (p.map f) (x.map f) t l := by
  simp only [scaled,h.mulInput,masked_map a b f h,RowInput.map]
theorem instruction_map (n t : Nat) (lanes : Fin 16 → Fin 64) (old : F) :
    f (instruction a p x n t lanes old)=instruction b (p.map f) (x.map f) n t lanes (f old) := by
  simp only [instruction,h.mma16,scaled_map a b f h,masked_map a b f h,RowInput.map]
theorem dotBlock_map (plan : DotPlan) (n t : Nat) (old : F) :
    f (dotBlock a p x plan n t old)=dotBlock b (p.map f) (x.map f) plan n t (f old) := by
  unfold dotBlock
  induction plan.instructions generalizing old with
  | nil => rfl
  | cons lanes rest ih => simp only [List.foldl_cons,ih,instruction_map a b f h]
theorem accumulation_map (plans : Nat → DotPlan) (n t : Nat) :
    f (accumulation a p x plans n t)=accumulation b (p.map f) (x.map f) plans n t := by
  induction t with
  | zero => exact h.zero
  | succ t ih => simp only [accumulation,dotBlock_map a b f h,ih]
theorem square_map (v : F) : f (square a v)=square b (f v) := by
  simp only [square,h.mulFTZ,h.one]
theorem moment_map (l : Fin 64) (t : Nat) :
    f (moment a p x l t)=moment b (p.map f) (x.map f) l t := by
  induction t with
  | zero => exact h.zero
  | succ t ih => simp only [moment,h.add,ih,square_map a b f h,masked_map a b f h,RowInput.map]
theorem tree_map (tree : Tree) (v : Fin 64 → F) :
    f (tree.eval a.add v)=tree.eval b.add (fun k => f (v k)) := by
  induction tree with
  | leaf k => rfl
  | node u v iu iv => simp only [Tree.eval,h.add,iu,iv]
theorem norm_map (reduction : Reduction) :
    f (norm a p x reduction)=norm b (p.map f) (x.map f) reduction := by
  simp only [norm,h.rsqrt,h.add,h.divide,h.ofInt,tree_map a b f h,moment_map a b f h,Parameters.map]
theorem projection_map (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    f (projection a p x plans reduction n)=projection b (p.map f) (x.map f) plans reduction n := by
  simp only [projection,h.mul,accumulation_map a b f h,norm_map a b f h,Parameters.map]


theorem value_map (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    f (value a p x plans reduction n)=value b (p.map f) (x.map f) plans reduction n := by
  simp only [value,h.mul,h.sigmoidNeg,h.fma,h.neg,h.zero,projection_map a b f h,accumulation_map a b f h,norm_map a b f h,Parameters.map]
  rfl
#axiomsClean value_map

#axiomsClean accumulation_map
#axiomsClean moment_map

end
end VeriTile.Bench.Repairs.LlamaFF
