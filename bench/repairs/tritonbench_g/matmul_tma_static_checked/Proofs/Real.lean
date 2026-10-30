import bench.repairs.tritonbench_g.matmul_tma_static_checked.Proofs.Tensor

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.RealModel
open VeriTile.Triton FP.Structural FPValue Host TensorABI
noncomputable section
set_option maxHeartbeats 3000000

def algebra (rest : Algebra ℝ) : Algebra ℝ := {rest with cast := fun _ _ _ x => x}
def contraction {batch : TileShape} {m k n : Nat}
    (a : TileIndex (batch++[m,k]) → ℝ) (b : TileIndex (batch++[k,n]) → ℝ) : TileIndex (batch++[m,n]) → ℝ :=
  match batch with
  | [] => fun i => ∑ t : Fin k, a (i.1,t,PUnit.unit)*b (t,i.2.1,PUnit.unit)
  | _::tail => fun i => contraction (batch:=tail) (fun j => a (i.1,j)) (fun j => b (i.1,j)) i.2
termination_by batch.length
def dot : Dot ℝ := ⟨fun _ _ {batch} {m} {k} {n} a b => contraction a b⟩

/-- Independent ordinary matrix product of the original physical views. -/
def product (read : Nat → ℝ) (h : Shape) (f : Flags) (a b : TensorABI.Tensor)
    (i : CIndex (params h f)) : ℝ :=
  ∑ k : Fin h.inner, logical read 2 a i.1.val k.val * logical read 2 b k.val i.2.1.val

theorem output_product (rest : Algebra ℝ) (read : Nat → ℝ) (h : Shape) (f : Flags)
    (a b c : TensorABI.Tensor) (abi : Binding h f a b c) (s : State ℝ) (input : Inputs read h a b s) :
    outputValues (algebra rest) dot (params h f) f.half s = product read h f a b := by
  have inputs := source_inputs read h f a b c abi s input
  funext i
  simp only [outputValues,algebra,dot,contraction,product,inputs.1,inputs.2]
  rfl

/-- Physical interpretation of the source's completed stores at element starts.
Arbitrary mathematical Reals are not assigned a floating-point byte encoding. -/
def render (read : Nat → ℝ) (h : Shape) (f : Flags) (c : TensorABI.Tensor) (s : State ℝ) : Nat → ℝ := fun ptr =>
  if c.metadata.dataPtr.toNat ≤ ptr ∧ ptr < c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns) ∧
      (ptr-c.metadata.dataPtr.toNat)%width f.half=0 then
    value (outputDType f.half) (s.mem "C" ((ptr-c.metadata.dataPtr.toNat)/width f.half))
  else read ptr

theorem render_at (read : Nat → ℝ) (h : Shape) (f : Flags) (c : TensorABI.Tensor) (s : State ℝ)
    (off : Nat) (bound : off<h.rows*h.columns) :
    render read h f c s (c.metadata.dataPtr.toNat+width f.half*off) = value (outputDType f.half) (s.mem "C" off) := by
  have inside : c.metadata.dataPtr.toNat ≤ c.metadata.dataPtr.toNat+width f.half*off ∧
      c.metadata.dataPtr.toNat+width f.half*off<c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns) ∧
      (c.metadata.dataPtr.toNat+width f.half*off-c.metadata.dataPtr.toNat)%width f.half=0 := by
    cases f.half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
  have index : (c.metadata.dataPtr.toNat+width f.half*off-c.metadata.dataPtr.toNat)/width f.half=off := by
    cases f.half <;> simp only [width,Bool.false_eq_true,↓reduceIte] at * <;> omega
  simp only [render,if_pos inside,index]

theorem frame (read : Nat → ℝ) (h : Shape) (f : Flags) (c : TensorABI.Tensor) (s : State ℝ)
    (ptr : Nat) (outside : ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns)≤ptr) :
    render read h f c s ptr=read ptr := by
  simp only [render,if_neg (by omega : ¬(c.metadata.dataPtr.toNat≤ptr ∧
    ptr<c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns) ∧ (ptr-c.metadata.dataPtr.toNat)%width f.half=0))]

structure TensorPost (read final : Nat → ℝ) (h : Shape) (f : Flags) (a b c : TensorABI.Tensor) : Prop where
  values : ∀ i : CIndex (params h f), logical final (width f.half) c i.1.val i.2.1.val = product read h f a b i
  frame : ∀ ptr, ptr<c.metadata.dataPtr.toNat ∨ c.metadata.dataPtr.toNat+width f.half*(h.rows*h.columns)≤ptr → final ptr=read ptr

theorem tensor_post (read : Nat → ℝ) (h : Shape) (f : Flags) (a b c : TensorABI.Tensor)
    (abi : Binding h f a b c) (s t : State ℝ)
    (post : Post (params h f) f.half s t (product read h f a b)) :
    TensorPost read (render read h f c t) h f a b c := by
  refine ⟨?_,frame read h f c t⟩
  intro i
  rw [logical,c_address h f a b c abi i,render_at read h f c t (cOffset (params h f) i) (c_bound h f regions i)]
  have hp := post.1 i
  rw [regionC] at hp
  rw [hp]
  exact read_stored _ _

#specNonCircular product avoiding [kernel,source,FP.Structural.exec,scatter]
#axiomsClean output_product
#axiomsClean tensor_post
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticChecked.RealModel
