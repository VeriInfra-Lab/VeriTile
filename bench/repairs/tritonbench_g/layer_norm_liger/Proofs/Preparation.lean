import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Wrapper

namespace VeriTile.Bench.Repairs.LayerNormLiger.Preparation
open VeriTile.Triton FP.Structural VeriTile.Torch
noncomputable section

/-- Logical Tensor descriptor at the ordinary Torch API boundary. The values
belong to the Tensor, not to any normalization kernel or its reference. -/
structure Tensor (α : Type) where
  object : Nat
  metadata : TensorMetadata
  region : RegionName
  logical : Nat → α

structure Library (α : Type) where
  contiguous : Tensor α → Option (Tensor α)
  view : Tensor α → List Nat → Option (Tensor α)

/-- Generic contiguous/view API contracts. They do not know this task's
normalization, gradients, allocations or kernels. Failure is allowed. -/
structure Laws {α : Type} (lib : Library α) : Prop where
  contiguous : ∀ a b, a.metadata.dtype = .floating .fp32 → lib.contiguous a = some b →
    b.metadata.shape = a.metadata.shape ∧ b.metadata.dtype = a.metadata.dtype ∧
    b.metadata.device = a.metadata.device ∧ b.metadata.isContiguous = Bool.true ∧
    (∀ i, i < a.metadata.numel → b.logical i = a.logical i) ∧ 4 * b.metadata.numel ≤ 2^64
  view : ∀ a shape b, lib.view a shape = some b →
    shape.prod = a.metadata.numel → b.metadata.shape = shape ∧
    b.metadata.dtype = a.metadata.dtype ∧ b.metadata.device = a.metadata.device ∧
    b.region = a.region ∧ (∀ i, i < a.metadata.numel → b.logical i = a.logical i)

structure Prepared (α : Type) where
  (x weight bias dy : Tensor α)

def run {α : Type} (lib : Library α) (x w b dy : Tensor α) : Option (Prepared α) := do
  let cx ← lib.contiguous x
  let cw ← lib.contiguous w
  let cb ← lib.contiguous b
  let cg ← lib.contiguous dy
  let vx ← lib.view cx [Contract.rows x.metadata,Contract.width x.metadata]
  let vg ← lib.view cg [Contract.rows dy.metadata,Contract.width dy.metadata]
  return ⟨vx,cw,cb,vg⟩

def regions {α : Type} (p : Prepared α) : Wrapper.Inputs :=
  ⟨p.x.region,p.weight.region,p.bias.region,p.dy.region⟩

/-- Ordinary binding of the actual prepared Tensor to typed memory. It states
only where the input values reside after successful API preparation. -/
def Resident {α : Type} (s : State α) (t : Tensor α) : Prop :=
  ∀ i, i < t.metadata.numel → s.mem t.region i = .mk .fp32 (t.logical i)

structure Ready {α : Type} (s : State α) (p : Prepared α) : Prop where
  x : Resident s p.x
  weight : Resident s p.weight
  bias : Resident s p.bias
  dy : Resident s p.dy

structure Result {α : Type} (x w b dy : Tensor α) (p : Prepared α) : Prop where
  xShape : p.x.metadata.shape = [Contract.rows x.metadata,Contract.width x.metadata]
  weightShape : p.weight.metadata.shape = w.metadata.shape
  biasShape : p.bias.metadata.shape = b.metadata.shape
  dyShape : p.dy.metadata.shape = [Contract.rows dy.metadata,Contract.width dy.metadata]
  input : ∀ i, i < x.metadata.numel → p.x.logical i = x.logical i
  weight : ∀ i, i < w.metadata.numel → p.weight.logical i = w.logical i
  bias : ∀ i, i < b.metadata.numel → p.bias.logical i = b.logical i
  gradient : ∀ i, i < dy.metadata.numel → p.dy.logical i = dy.logical i
  capacity : 4 * x.metadata.numel ≤ 2^64

theorem complete {α : Type} (lib : Library α) (laws : Laws lib) (x w b dy : Tensor α)
    (eps : Contract.Epsilon) (h : Contract.InputContract x.metadata w.metadata b.metadata eps)
    (g : Contract.GradientContract dy.metadata x.metadata) (p : Prepared α)
    (success : run lib x w b dy = some p) : Result x w b dy p := by
  unfold run at success
  cases hc : lib.contiguous x with
  | none => simp [hc] at success
  | some cx =>
    cases hw : lib.contiguous w with
    | none => simp [hc,hw] at success
    | some cw =>
      cases hb : lib.contiguous b with
      | none => simp [hc,hw,hb] at success
      | some cb =>
        cases hg : lib.contiguous dy with
        | none => simp [hc,hw,hb,hg] at success
        | some cg =>
          cases hvx : lib.view cx [Contract.rows x.metadata,Contract.width x.metadata] with
          | none => simp [hc,hw,hb,hg,hvx] at success
          | some vx =>
            cases hvg : lib.view cg [Contract.rows dy.metadata,Contract.width dy.metadata] with
            | none => simp [hc,hw,hb,hg,hvx,hvg] at success
            | some vg =>
              have result : p = ⟨vx,cw,cb,vg⟩ := by simpa [hc,hw,hb,hg,hvx,hvg] using success.symm
              subst p
              have xc := laws.contiguous x cx h.input.dtype hc
              have wc := laws.contiguous w cw h.weight.dtype hw
              have bc := laws.contiguous b cb h.bias.dtype hb
              have gc := laws.contiguous dy cg g.tensor.dtype hg
              have xcount : cx.metadata.numel = x.metadata.numel := by simp [TensorMetadata.numel,xc.1]
              have gcount : cg.metadata.numel = dy.metadata.numel := by simp [TensorMetadata.numel,gc.1]
              have vxlaw := laws.view cx _ vx hvx (by simp [xcount,Contract.shape_elements x.metadata h.rank])
              have vglaw := laws.view cg _ vg hvg (by simp [gcount,Contract.shape_elements dy.metadata g.rank])
              refine ⟨vxlaw.1,wc.1,bc.1,vglaw.1,?_,wc.2.2.2.2.1,bc.2.2.2.2.1,?_,?_⟩
              · intro i hi
                exact (vxlaw.2.2.2.2 i (by simpa [xcount] using hi)).trans (xc.2.2.2.2.1 i hi)
              · intro i hi
                exact (vglaw.2.2.2.2 i (by simpa [gcount] using hi)).trans (gc.2.2.2.2.1 i hi)
              · simpa [xcount] using xc.2.2.2.2.2

/-- The actual kernel input cells denote the original logical coordinates.
The original ensure_contiguous preparation is not replaced by a new copy policy. -/
theorem input_cells {α : Type} (s : State α) (x w b dy : Tensor α) (p : Prepared α)
    (h : Result x w b dy p) (ready : Ready s p) (rank : 0 < x.metadata.shape.length) :
    ∀ i, i < x.metadata.numel → s.mem (regions p).x i = .mk .fp32 (x.logical i) := by
  intro i hi
  have size : p.x.metadata.numel = x.metadata.numel := by
    simp only [TensorMetadata.numel,h.xShape,List.prod_cons,List.prod_nil,Nat.mul_one]
    exact (Contract.shape_elements x.metadata rank).symm
  rw [show (regions p).x = p.x.region from rfl, ready.x i (by simpa [size] using hi),h.input i hi]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormLiger.Preparation
