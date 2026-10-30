import bench.tritonbench_g.ksoftmax_triton.Proofs.Graph
import bench.tritonbench_g.masked_add_cuda.Proofs.Words

namespace VeriTile.Bench.Repairs.Ksoftmax.Words
open VeriTile.Triton FP.Structural
open Graph (Word Expr Operations eval)
noncomputable section
abbrev finite := MaskedAddCuda.Words.finite

def softwareAdd (x y : Word) : Word := (FP.add FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩).bits
def softwareSub (x y : Word) : Word := (FP.sub FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩).bits
def softwareMul (x y : Word) : Word := (FP.mul FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩).bits
def softwareFMA (x y z : Word) : Word := (FP.fma FP.Config.ieeeValue (⟨x⟩ : FP.Value .fp32) ⟨y⟩ ⟨z⟩).bits

/-- Named raw-word reference ports for the selected original FP32 intrinsics.
These are not assumed correctly rounded Real exp/log or universal IEEE ports.
Discharge pins the compiler/options/target, FTZ and exceptional-value policy. -/
structure PrimitiveReference where
  divisionFull32 : Word → Word → Word
  maximum32 : Word → Word → Word
  exp2Approx32 : Word → Word
  libdeviceLog32 : Word → Word

def referenceOps (reference : PrimitiveReference) : Operations :=
  ⟨softwareAdd,softwareSub,softwareMul,reference.divisionFull32,reference.maximum32,
    softwareFMA,reference.exp2Approx32,reference.libdeviceLog32⟩

def AddDomain (x y : Word) : Prop :=
  (finite x ∧ finite y) ∨ (x=Graph.negInf ∧ y=Graph.negInf)
def SubDomain (x y : Word) : Prop :=
  (finite x ∨ x=Graph.negInf) ∧ finite y
def MulDomain (x y : Word) : Prop :=
  (finite x ∧ finite y) ∨ (x=Graph.negInf ∧ y=Graph.log2e)

/-- Eight instruction-local ports; no kernel output, success, address, schedule,
copyback, memory frame, or whole reduction equality is a numerical premise. -/
structure LocalExact (native : Operations) (reference : PrimitiveReference) : Prop where
  add : ∀ x y, AddDomain x y → native.add32 x y=softwareAdd x y
  sub : ∀ x y, SubDomain x y → native.sub32 x y=softwareSub x y
  mul : ∀ x y, MulDomain x y → native.mul32 x y=softwareMul x y
  fma : ∀ x y z, finite x → finite y → finite z → native.fmaRN32 x y z=softwareFMA x y z
  division : ∀ x y, native.divFull32 x y=reference.divisionFull32 x y
  maximum : ∀ x y, native.max32 x y=reference.maximum32 x y
  exp2 : ∀ x, native.ex2Approx32 x=reference.exp2Approx32 x
  logarithm : ∀ x, native.libdeviceLog32 x=reference.libdeviceLog32 x

/-- Domain lists operands at every actual rounding node, including intermediate
reductions/FMA and exponential scaling. Finite leaves alone are insufficient. -/
def NumericDomain (ops : Operations) : Expr → Prop
  | .leaf _ | .input _ _ _ => True
  | .load x | .neg x | .ex2 x | .log x => NumericDomain ops x
  | .add x y => NumericDomain ops x ∧ NumericDomain ops y ∧ AddDomain (eval ops x) (eval ops y)
  | .sub x y => NumericDomain ops x ∧ NumericDomain ops y ∧ SubDomain (eval ops x) (eval ops y)
  | .mul x y => NumericDomain ops x ∧ NumericDomain ops y ∧ MulDomain (eval ops x) (eval ops y)
  | .div x y | .max x y => NumericDomain ops x ∧ NumericDomain ops y
  | .fma x y z => NumericDomain ops x ∧ NumericDomain ops y ∧ NumericDomain ops z ∧
      finite (eval ops x) ∧ finite (eval ops y) ∧ finite (eval ops z)

theorem local_exact (native : Operations) (reference : PrimitiveReference) (ports : LocalExact native reference)
    (e : Expr) (domain : NumericDomain native e) : eval native e=eval (referenceOps reference) e := by
  induction e with
  | leaf w => rfl
  | input objectId index w => rfl
  | load x ih => exact ih domain
  | neg x ih => exact congrArg (fun w => w ^^^ BitVec.ofNat 32 (2^31)) (ih domain)
  | add x y ix iy =>
    obtain ⟨dx,dy,valid⟩:=domain
    exact (ports.add _ _ valid).trans (congrArg₂ softwareAdd (ix dx) (iy dy))
  | sub x y ix iy =>
    obtain ⟨dx,dy,valid⟩:=domain
    exact (ports.sub _ _ valid).trans (congrArg₂ softwareSub (ix dx) (iy dy))
  | mul x y ix iy =>
    obtain ⟨dx,dy,valid⟩:=domain
    exact (ports.mul _ _ valid).trans (congrArg₂ softwareMul (ix dx) (iy dy))
  | div x y ix iy =>
    exact (ports.division _ _).trans (congrArg₂ reference.divisionFull32 (ix domain.1) (iy domain.2))
  | max x y ix iy =>
    exact (ports.maximum _ _).trans (congrArg₂ reference.maximum32 (ix domain.1) (iy domain.2))
  | fma x y z ix iy iz =>
    obtain ⟨dx,dy,dz,fx,fy,fz⟩:=domain
    change native.fmaRN32 (eval native x) (eval native y) (eval native z)=_
    rw [ports.fma _ _ _ fx fy fz,ix dx,iy dy,iz dz]
    rfl
  | ex2 x ih => exact (ports.exp2 _).trans (congrArg reference.exp2Approx32 (ih domain))
  | log x ih => exact (ports.logarithm _).trans (congrArg reference.libdeviceLog32 (ih domain))

theorem reference_exact (rest : Algebra Expr) (profile : Graph.Profile) (native : Operations)
    (reference : PrimitiveReference) (ports : LocalExact native reference) (op : Operation)
    (Q K batch row leftId rightId : Nat) {D : Nat} (left right : Nat → Word) (i : Fin D)
    (domain : NumericDomain native ((gridValue (Graph.algebra rest profile) op Q K batch row
      (fun j => .input leftId j (left j)) (fun j => .input rightId j (right j)) i).read .real)) :
    Graph.reference rest profile native op Q K batch row leftId rightId left right i=
      Graph.reference rest profile (referenceOps reference) op Q K batch row leftId rightId left right i :=
  local_exact native reference ports _ domain

#specNonCircular LocalExact avoiding [runRows]
#axiomsClean local_exact
#axiomsClean reference_exact
end
end VeriTile.Bench.Repairs.Ksoftmax.Words
