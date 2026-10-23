import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.NumericOrigins
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.StorageOrigins

namespace VeriTile.Bench.Repairs.CrossEntropyOps.Resident
open VeriTile.Triton
open CrossEntropy1.Floating
open CrossEntropy1.Host (World value)
open CrossEntropy1.Physical (Layout Byte byteOf flipSign flipSign_involutive tensorByte)
open VeriTile.Bench.Repairs.CrossEntropy1.TensorModel (Tensor)
noncomputable section

def rawWordAt {σ : Type} (layout : Layout σ) (observe : σ → Nat → Nat → Byte)
    (native : σ) (region : RegionName) (index : Nat) : Word :=
  let b := rawByte layout observe native region index
  ((b ⟨3,by decide⟩).append (b ⟨2,by decide⟩)).append
    ((b ⟨1,by decide⟩).append (b ⟨0,by decide⟩))

def wordInside {σ : Type} (layout : Layout σ) (native : σ) (r : RegionName) (i : Nat) : Prop :=
  4*(i+1)≤(layout.buffer native r).capacity

/-- One source-labelled execution fixes both the origin tape and the complete
occurrence fusion policy. A caller cannot choose a new policy per output.
The ordinary frontend contract must jointly connect this tape to its AST,
entry, complete ABI, source run and native execution; this data alone does not
assert that connection. -/
structure OriginTrace {σ : Type} (selection : Selection σ) (fusion : ConfiguredPlan.FusionSelection σ)
    (readRaw : σ → RegionName → Nat → Word) where
  policy : ConfiguredPlan.Policy fusion
  initial : Origins.State selection
  final : Origins.State selection
  events : List (Origins.Event σ)
  steps : Origins.Run selection readRaw initial events final
  initialWellFormed : initial.WellFormed

def OriginTrace.requestsBound {σ : Type} {selection : Selection σ} {fusion readRaw}
    (trace : OriginTrace selection fusion readRaw) (requests : List (InvocationGraphs.Request σ)) : Prop :=
  ∀ event∈trace.events, trace.policy.request event.point.site.call∈requests

def storedWord {σ : Type} {selection : Selection σ} (contract : Occurrence σ → Bool)
    (node : Origins.Value selection) (negative : Bool) (numerics : Numerics) : Word :=
  let core := ConfiguredPlan.reference contract node.plan (signNumerics numerics)
  if negative then flipSign core else core

/-- Fixed representation of defined source storage. Only assigned SSA cells
need a numeric node; allocation holes remain undefined. Final object exposure
is separate and contains only ordinary descriptors and coordinates. -/
structure NumericRep {σ : Type} {selection : Selection σ} {fusion}
    (layout : Layout σ) (observe : σ → Nat → Nat → Byte) (native : Numerics)
    (trace : OriginTrace selection fusion (rawWordAt layout observe))
    (source : World SignedExpr σ) (exports : List Tensor) : Prop where
  undefined : ∀ r i, source.live r → wordInside layout source.native r i → trace.final.cells (r,i)=none →
    (source.memory r i).read .real=⟨.raw (rawWordAt layout observe source.native r i),Bool.false⟩
  defined : ∀ region index stored, trace.final.cells (region,index)=some stored → source.live region →
    wordInside layout source.native region index →
    ∃ node, trace.final.nodes stored.id=some node ∧
      (source.memory region index).read .real=⟨node.expression,stored.negative⟩ ∧
      ∀ byte, rawByte layout observe source.native region index byte=
        byteOf (storedWord trace.policy.contract node stored.negative native) byte
  exposed : ∀ tensor∈exports, tensor.dtype="torch.float32" → ∀ index, index<tensor.numel →
    source.live tensor.region ∧ wordInside layout source.native tensor.region (tensor.index index) ∧
      ∃ stored, trace.final.cells (tensor.region,tensor.index index)=some stored

/-- Derived observation of the existing node, rather than a newly chosen
SelectedPlan. `core` is the only index transport used after the source theorem. -/
structure Observation {σ : Type} {selection : Selection σ} {fusion readRaw}
    (trace : OriginTrace selection fusion readRaw) (tensor : Tensor) (index : Nat) (expected : SignedExpr) where
  stored : Origins.Stored
  node : Origins.Value selection
  storedAt : trace.final.cells (tensor.region,tensor.index index)=some stored
  nodeAt : trace.final.nodes stored.id=some node
  core : node.expression=expected.core
  sign : Bool.xor stored.negative tensor.isNeg=expected.negative

def Observation.plan {σ : Type} {selection : Selection σ} {fusion readRaw}
    {trace : OriginTrace selection fusion readRaw} {tensor index expected}
    (observation : Observation trace tensor index expected) : SelectedPlan selection expected.core :=
  observation.node.transport observation.core

def Observation.reference {σ : Type} {selection : Selection σ} {fusion readRaw}
    {trace : OriginTrace selection fusion readRaw} {tensor index expected}
    (observation : Observation trace tensor index expected) (n : Numerics) : Word :=
  signedReference trace.policy.contract observation.plan n

theorem NumericRep.observation {σ : Type} {selection : Selection σ} {fusion}
    {layout : Layout σ} {observe native trace source exports}
    (rep : NumericRep (selection:=selection) (fusion:=fusion) layout observe native trace source exports)
    (integer : World SignedExpr σ → Tensor → Nat → Int) (tensor : Tensor) (member : tensor∈exports)
    (dtype : tensor.dtype="torch.float32") (index : Nat) (inside : index<tensor.numel)
    (expected : SignedExpr) (logical : value (numericObservations integer) source tensor index=expected) :
    Nonempty (Observation trace tensor index expected) := by
  obtain ⟨live,backed,stored,storedAt⟩ := rep.exposed tensor member dtype index inside
  obtain ⟨node,nodeAt,cell,bytes⟩ := rep.defined _ _ stored storedAt live backed
  have result : value (numericObservations integer) source tensor index=
      SignedExpr.mk node.expression (Bool.xor stored.negative tensor.isNeg) := by
    simp only [value,numericObservations,cell,SignedExpr.neg]
    cases tensor.isNeg <;> cases stored.negative <;> rfl
  have same := result.symm.trans logical
  exact ⟨⟨stored,node,storedAt,nodeAt,congrArg SignedExpr.core same,congrArg SignedExpr.negative same⟩⟩

theorem NumericRep.reference_bytes {σ : Type} {selection : Selection σ} {fusion}
    {layout : Layout σ} {observe native trace source exports}
    (rep : NumericRep (selection:=selection) (fusion:=fusion) layout observe native trace source exports)
    {tensor : Tensor} {index : Nat} {expected : SignedExpr}
    (observation : Observation trace tensor index expected) (live : source.live tensor.region)
    (backed : wordInside layout source.native tensor.region (tensor.index index))
    (reference : Numerics) (ports : Obligations (signNumerics native) (signNumerics reference)) (byte : Fin 4) :
    tensorByte layout observe source.native tensor index byte=
      byteOf (if tensor.isNeg then flipSign (observation.reference reference)
        else observation.reference reference) byte := by
  obtain ⟨node,nodeAt,cell,bytes⟩ := rep.defined _ _ observation.stored observation.storedAt live backed
  have nodeEq : node=observation.node := Option.some.inj (nodeAt.symm.trans observation.nodeAt)
  subst node
  have same := observation.node.transport_reference observation.core trace.policy.contract (signNumerics reference)
  have numerical := ConfiguredPlan.reference_equiv trace.policy.contract observation.node.plan
    (signNumerics native) (signNumerics reference) ports
  have bits := bytes byte
  change tensorByte layout observe source.native tensor index byte=_ at bits
  rw [bits]
  unfold storedWord Observation.reference signedReference Observation.plan
  rw [same,numerical,←observation.sign]
  cases tensor.isNeg <;> cases observation.stored.negative <;>
    simp only [Bool.xor_false,Bool.xor_true,
      Bool.not_false,Bool.not_true,Bool.false_eq_true,if_false,if_true,flipSign_involutive]

#axiomsClean NumericRep.observation
#axiomsClean NumericRep.reference_bytes
#specNonCircular NumericRep avoiding [Spec.loss,Spec.zLoss,Spec.gradient,Source.forward,Source.backward,Host.forward,Host.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Resident
