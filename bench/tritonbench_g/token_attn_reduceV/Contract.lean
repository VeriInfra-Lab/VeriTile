import bench.tritonbench_g.token_attn_reduceV.Kernels
import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
open VeriTile.Triton
noncomputable section

inductive Role where
  | probability | value | output | pages | request | start | length
  deriving DecidableEq, Repr
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata
abbrev Word := BitVec 32
abbrev Reader := Nat → Word
def roles : List Role := [.probability,.value,.output,.pages,.request,.start,.length]
def reads : List Role := [.probability,.value,.pages,.request,.start,.length]
def integers : List Role := [.pages,.request,.start,.length]
def base (m : Inputs) (r : Role) := (m r).dataPtr.toNat
def batches (m : Inputs) := (m .request).numel
def heads (m : Inputs) := (m .probability).shape.headD 0
def tokens (m : Inputs) := (m .probability).shape[1]?.getD 0
def cache (m : Inputs) := (m .value).shape.headD 0
def kvheads (m : Inputs) := (m .value).shape[1]?.getD 0
def width (m : Inputs) := (m .value).shape[2]?.getD 0
def requests (m : Inputs) := (m .pages).shape.headD 0
def columns (m : Inputs) := (m .pages).shape[1]?.getD 0
def group (m : Inputs) := heads m / kvheads m
def raw (m : Inputs) (read : Reader) (r : Role) (i : Nat) : Int :=
  (read (base m r+4*i)).toInt
def index (m : Inputs) (read : Reader) (r : Role) (i : Nat) : Nat := (raw m read r i).toNat
def page (m : Inputs) (read : Reader) (b n : Nat) : Nat :=
  index m read .pages (index m read .request b*columns m+n)

/-- Exactly the original checks before metadata contents and the empty branch.
Tensor status and nonnegative ordinary Torch strides are API observations. -/
structure Header (m : Inputs) : Prop where
  ranks : (m .probability).shape.length=2 ∧ (m .value).shape.length=3 ∧
    (m .output).shape.length=3 ∧ (m .pages).shape.length=2 ∧
    ∀ r∈[Role.request,.start,.length],(m r).shape.length=1
  dimension : width m=16 ∨ width m=32 ∨ width m=64 ∨ width m=128
  headShape : 0<heads m ∧ 0<kvheads m ∧ heads m % kvheads m=0
  outputShape : (m .output).shape=[batches m,heads m,width m]
  vectors : ∀ r∈[Role.request,.start,.length],(m r).shape=[batches m]
  floating : ∀ r∈[Role.probability,.value,.output],(m r).dtype=.floating .fp32
  integer : ∀ r∈integers,(m r).dtype=.other "int32"
  cuda : (m .probability).device.kind="cuda"
  device : ∀ r,(m r).device=(m .probability).device
  ordinary : ∀ r,(m r).isNeg=Bool.false ∧ (m r).isContiguous=Bool.true
  extents : ∀ r,(m r).numel≤2^31-128
  padded : (m .probability).numel+128<2^31 ∧
    (m .pages).numel+tokens m+128<2^31 ∧ columns m+tokens m+128<2^31

def Contents (m : Inputs) (read : Reader) : Prop :=
  ∀ b<batches m,
    (0≤raw m read .request b ∧ raw m read .request b<(requests m : Int)) ∧
    (0≤raw m read .length b ∧ raw m read .length b≤(columns m : Int)) ∧
    (0≤raw m read .start b ∧ raw m read .start b+raw m read .length b≤(tokens m : Int)) ∧
    (∀ n < index m read .length b,
      0≤raw m read .pages (index m read .request b*columns m+n) ∧
      raw m read .pages (index m read .request b*columns m+n)<(cache m : Int))

def Apart (m : Inputs) (a b : Role) : Prop :=
  (m a).numel=0 ∨ (m b).numel=0 ∨
  base m a+4*(m a).numel≤base m b ∨ base m b+4*(m b).numel≤base m a

/-- The original live-call pointer and view-byte checks. Read/read aliases of
the same dtype remain admitted; this does not require allocation separation. -/
structure Physical (m : Inputs) : Prop where
  mixed : ∀ f∈[Role.probability,.value],∀ i∈integers,Apart m f i
  backing : ∀ r,base m r+4*(m r).numel≤2^64
  alignment : ∀ r,0<(m r).numel → base m r%4=0
  output : ∀ r∈reads,Apart m .output r

structure OriginalContract (m : Inputs) (read : Reader) : Prop where
  header : Header m
  contents : Contents m read
  physical : 0<batches m → Physical m

/-- The sole additional premise applies only when a grid is submitted. -/
structure InputContract (m : Inputs) (read : Reader) extends OriginalContract m read : Prop where
  gridY : 0<batches m → heads m≤65535

inductive InputError where | tensors | header | contents | gridY | physical
  deriving DecidableEq, Repr
open Classical in
def check (m : Inputs) (read : Reader) : Except InputError Bool :=
  if ¬Header m then .error .header
  else if ¬Contents m read then .error .contents
  else if batches m=0 then .ok Bool.false
  else if ¬heads m≤65535 then .error .gridY
  else if ¬Physical m then .error .physical
  else .ok Bool.true

def checkArguments (m : Option Inputs) (read : Reader) : Except InputError Bool :=
  match m with | none => .error .tensors | some m => check m read

theorem exact_domain (m : Inputs) (read : Reader) :
    InputContract m read ↔ OriginalContract m read ∧ (0<batches m → heads m≤65535) :=
  ⟨fun h => ⟨h.toOriginalContract,h.gridY⟩,fun h => ⟨h.1,h.2⟩⟩

theorem check_launch_iff (m : Inputs) (read : Reader) :
    check m read=.ok Bool.true ↔ 0<batches m ∧ InputContract m read := by
  classical
  constructor
  · intro accepted
    unfold check at accepted
    split_ifs at accepted <;> try contradiction
    rename_i hh hc hb hy hp
    exact ⟨by omega,⟨⟨by tauto,by tauto,fun _ => by tauto⟩,fun _ => by omega⟩⟩
  · rintro ⟨live,c⟩
    simp [check,c.header,c.contents,show batches m≠0 by omega,c.gridY live,c.physical live]

theorem check_empty_iff (m : Inputs) (read : Reader) :
    check m read=.ok Bool.false ↔ batches m=0 ∧ InputContract m read := by
  classical
  constructor
  · intro accepted
    unfold check at accepted
    split_ifs at accepted <;> try contradiction
    rename_i hh hc hb
    exact ⟨hb,⟨⟨by tauto,by tauto,fun live => by omega⟩,fun live => by omega⟩⟩
  · rintro ⟨empty,c⟩
    simp [check,c.header,c.contents,empty]

theorem python_checks_iff (m : Inputs) (read : Reader) :
    (∃ launch,check m read=.ok launch) ↔ InputContract m read := by
  constructor
  · rintro ⟨launch,h⟩
    cases launch
    · exact (check_empty_iff m read).mp h |>.2
    · exact (check_launch_iff m read).mp h |>.2
  · intro h
    by_cases empty : batches m=0
    · exact ⟨Bool.false,(check_empty_iff m read).mpr ⟨empty,h⟩⟩
    · exact ⟨Bool.true,(check_launch_iff m read).mpr ⟨by omega,h⟩⟩

theorem empty_domain_unchanged (m : Inputs) (read : Reader) (empty : batches m=0) :
    InputContract m read ↔ OriginalContract m read := by
  rw [exact_domain]
  simp [empty]

#axiomsClean exact_domain
#axiomsClean python_checks_iff
#axiomsClean empty_domain_unchanged
end
end VeriTile.Bench.Repairs.TokenAttnReduceVGridYChecked
