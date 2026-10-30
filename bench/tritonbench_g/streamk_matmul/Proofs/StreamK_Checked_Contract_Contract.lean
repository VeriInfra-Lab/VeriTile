import VeriTile.Torch.Tensor
import bench.tritonbench_g.streamk_matmul.Sources.Foundation

namespace VeriTile.Bench.Repairs.StreamK.Checked.Contract
open VeriTile.Torch
open VeriTile.Bench.TritonBenchG.StreamK
open Whole001

structure Config where
  programs : Int
  blockM : Nat
  blockN : Nat
  blockK : Nat
  twoTiles : Bool
  stages : Nat
  warps : Nat

def rows (t : TensorMetadata) := t.shape[0]?.getD 0
def columns (t : TensorMetadata) := t.shape[1]?.getD 0
def plan (a b : TensorMetadata) (c : Config) : HostInput :=
  ⟨rows a,columns b,columns a,c.blockM,c.blockN,c.blockK,c.programs.toNat,c.twoTiles⟩
def Block (n : Nat) : Prop := 0<n ∧ n≤2^20 ∧ n &&& (n-1)=0
instance (n : Nat) : Decidable (Block n) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))
def Profile (c : Config) : Prop :=
  Block c.blockM ∧ Block c.blockN ∧ Block c.blockK ∧ 8≤c.blockK ∧ c.blockK%8=0 ∧
  0<c.stages ∧ c.warps∈[1,2,4,8,16,32]
instance (c : Config) : Decidable (Profile c) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
def TensorInputs (a b : TensorMetadata) : Prop :=
  a.layout=.strided ∧ b.layout=.strided ∧ a.shape.length=2 ∧ b.shape.length=2 ∧
  a.dtype=.floating .fp32 ∧ b.dtype=.floating .fp32 ∧
  a.device.kind="cuda" ∧ b.device.kind="cuda" ∧ b.device=a.device ∧
  a.isContiguous=Bool.true ∧ b.isContiguous=Bool.true ∧
  a.isNeg=Bool.false ∧ b.isNeg=Bool.false ∧ columns a=rows b
instance (a b : TensorMetadata) : Decidable (TensorInputs a b) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
def Empty (a b : TensorMetadata) : Prop := rows a=0 ∨ columns b=0
instance (a b : TensorMetadata) : Decidable (Empty a b) := inferInstanceAs (Decidable (_ ∨ _))
def Control (h : HostInput) : Prop :=
  h.m+h.bm-1<2^31 ∧ h.n+h.bn-1<2^31 ∧ h.k+h.bk-1<2^31 ∧
  8*ceilDiv h.n h.bn<2^31 ∧ Plan.tiles h<2^31 ∧ h.programs<2^31
instance (h : HostInput) : Decidable (Control h) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
def advancedK (h : HostInput) := ((Plan.select h).iterations+1)*h.bk
def pointerA (h : HostInput) := ceilDiv h.m h.bm*h.bm*h.k+advancedK h
def pointerB (h : HostInput) := advancedK h*h.n+ceilDiv h.n h.bn*h.bn
def pointerC (h : HostInput) := ceilDiv h.m h.bm*h.bm*max h.n 1+ceilDiv h.n h.bn*h.bn
def ByteEnvelope (h : HostInput) : Prop :=
  4*pointerA h<2^63 ∧ 4*pointerB h<2^63 ∧ 4*pointerC h<2^63 ∧
  4*(h.programs*h.bm*h.bn)<2^63 ∧ 4*(h.m*h.k)<2^63 ∧ 4*(h.k*h.n)<2^63 ∧ 4*(h.m*h.n)<2^63
instance (h : HostInput) : Decidable (ByteEnvelope h) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _ ∧ _))
def StreamEnvelope (h : HostInput) : Prop :=
  (Plan.select h).streamTiles*(Plan.select h).iterations+(Plan.select h).iterations<2^63
instance (h : HostInput) : Decidable (StreamEnvelope h) := inferInstanceAs (Decidable (_<_))
def Window (t : TensorMetadata) : Prop :=
  t.numel=0 ∨ (t.dataPtr.toNat%4=0 ∧ t.dataPtr.toNat+4*t.numel≤2^64)
instance (t : TensorMetadata) : Decidable (Window t) := inferInstanceAs (Decidable (_ ∨ (_ ∧ _)))

structure InputContract (a b : TensorMetadata) (c : Config) : Prop where
  inputs : TensorInputs a b
  profile : Profile c
  execution : Empty a b ∨ (Control (plan a b c) ∧ StreamEnvelope (plan a b c) ∧
    ByteEnvelope (plan a b c) ∧ Window a ∧ Window b)

inductive Error where | tensor | metadata | profile | control | stream | bytes | window
  deriving DecidableEq,Repr
def checkTensorInputs (a b : TensorMetadata) (c : Config) : Except Error Unit :=
  if ¬TensorInputs a b then .error .metadata
  else if ¬Profile c then .error .profile
  else if Empty a b then .ok ()
  else if ¬Control (plan a b c) then .error .control
  else if ¬StreamEnvelope (plan a b c) then .error .stream
  else if ¬ByteEnvelope (plan a b c) then .error .bytes
  else if ¬(Window a ∧ Window b) then .error .window
  else .ok ()
def checkInputs (a b : ArgumentMetadata) (c : Config) : Except Error Unit :=
  match a,b with
  | .tensor a,.tensor b => checkTensorInputs a b c
  | _,_ => .error .tensor
def AdmittedArguments (a b : ArgumentMetadata) (c : Config) : Prop :=
  match a,b with
  | .tensor a,.tensor b => InputContract a b c
  | _,_ => False
end VeriTile.Bench.Repairs.StreamK.Checked.Contract
