import bench.tritonbench_g.chunk_gated_attention.Sources.HostShape
import bench.tritonbench_g.chunk_gated_attention.Sources.HostConfig
import bench.tritonbench_g.chunk_gla_simple.Sources.ByteMemory

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Admission
open VeriTile Triton
abbrev Tensor := ChunkGlaSimple.Admission.Tensor
abbrev Address := ChunkGlaSimple.Admission.Address

def Block (n : Nat) : Prop := 16≤n ∧ n &&& (n-1)=0
instance (n : Nat) : Decidable (Block n) := inferInstanceAs (Decidable (_ ∧ _))
def Fits (n : Nat) : Prop := 4*n<2^63
instance (n : Nat) : Decidable (Fits n) := inferInstanceAs (Decidable (_<_))

structure Inner where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  gateK : Bool
  query : Tensor
  key : Tensor
  value : Tensor
  gate : Tensor
  seed : Option Tensor
  final : Option Tensor

def Inner.config (r : Inner) : HostFP.Inner where
  dot:=.fp32
  gate:=.fp32
  history:=.fp32
  seed:=.fp32
  final:=.fp32
  k:="k"
  v:="v"
  g:="g"
  h:="h"
  h0:="h0"
  ht:="ht"
  heads:=r.B*r.H
  T:=r.T
  K:=r.K
  V:=r.V
  BT:=r.BT
  BK:=r.BK
  BV:=r.BV
  gateK:=r.gateK
  hasSeed:=r.seed.isSome
  storeFinal:=r.final.isSome

def Inner.tensors (r : Inner) := [r.query,r.key,r.value,r.gate]++r.seed.toList++r.final.toList
def Inner.checks (r : Inner) : List (Tensor×List Nat) :=
  [(r.query,[r.B,r.H,r.T,r.K]),(r.key,[r.B,r.H,r.T,r.K]),
   (r.value,[r.B,r.H,r.T,r.V]),(r.gate,[r.B,r.H,r.T,if r.gateK then r.K else r.V])] ++
  (r.seed.toList++r.final.toList).map (fun t=>(t,[r.B,r.H,r.K,r.V]))
def Inner.extents (r : Inner) : List Nat :=
  [r.B*r.H*r.T*r.K,r.B*r.H*r.T*r.V,r.B*r.H*r.K*r.V,r.B*r.H*r.config.NT*r.K*r.V]
def Inner.paddedExtents (r : Inner) : List Nat :=
  let c:=r.config
  [c.heads*c.PT*c.PK,c.heads*c.PT*c.PV,c.heads*c.NT*c.PK*c.PV,c.heads*c.PK*c.PV]
def Inner.nonempty (r : Inner) : Prop := r.B*r.H*r.T*r.K*r.V≠0
instance (r : Inner) : Decidable r.nonempty := inferInstanceAs (Decidable (_≠_))

/-- Natural dimensions and Bool gate mode are the results of the Python
builtin-type/nonnegative checks. All tensor and finite-span checks below are
the executable FP32 wrapper guards, including its empty branch. -/
def Inner.Contract (r : Inner) : Prop :=
  Block r.BT ∧ Block r.BK ∧ Block r.BV ∧
  max (max (r.BT*r.BK) (r.BT*r.BV)) (r.BK*r.BV)≤2^20 ∧
  (∀ p∈r.checks,p.1.Valid r.query.device p.2) ∧
  (∀ n∈r.extents,Fits n) ∧ (∀ t∈r.tensors,Address t) ∧
  (r.nonempty → ∀ n∈r.paddedExtents,Fits n)
instance (r : Inner) : Decidable r.Contract := inferInstanceAs (Decidable (_ ∧ _))
def Inner.check (r : Inner) : Bool := decide r.Contract
theorem Inner.check_iff (r : Inner) : r.check=Bool.true ↔ r.Contract := by simp [Inner.check]

theorem Inner.blocks (r : Inner) (checked : r.check=Bool.true) : 0<r.BT ∧ 0<r.BK ∧ 0<r.BV := by
  have h:=(r.check_iff).mp checked
  exact ⟨by have:=h.1.1;omega,by have:=h.2.1.1;omega,by have:=h.2.2.1.1;omega⟩

theorem Inner.padded_fits (r : Inner) (checked : r.check=Bool.true) (nonempty : r.nonempty) :
    Fits (r.config.heads*r.config.PT*r.config.PK) ∧ Fits (r.config.heads*r.config.PT*r.config.PV) ∧
    Fits (r.config.heads*r.config.NT*r.config.PK*r.config.PV) ∧ Fits (r.config.heads*r.config.PK*r.config.PV) := by
  have bounds:=((r.check_iff).mp checked).2.2.2.2.2.2.2 nonempty
  exact ⟨bounds _ (by simp [Inner.paddedExtents]),bounds _ (by simp [Inner.paddedExtents]),
    bounds _ (by simp [Inner.paddedExtents]),bounds _ (by simp [Inner.paddedExtents])⟩

theorem Inner.final_valid (r : Inner) (checked : r.check=Bool.true) (tensor : Tensor) (h : r.final=some tensor) :
    tensor.Valid r.query.device [r.B,r.H,r.K,r.V] := by
  have metadata:=((r.check_iff).mp checked).2.2.2.2.1
  exact metadata (tensor,[r.B,r.H,r.K,r.V]) (by simp [Inner.checks,h])

structure Prefix where
  B : Nat
  H : Nat
  T : Nat
  S : Nat
  BT : Nat
  input : Tensor

def Prefix.config (r : Prefix) (BS : Nat) : HostFP.Cumulative where
  dtype:=.fp32
  input:="s"
  output:="o"
  heads:=r.B*r.H
  T:=r.T
  S:=r.S
  BT:=r.BT
  BS:=BS
def Prefix.nonempty (r : Prefix) : Prop := r.B*r.H*r.T*r.S≠0
instance (r : Prefix) : Decidable r.nonempty := inferInstanceAs (Decidable (_≠_))
def Prefix.Contract (r : Prefix) : Prop :=
  Block r.BT ∧ max (r.BT*r.BT) (r.BT*64)≤2^20 ∧
  r.input.Valid r.input.device [r.B,r.H,r.T,r.S] ∧
  (r.nonempty → Fits (r.B*r.H*(HostFP.padded r.T r.BT)*(HostFP.padded r.S 64)) ∧ Address r.input)
instance (r : Prefix) : Decidable r.Contract := inferInstanceAs (Decidable (_ ∧ _))
def Prefix.check (r : Prefix) : Bool := decide r.Contract
theorem Prefix.check_iff (r : Prefix) : r.check=Bool.true ↔ r.Contract := by simp [Prefix.check]

#axiomsClean Inner.padded_fits
#axiomsClean Prefix.check_iff
end VeriTile.Bench.Repairs.ChunkGatedAttention.Admission
