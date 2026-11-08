import bench.tritonbench_g.attn_fwd_causal.Kernels
import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Source

namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
open VeriTile.Triton FP.Structural
noncomputable section

structure Strides where
  (batch head row dim : Nat)
  deriving DecidableEq
structure Parameters where
  (Q K V QS KS Out : RegionName)
  (batches heads length : Nat)

def Parameters.q (p : Parameters) : Strides :=
  ⟨max p.heads 1 * (max p.length 1 * 128), max p.length 1 * 128, 128, 1⟩
abbrev Parameters.k := Parameters.q
abbrev Parameters.v := Parameters.q
abbrev Parameters.o := Parameters.q

def kernel (p : Parameters) : ComputeKernel :=
  TypedSource.repairedKernel p.Q p.K p.V p.QS p.KS p.Out
    p.q.batch p.q.head p.q.row p.q.dim p.k.batch p.k.head p.k.row p.k.dim
    p.v.batch p.v.head p.v.row p.v.dim p.o.batch p.o.head p.o.row p.o.dim
    p.batches p.heads p.length 128 128 64

/-- Private total typed adapter for execution lemmas. Public endpoints must
instantiate both products using explicit fixed-instruction chains. -/
structure Dot (α : Type) where
  apply : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) →
    Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def withDot {α : Type} (M : Algebra α) (D : Dot α) : Algebra α :=
  {M with
    dot := fun p {_} {_} {_} {_} a b => some (D.apply p .real .real .real {} a b)
    dotWithOptions := fun p options {_} {_} {_} {_} a b => some (D.apply p .real .real .real options a b)
    dotTyped := fun p l r o options {_} {_} {_} {_} a b => some (D.apply p l r o options a b)}

def parts (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernel p).surfaceBody "__veritile_statement_jit_local_12_start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def diagonal (p : Parameters) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (kernel p).surfaceBody "__veritile_statement_jit_local_29_start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def middle (p : Parameters) : List ComputeStmt :=
  (VeriTile.Triton.Source.dynamicRange? (parts p).after "__veritile_statement_jit_local_29_start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩ |>.before

/-- Proof views of the two hygienically expanded helper invocations. -/
def loopParts (p : Parameters) (phase : Bool) : VeriTile.Triton.Source.DynamicRange :=
  if phase then diagonal p else parts p

def workName (phase : Bool) (role : String) : String :=
  match role with
  | "acc"=>if phase then "__veritile_statement_jit_17" else "__veritile_statement_jit_0"
  | "l_i"=>if phase then "__veritile_statement_jit_18" else "__veritile_statement_jit_1"
  | "m_i"=>if phase then "__veritile_statement_jit_19" else "__veritile_statement_jit_2"
  | "q"=>if phase then "__veritile_statement_jit_20" else "__veritile_statement_jit_3"
  | "q_scale"=>if phase then "__veritile_statement_jit_21" else "__veritile_statement_jit_4"
  | "K_ptrs"=>if phase then "__veritile_statement_jit_22" else "__veritile_statement_jit_5"
  | "K_scale_ptr"=>if phase then "__veritile_statement_jit_23" else "__veritile_statement_jit_6"
  | "V_ptrs"=>if phase then "__veritile_statement_jit_24" else "__veritile_statement_jit_7"
  | "start_m"=>if phase then "__veritile_statement_jit_25" else "__veritile_statement_jit_8"
  | "offs_m"=>if phase then "__veritile_statement_jit_26" else "__veritile_statement_jit_9"
  | "offs_n"=>if phase then "__veritile_statement_jit_27" else "__veritile_statement_jit_10"
  | "N_CTX"=>if phase then "__veritile_statement_jit_28" else "__veritile_statement_jit_11"
  | other=>(if phase then "__veritile_statement_jit_local_29_" else "__veritile_statement_jit_local_12_")++other

def stop {α : Type} (p : Parameters) (s : State α) : Nat := min ((s.pids 0+1)*128) p.length

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source
