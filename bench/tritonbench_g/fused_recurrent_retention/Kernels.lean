import Mathlib
import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! Complete ordinary FP32 forward/backward source and the existing compareRight
candidate. Scale is an actual external FP32 launch register; no placeholder or
statement deletion constructs the authoritative source. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Surface
open VeriTile.Triton
set_option maxRecDepth 24000

def forward (dtype out_dtype : FloatDType)
        (q k v o initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
      triton {
        tl.ports inputs=[q, k, v, initial_state], outputs=[o, final_state]
        tl.region q = tl.float32, k = tl.float32, v = tl.float32, o = tl.float32,
          initial_state = tl.float32, final_state = tl.float32
        tl.param scale : tl.float32
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
      
        mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK)
        mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(DV)
        mask_kv = mask_bk[None, :] & mask_bv[:, None]
      
        h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
      
        if USE_INITIAL_STATE {
          p_init_s = initial_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
          h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        }
      
        for _t in range($(0), $(T), $(1)) {
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
      
          h = b_b * h + _k[None, :] * _v[:, None]
          _o = h * _q[None, :]
          _o = tl.sum(_o, axis=1)
          tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
      
          p_q += $(DK)
          p_k += $(DK)
          p_o += $(DV)
          p_v += $(DV)
        }
      
        if STORE_FINAL_STATE {
          p_final_s = final_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
          tl.store(p_final_s, (h).to(p_final_s.dtype.element_ty), mask=mask_kv)
        }
      }

def backward (dtype out_dtype : FloatDType)
        (q k v do_ dq dk dv initial_state d_final_state : RegionName)
        (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v : Nat)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
      triton {
        tl.ports inputs=[q, k, v, do_, initial_state, d_final_state], outputs=[dq, dk, dv]
        tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32,
          dq = tl.float32, dk = tl.float32, dv = tl.float32, initial_state = tl.float32, d_final_state = tl.float32
        tl.param scale : tl.float32
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_do = do_ + (i_bh // $(H)) * $(s_do_b) + i_h * $(s_do_h) + (i_v * $(BV) + tl.arange(0, $(BV))) * $(s_do_d)
        p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
      
        mask_bk = (i_k * $(BK) + tl.arange(0, $(BK))) < $(DK)
        mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(DV)
      
        h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
      
        if USE_INITIAL_STATE {
          mask_kv = mask_bk[:, None] & mask_bv[None, :]
          p_init_s = initial_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
          h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        }
      
        for i in range($(0), $(T), $(1)) {
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
      
          h = b_b * h + _k[:, None] * _v[None, :]
          _d_q = h * _do[None, :]
          d_q = tl.sum(_d_q, axis=1) * scale
          tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
      
          p_k += $(DK)
          p_do += $(s_do_t)
          p_v += $(DV)
          p_dq += $(DK)
        }
      
        tl.debug_barrier()
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_do = do_ + (i_bh // $(H)) * $(s_do_b) + i_h * $(s_do_h) + (i_v * $(BV) + tl.arange(0, $(BV))) * $(s_do_d) + ($(T) - 1) * $(s_do_t)
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_FINAL_GRAD {
          p_df = d_final_state + (i_bh // $(H)) * $(s_df_b) + i_h * $(s_df_h) +
            (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(s_df_k) +
            (i_v * $(BV) + tl.arange(0, $(BV))[None, :]) * $(s_df_v)
          d_h = tl.load(p_df, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
        }
      
        for _t in range($(0), $(T), $(1)) {
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          d_h += _q[:, None] * _do[None, :]
          d_k = tl.sum(d_h * _v[None, :], axis=1)
          d_v = tl.sum(d_h * _k[:, None], axis=0)
      
          d_h *= b_b
          tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
          tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
      
          p_do -= $(s_do_t)
          p_q -= $(DK)
          p_k -= $(DK)
          p_v -= $(DV)
          p_dk -= $(DK)
          p_dv -= $(DV)
        }
      }

/-- Existing comparison-orientation candidate, with the full original body. -/
def forwardCompareRight (dtype out_dtype : FloatDType)
        (q k v o initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) : ComputeKernel :=
      triton {
        tl.ports inputs=[q, k, v, initial_state], outputs=[o, final_state]
        tl.region q = tl.float32, k = tl.float32, v = tl.float32, o = tl.float32,
          initial_state = tl.float32, final_state = tl.float32
        tl.param scale : tl.float32
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_o = o + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
      
        mask_bk = $(DK) > (i_k * $(BK) + tl.arange(0, $(BK)))
        mask_bv = (i_v * $(BV) + tl.arange(0, $(BV))) < $(DV)
        mask_kv = mask_bk[None, :] & mask_bv[:, None]
      
        h = tl.zeros([$(BV), $(BK)], dtype=tl.float32)
      
        if USE_INITIAL_STATE {
          p_init_s = initial_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
          h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        }
      
        for _t in range($(0), $(T), $(1)) {
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
      
          h = b_b * h + _k[None, :] * _v[:, None]
          _o = h * _q[None, :]
          _o = tl.sum(_o, axis=1)
          tl.store(p_o, (_o).to(p_o.dtype.element_ty), mask=mask_bv)
      
          p_q += $(DK)
          p_k += $(DK)
          p_o += $(DV)
          p_v += $(DV)
        }
      
        if STORE_FINAL_STATE {
          p_final_s = final_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[None, :]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[:, None])
          tl.store(p_final_s, (h).to(p_final_s.dtype.element_ty), mask=mask_kv)
        }
      }

def backwardCompareRight (dtype out_dtype : FloatDType)
        (q k v do_ dq dk dv initial_state d_final_state : RegionName)
        (s_do_b s_do_h s_do_t s_do_d s_df_b s_df_h s_df_k s_df_v : Nat)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T : Nat) (scale : ℝ)
        (BK BV DK DV : Nat) (USE_INITIAL_STATE USE_FINAL_GRAD : Bool) : ComputeKernel :=
      triton {
        tl.ports inputs=[q, k, v, do_, initial_state, d_final_state], outputs=[dq, dk, dv]
        tl.region q = tl.float32, k = tl.float32, v = tl.float32, do_ = tl.float32,
          dq = tl.float32, dk = tl.float32, dv = tl.float32, initial_state = tl.float32, d_final_state = tl.float32
        tl.param scale : tl.float32
        i_v = tl.program_id(0)
        i_k = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV))
        p_do = do_ + (i_bh // $(H)) * $(s_do_b) + i_h * $(s_do_h) + (i_v * $(BV) + tl.arange(0, $(BV))) * $(s_do_d)
        p_dq = dq + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK))
      
        mask_bk = $(DK) > (i_k * $(BK) + tl.arange(0, $(BK)))
        mask_bv = i_v * $(BV) + tl.arange(0, $(BV)) < $(DV)
      
        h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
      
        if USE_INITIAL_STATE {
          mask_kv = mask_bk[:, None] & mask_bv[None, :]
          p_init_s = initial_state + i_bh * $(DK) * $(DV) +
            (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(DV) +
            (i_v * $(BV) + tl.arange(0, $(BV))[None, :])
          h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)
        }
      
        for i in range($(0), $(T), $(1)) {
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
      
          h = b_b * h + _k[:, None] * _v[None, :]
          _d_q = h * _do[None, :]
          d_q = tl.sum(_d_q, axis=1) * scale
          tl.store(p_dq, (d_q).to(p_dq.dtype.element_ty), mask=mask_bk)
      
          p_k += $(DK)
          p_do += $(s_do_t)
          p_v += $(DV)
          p_dq += $(DK)
        }
      
        tl.debug_barrier()
      
        p_q = q + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_k = k + i_bh * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_do = do_ + (i_bh // $(H)) * $(s_do_b) + i_h * $(s_do_h) + (i_v * $(BV) + tl.arange(0, $(BV))) * $(s_do_d) + ($(T) - 1) * $(s_do_t)
        p_v = v + i_bh * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        p_dk = dk + (i_bh + i_v * $(B) * $(H)) * $(s_qk_h) + i_k * $(BK) + tl.arange(0, $(BK)) + ($(T) - 1) * $(DK)
        p_dv = dv + (i_bh + i_k * $(B) * $(H)) * $(s_vo_h) + i_v * $(BV) + tl.arange(0, $(BV)) + ($(T) - 1) * $(DV)
        d_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_FINAL_GRAD {
          p_df = d_final_state + (i_bh // $(H)) * $(s_df_b) + i_h * $(s_df_h) +
            (i_k * $(BK) + tl.arange(0, $(BK))[:, None]) * $(s_df_k) +
            (i_v * $(BV) + tl.arange(0, $(BV))[None, :]) * $(s_df_v)
          d_h = tl.load(p_df, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)
        }
      
        for _t in range($(0), $(T), $(1)) {
          _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
          _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
          _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
          _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
          d_h += _q[:, None] * _do[None, :]
          d_k = tl.sum(d_h * _v[None, :], axis=1)
          d_v = tl.sum(d_h * _k[:, None], axis=0)
      
          d_h *= b_b
          tl.store(p_dk, (d_k).to(p_dk.dtype.element_ty), mask=mask_bk)
          tl.store(p_dv, (d_v).to(p_dv.dtype.element_ty), mask=mask_bv)
      
          p_do -= $(s_do_t)
          p_q -= $(DK)
          p_k -= $(DK)
          p_v -= $(DV)
          p_dk -= $(DK)
          p_dv -= $(DV)
        }
      }

end VeriTile.Bench.Repairs.FusedRecurrentRetention.Surface

/-! Pure source declarations, consumed unchanged by the proof closure. Exact
per-program forward/reverse carries, source grid stores and host reductions. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators

structure Arithmetic (α : Type) where
  zero : α
  one : α
  negFive : α
  fromNat : Nat → α
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  exp2 : α → α
  scale : Nat → α

def realArithmetic : Arithmetic ℝ where
  zero := 0
  one := 1
  negFive := -5
  fromNat := Nat.cast
  add := (·+·)
  sub := (·-·)
  mul := (·*·)
  fma := fun x y z => x*y+z
  exp2 := fun x => Real.exp (x*Real.log 2)
  scale := fun n => (Real.sqrt (n : ℝ))⁻¹

def decay (a : Arithmetic α) (head : Nat) : α :=
  a.sub a.one (a.exp2 (a.sub a.negFive (a.mul (a.fromNat head) a.one)))

def foldN (step : Nat → β → β) : Nat → β → β
  | 0,z => z
  | n+1,z => step n (foldN step n z)

inductive DotPlan where
  | product (index : Nat)
  | add (left right : DotPlan)
  | fma (index : Nat) (tail : DotPlan)
  deriving Repr

def DotPlan.leaves : DotPlan → List Nat
  | .product i => [i]
  | .add l r => l.leaves++r.leaves
  | .fma i t => i::t.leaves

def DotPlan.eval (a : Arithmetic α) (x y : Nat → α) : DotPlan → α
  | .product i => a.mul (x i) (y i)
  | .add l r => a.add (l.eval a x y) (r.eval a x y)
  | .fma i t => a.fma (x i) (y i) (t.eval a x y)

def DotPlan.Covers (p : DotPlan) (n : Nat) : Prop := p.leaves.Perm (List.range n)

inductive SumPlan where
  | zero
  | input (index : Nat)
  | add (left right : SumPlan)
  deriving Repr

def SumPlan.leaves : SumPlan → List Nat
  | .zero => []
  | .input i => [i]
  | .add l r => l.leaves++r.leaves

def SumPlan.eval (a : Arithmetic α) (x : Nat → α) : SumPlan → α
  | .zero => a.zero
  | .input i => x i
  | .add l r => a.add (l.eval a x) (r.eval a x)

def SumPlan.Covers (p : SumPlan) (n : Nat) : Prop := p.leaves.Perm (List.range n)

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  DK : Nat
  DV : Nat
  deriving DecidableEq

def tileCount (n width : Nat) : Nat := (n-1)/width+1

def Shape.BK (s : Shape) := min s.DK 32

def Shape.BV (s : Shape) := min s.DV 32

def Shape.NK (s : Shape) := tileCount s.DK s.BK

def Shape.NV (s : Shape) := tileCount s.DV s.BV

def Shape.BH (s : Shape) := s.B*s.H

def Shape.Admitted (s : Shape) : Prop :=
  0<s.DK ∧ 0<s.DV ∧ s.BK∈({1,2,4,8,16,32} : Finset Nat) ∧
    s.BV∈({1,2,4,8,16,32} : Finset Nat)

def rowAddress (s : Shape) (width head time channel : Nat) : Nat :=
  (head*s.T+time)*width+channel

def stateOffset (s : Shape) (head key value : Nat) : Nat :=
  (head*s.DK+key)*s.DV+value

def partialOffset (s : Shape) (width split head time channel : Nat) : Nat :=
  ((split*s.BH+head)*s.T+time)*width+channel

structure Inputs (α : Type) where
  q : Nat → Nat → Nat → α
  k : Nat → Nat → Nat → α
  v : Nat → Nat → Nat → α
  do_ : Nat → Nat → Nat → α
  seed : Nat → Nat → Nat → α
  finalGrad : Nat → Nat → Nat → α

def keyLoad (a : Arithmetic α) (s : Shape) (f : Nat → Nat → Nat → α) (h t k : Nat) : α :=
  if k<s.DK then f h t k else a.zero

def valueLoad (a : Arithmetic α) (s : Shape) (f : Nat → Nat → Nat → α) (h t v : Nat) : α :=
  if v<s.DV then f h t v else a.zero

def stateLoad (a : Arithmetic α) (s : Shape) (f : Nat → Nat → Nat → α) (h k v : Nat) : α :=
  if k<s.DK ∧ v<s.DV then f h k v else a.zero

structure Plans where
  output : DotPlan
  dq : DotPlan
  dk : DotPlan
  dv : DotPlan
  sumO : Nat → SumPlan
  sumQ : Nat → SumPlan
  sumK : Nat → SumPlan
  sumV : Nat → SumPlan

structure Plans.Valid (p : Plans) (s : Shape) : Prop where
  output : p.output.Covers s.BK
  dq : p.dq.Covers s.BV
  dk : p.dk.Covers s.BV
  dv : p.dv.Covers s.BK
  sumO : ∀ i,(p.sumO i).Covers s.NK
  sumQ : ∀ i,(p.sumQ i).Covers s.NV
  sumK : ∀ i,(p.sumK i).Covers s.NV
  sumV : ∀ i,(p.sumV i).Covers s.NK

abbrev Address := Nat × Int

abbrev Memory (α : Type) := Address → α

structure Tensor4 where
  storage : Nat
  offset : Nat
  capacity : Nat
  device : Nat
  strides : Fin 4 → Nat
  deriving DecidableEq

def Tensor4.address (a : Tensor4) (b h t d : Nat) : Address :=
  (a.storage,(a.offset+b*a.strides 0+h*a.strides 1+t*a.strides 2+d*a.strides 3 : Nat))

structure InputBuffers where
  q : Tensor4
  k : Tensor4
  v : Tensor4
  do_ : Tensor4
  seed : Tensor4
  finalGrad : Tensor4

def InputBuffers.qAddress (b : InputBuffers) (s : Shape) (h t k : Nat) : Address :=
  (b.q.storage,(b.q.offset+h*b.q.strides 1+t*s.DK+k : Nat))

def InputBuffers.kAddress (b : InputBuffers) (s : Shape) (h t k : Nat) : Address :=
  (b.k.storage,(b.k.offset+h*b.q.strides 1+t*s.DK+k : Nat))

def InputBuffers.vAddress (b : InputBuffers) (s : Shape) (h t v : Nat) : Address :=
  (b.v.storage,(b.v.offset+h*b.v.strides 1+t*s.DV+v : Nat))

def InputBuffers.doAddress (b : InputBuffers) (s : Shape) (h t v : Nat) : Address :=
  b.do_.address (h/s.H) (h%s.H) t v

def InputBuffers.seedAddress (b : InputBuffers) (s : Shape) (h k v : Nat) : Address :=
  (b.seed.storage,(b.seed.offset+stateOffset s h k v : Nat))

def InputBuffers.finalAddress (b : InputBuffers) (s : Shape) (h k v : Nat) : Address :=
  b.finalGrad.address (h/s.H) (h%s.H) k v

def InputBuffers.observe (b : InputBuffers) (s : Shape) (m : Memory α) : Inputs α where
  q := fun h t k => m (b.qAddress s h t k)
  k := fun h t k => m (b.kAddress s h t k)
  v := fun h t v => m (b.vAddress s h t v)
  do_ := fun h t v => m (b.doAddress s h t v)
  seed := fun h k v => m (b.seedAddress s h k v)
  finalGrad := fun h k v => m (b.finalAddress s h k v)

def InputBuffers.storageIds (b : InputBuffers) : Finset Nat :=
  {b.q.storage,b.k.storage,b.v.storage,b.do_.storage,b.seed.storage,b.finalGrad.storage}

def writeWord (m : Memory α) (p : Address) (x : α) : Memory α :=
  fun q => if q=p then x else m q

def fill (storage : Nat) (value : Nat → α) (items : List Nat) (m : Memory α) : Memory α :=
  items.foldl (fun current (i : Nat) => writeWord current (storage,(i : Int)) (value i)) m

def fillRange (storage count : Nat) (value : Nat → α) (m : Memory α) : Memory α :=
  fill storage value (List.range count) m

structure Allocation (n : Nat) (caller : Set Nat) where
  storage : Fin n → Nat
  distinct : Function.Injective storage
  fresh : ∀ i,storage i∉caller

def flatChannel (width i : Nat) := i%width

def flatTime (s : Shape) (width i : Nat) := i/width%s.T

def flatHead (s : Shape) (width i : Nat) := i/width/s.T%s.BH

def flatSplit (s : Shape) (width i : Nat) := i/width/s.T/s.BH

def rowCount (s : Shape) (width : Nat) := s.BH*s.T*width

inductive PartialKind where
  | output | dq | dk | dv | finalState
  deriving DecidableEq

def PartialKind.width (kind : PartialKind) (s : Shape) : Nat :=
  if kind=.dq ∨ kind=.dk then s.DK else s.DV

structure ProgramId where
  iv : Nat
  ik : Nat
  head : Nat
  deriving DecidableEq

def ProgramId.Valid (p : ProgramId) (s : Shape) := p.iv<s.NV ∧ p.ik<s.NK ∧ p.head<s.BH

def ProgramId.key (p : ProgramId) (s : Shape) (lane : Nat) := p.ik*s.BK+lane

def ProgramId.value (p : ProgramId) (s : Shape) (lane : Nat) := p.iv*s.BV+lane

abbrev TileState (α : Type) := Nat → Nat → α

def programForwardCarry (a : Arithmetic α) (s : Shape) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (n : Nat) : TileState α :=
  foldN (fun t old lk lv => a.fma (decay a (p.head%s.H)) (old lk lv)
    (a.mul (keyLoad a s x.k p.head t (p.key s lk)) (valueLoad a s x.v p.head t (p.value s lv)))) n
    (fun lk lv => if seed then a.add a.zero (stateLoad a s x.seed p.head (p.key s lk) (p.value s lv)) else a.zero)

def programReverseCarry (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (p : ProgramId) (n : Nat) : TileState α :=
  foldN (fun r old lk lv =>
    let t := s.T-1-r
    a.mul (a.fma (a.mul (keyLoad a s x.q p.head t (p.key s lk)) (a.scale s.DK))
      (valueLoad a s x.do_ p.head t (p.value s lv)) (old lk lv)) (decay a (p.head%s.H))) n
    (fun lk lv => if final then stateLoad a s x.finalGrad p.head (p.key s lk) (p.value s lv) else a.zero)

def programAdjoint (a : Arithmetic α) (s : Shape) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lk lv : Nat) : α :=
  a.fma (a.mul (keyLoad a s x.q p.head t (p.key s lk)) (a.scale s.DK))
    (valueLoad a s x.do_ p.head t (p.value s lv))
    (programReverseCarry a s x final p (s.T-1-t) lk lv)

def programO (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (t lv : Nat) : α :=
  plans.output.eval a (fun lk => programForwardCarry a s x seed p (t+1) lk lv)
    (fun lk => a.mul (keyLoad a s x.q p.head t (p.key s lk)) (a.scale s.DK))

def programQ (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (seed : Bool)
    (p : ProgramId) (t lk : Nat) : α :=
  a.mul (plans.dq.eval a (fun lv => programForwardCarry a s x seed p (t+1) lk lv)
    (fun lv => valueLoad a s x.do_ p.head t (p.value s lv))) (a.scale s.DK)

def programK (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lk : Nat) : α :=
  plans.dk.eval a (fun lv => programAdjoint a s x final p t lk lv)
    (fun lv => valueLoad a s x.v p.head t (p.value s lv))

def programV (a : Arithmetic α) (s : Shape) (plans : Plans) (x : Inputs α) (final : Bool)
    (p : ProgramId) (t lv : Nat) : α :=
  plans.dv.eval a (fun lk => programAdjoint a s x final p t lk lv)
    (fun lk => keyLoad a s x.k p.head t (p.key s lk))

def reduction (a : Arithmetic α) (plan : Nat → SumPlan) (input output count : Nat) (m : Memory α) : Memory α :=
  fillRange output count (fun offset => (plan offset).eval a (fun split => m (input,((split*count+offset : Nat) : Int)))) m

def sourceCell (a : Arithmetic α) (s : Shape) (p : Plans) (x : Inputs α)
    (seed final : Bool) (kind : PartialKind) (i : Nat) : α :=
  let width := kind.width s
  let h := flatHead s width i
  let t := flatTime s width i
  let channel := flatChannel width i
  let split := flatSplit s width i
  match kind with
  | .output => programO a s p x seed ⟨channel/s.BV,split,h⟩ t (channel%s.BV)
  | .dq => programQ a s p x seed ⟨split,channel/s.BK,h⟩ t (channel%s.BK)
  | .dk => programK a s p x final ⟨split,channel/s.BK,h⟩ t (channel%s.BK)
  | .dv => programV a s p x final ⟨channel/s.BV,split,h⟩ t (channel%s.BV)
  | .finalState =>
      let h := i/s.DV/s.DK
      let k := i/s.DV%s.DK
      let v := i%s.DV
      programForwardCarry a s x seed ⟨v/s.BV,k/s.BK,h⟩ s.T (k%s.BK) (v%s.BV)

def sourcePhase (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (seed final : Bool) (kind : PartialKind) (storage count : Nat) (m : Memory α) : Memory α :=
  fillRange storage count (sourceCell a s p (b.observe s m) seed final kind) m

def sourceForward (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 3 caller) (seed final : Bool) (m : Memory α) : Memory α :=
  let intermediate := sourcePhase a s p b seed Bool.false .output (alloc.storage 0) (s.NK*rowCount s s.DV) m
  let intermediate := if final then sourcePhase a s p b seed Bool.false .finalState (alloc.storage 1)
      (s.BH*s.DK*s.DV) intermediate else intermediate
  reduction a p.sumO (alloc.storage 0) (alloc.storage 2) (rowCount s s.DV) intermediate

def sourceBackward (a : Arithmetic α) (s : Shape) (p : Plans) (b : InputBuffers)
    (alloc : Allocation 6 caller) (seed final : Bool) (m : Memory α) : Memory α :=
  let dq := sourcePhase a s p b seed final .dq (alloc.storage 0) (s.NV*rowCount s s.DK) m
  let dk := sourcePhase a s p b seed final .dk (alloc.storage 1) (s.NV*rowCount s s.DK) dq
  let dv := sourcePhase a s p b seed final .dv (alloc.storage 2) (s.NK*rowCount s s.DV) dk
  let dq := reduction a p.sumQ (alloc.storage 0) (alloc.storage 3) (rowCount s s.DK) dv
  let dk := reduction a p.sumK (alloc.storage 1) (alloc.storage 4) (rowCount s s.DK) dq
  reduction a p.sumV (alloc.storage 2) (alloc.storage 5) (rowCount s s.DV) dk

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention
