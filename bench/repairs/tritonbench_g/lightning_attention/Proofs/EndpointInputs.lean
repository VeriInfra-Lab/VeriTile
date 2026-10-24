import bench.repairs.tritonbench_g.lightning_attention.Proofs.HostAPI
import bench.repairs.tritonbench_g.lightning_attention.Proofs.PhaseReal
import bench.repairs.tritonbench_g.lightning_attention.Proofs.TargetLoss
namespace VeriTile.Bench.Repairs.LightningAttention.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def callerQ (c : Invocation) (before : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H) :=
  fun (t : Fin c.dims.N) (a : Fin c.dims.D) => callerRead Neg.neg before c.query b h t a
def callerK (c : Invocation) (before : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H) :=
  fun (t : Fin c.dims.N) (a : Fin c.dims.D) => callerRead Neg.neg before c.key b h t a
def callerV (c : Invocation) (before : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H) :=
  fun (t : Fin c.dims.N) (a : Fin c.dims.E) => callerRead Neg.neg before c.value b h t a
def callerG (c : Invocation) (before : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H) :=
  fun (t : Fin c.dims.N) (a : Fin c.dims.E) => callerRead Neg.neg before c.incoming b h t a

def headFin (d : Dimensions) (b : Fin d.B) (h : Fin d.H) : Fin (d.B*d.H) :=
  ⟨b.val*d.H+h.val,by
    have hb := Nat.mul_le_mul_right d.H (Nat.succ_le_of_lt b.isLt)
    have hh := h.isLt
    rw [Nat.succ_mul] at hb
    omega⟩

theorem packed_expand (d : Dimensions) (width b h t f : Nat) :
    packed d width b h t f=(b*d.H+h)*d.N*width+t*width+f := by
  unfold packed; ring

theorem query_binding (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (fallback : Memory ℝ) (copies : ForwardCopies Neg.neg c a before copied)
    (b : Fin c.dims.B) (h : Fin c.dims.H) :
    Real.logicalQ c.dims.parameters (headFin c.dims b h).val (logical a copied fallback)=callerQ c before b h := by
  funext t f
  exact forward_query_copy Neg.neg c a before copied fallback copies b h t f

theorem key_binding (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (fallback : Memory ℝ) (copies : ForwardCopies Neg.neg c a before copied)
    (b : Fin c.dims.B) (h : Fin c.dims.H) :
    Real.logicalK c.dims.parameters (headFin c.dims b h).val (logical a copied fallback)=callerK c before b h := by
  funext t f
  exact forward_key_copy Neg.neg c a before copied fallback copies b h t f

theorem value_binding (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (fallback : Memory ℝ) (copies : ForwardCopies Neg.neg c a before copied)
    (b : Fin c.dims.B) (h : Fin c.dims.H) :
    Real.logicalV c.dims.parameters (headFin c.dims b h).val (logical a copied fallback)=callerV c before b h := by
  funext t f
  exact forward_value_copy Neg.neg c a before copied fallback copies b h t f

theorem incoming_binding (c : Invocation) (a : Allocations) (before copied : Nat → Cell ℝ)
    (fallback : Memory ℝ) (copies : IncomingCopy Neg.neg c a before copied)
    (b : Fin c.dims.B) (h : Fin c.dims.H) :
    Real.logicalG c.dims.parameters (headFin c.dims b h).val (logical a copied fallback)=callerG c before b h := by
  funext t f
  exact backward_incoming_copy Neg.neg c a before copied fallback copies b h t f

theorem saved_copies (c : Invocation) (a : Allocations) (before forwardCopied backwardCopied : Nat → Cell ℝ)
    (copies : ForwardCopies Neg.neg c a before forwardCopied)
    (saved : SavedInputs c a forwardCopied backwardCopied) : ForwardCopies Neg.neg c a before backwardCopied :=
  ⟨fun b h t f => (saved.query b h t f).trans (copies.query b h t f),
   fun b h t f => (saved.key b h t f).trans (copies.key b h t f),
   fun b h t f => (saved.value b h t f).trans (copies.value b h t f)⟩
#axiomsClean packed_expand
#axiomsClean query_binding
#axiomsClean key_binding
#axiomsClean value_binding
#axiomsClean incoming_binding
#axiomsClean saved_copies
end
end VeriTile.Bench.Repairs.LightningAttention.Host
