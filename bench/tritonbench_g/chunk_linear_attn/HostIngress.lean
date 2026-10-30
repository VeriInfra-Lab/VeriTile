import bench.tritonbench_g.chunk_linear_attn.HostPhysical
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.Host
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false

def callerRead (physical : Nat → Cell ℝ) (src : Tensor) (b h t e : Nat) : ℝ :=
  (physical (src.base+offset4 src b h t e)).read .real

def callerInput (c : Invocation) (physical : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H) :
    Real.Inputs c.dims.T c.dims.K c.dims.V where
  q := fun t e => callerRead physical c.query b.val h.val t.val e.val
  k := fun t e => callerRead physical c.key b.val h.val t.val e.val
  v := fun t j => callerRead physical c.value b.val h.val t.val j.val
  initial := fun e j => match c.initial with
    | none => 0
    | some src => callerRead physical src b.val h.val e.val j.val

def callerIncoming (c : Invocation) (physical : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H)
    (t : Fin c.dims.T) (j : Fin c.dims.V) : ℝ :=
  match c.incoming with | none => 0 | some src => callerRead physical src b.val h.val t.val j.val

def callerTerminal (c : Invocation) (physical : Nat → Cell ℝ) (b : Fin c.dims.B) (h : Fin c.dims.H)
    (e : Fin c.dims.K) (j : Fin c.dims.V) : ℝ :=
  match c.terminal with | none => 0 | some src => callerRead physical src b.val h.val e.val j.val

/-- Ordinary contiguous descriptor interpretation, including arbitrary strides
on singleton axes. This is an address/layout fact, not a tensor value premise. -/
structure InitialLayout (c : Invocation) : Prop where
  packed : ∀ src∈c.initial, ∀ (b : Fin c.dims.B) (h : Fin c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V),
    offset4 src b.val h.val e.val j.val=((b.val*c.dims.H+h.val)*c.dims.K+e.val)*c.dims.V+j.val

 theorem caller_inputs (c : Invocation) (cc : Contract c) (a : Allocations)
    (initial : InitialLayout c) (physical : Nat → Cell ℝ) (fallback : Memory ℝ)
    (b : Fin c.dims.B) (h : Fin c.dims.H) :
    Real.inputs c.mathCall.backward (b.val*c.dims.H+h.val) (logical c a physical fallback) c.initial.isSome=
      callerInput c physical b h := by
  unfold Real.inputs callerInput
  congr 1
  · funext t e
    change (physical (c.query.base+((b.val*c.dims.H+h.val)*(c.dims.T*c.dims.K)+e.val+t.val*c.dims.K))).read .real=_
    unfold callerRead offset4
    simp only [cc.queryStrides,inputStrides,Bool.false_eq_true,if_false,List.getElem!_cons_zero,List.getElem!_cons_succ]
    congr 2
    ring
  · funext t e
    change (physical (c.key.base+((b.val*c.dims.H+h.val)*(c.dims.T*c.dims.K)+e.val+t.val*c.dims.K))).read .real=_
    unfold callerRead offset4
    simp only [cc.keyStrides,cc.queryStrides,inputStrides,Bool.false_eq_true,if_false,List.getElem!_cons_zero,List.getElem!_cons_succ]
    congr 2
    ring
  · funext t j
    change (physical (c.value.base+((b.val*c.dims.H+h.val)*(c.dims.T*c.dims.V)+t.val*c.dims.V+j.val))).read .real=_
    unfold callerRead offset4
    simp only [cc.valueStrides,inputStrides,if_true,List.getElem!_cons_zero,List.getElem!_cons_succ]
    congr 2
    ring
  · funext e j
    cases hi : c.initial with
    | none => simp [Real.inputs,Real.endpoint,callerInput,hi]
    | some src =>
      have he := initial.packed src (by simp [hi]) b h e j
      simp only [Real.inputs,Real.endpoint,hi,Option.isSome_some,if_true,seedName,Bool.false_eq_true,if_false]
      have hd : descriptor c a "initial"=some src := by simp [descriptor,hi]
      simp only [logical,hd]
      change (physical (src.base+((b.val*c.dims.H+h.val)*c.dims.K*c.dims.V+e.val*c.dims.V+j.val))).read .real=_
      simp only [callerInput,hi,callerRead,he]
      congr 2
      ring

 theorem caller_terminal (c : Invocation) (a : Allocations) (ac : AllocationContract c a)
    (physical : Nat → Cell ℝ) (fallback : Memory ℝ) (copies : CotangentCopies Real.algebra c a physical)
    (b : Fin c.dims.B) (h : Fin c.dims.H) (e : Fin c.dims.K) (j : Fin c.dims.V) :
    Real.endpoint c.mathCall.backward (b.val*c.dims.H+h.val) (logical c a physical fallback)
      Bool.true c.terminal.isSome e.val j.val=callerTerminal c physical b h e j := by
  cases ht : c.terminal with
  | none => simp [Real.endpoint,callerTerminal,ht]
  | some src =>
    have hf := ac.terminalFlag
    cases hd : a.terminalDense with
    | none => simp [hd,ht] at hf
    | some dst =>
      have hc := copies.terminal src (by simp [ht]) dst (by simp [hd]) b h e j
      simp only [Real.endpoint,ht,Option.isSome_some,if_true,seedName]
      have hdesc : descriptor c a "terminalGradient"=some dst := by simp [descriptor,hd]
      simp only [logical,hdesc]
      change (physical (dst.base+((b.val*c.dims.H+h.val)*c.dims.K*c.dims.V+e.val*c.dims.V+j.val))).read .real=_
      have he : dst.base+((b.val*c.dims.H+h.val)*c.dims.K*c.dims.V+e.val*c.dims.V+j.val)=
          dst.base+((b.val*c.dims.H+h.val)*c.dims.K+e.val)*c.dims.V+j.val := by ring
      rw [he,hc]
      simp only [callerTerminal,ht,callerRead]

 theorem caller_target (c : Invocation) (cc : Contract c) (a : Allocations) (ac : AllocationContract c a)
    (initial : InitialLayout c) (physical : Nat → Cell ℝ) (fallback : Memory ℝ)
    (copies : CotangentCopies Real.algebra c a physical) (b : Fin c.dims.B) (h : Fin c.dims.H) :
    c.mathCall.target (b.val*c.dims.H+h.val) (logical c a physical fallback)=
      Real.gradients c.scale.scalarValue (callerInput c physical b h)
        (callerIncoming c physical b h) (callerTerminal c physical b h) := by
  unfold Real.Call.target
  change Real.gradients c.scale.scalarValue
    (Real.inputs c.mathCall.backward (b.val*c.dims.H+h.val) (logical c a physical fallback) c.initial.isSome)
    (fun t j => Real.incoming c.mathCall.backward (b.val*c.dims.H+h.val) (logical c a physical fallback) t.val j.val)
    (fun e j => Real.endpoint c.mathCall.backward (b.val*c.dims.H+h.val) (logical c a physical fallback) Bool.true c.terminal.isSome e.val j.val)=_
  rw [caller_inputs c cc a initial physical fallback b h]
  congr 1
  · funext t j
    exact incoming_copy c a physical fallback copies b h t j
  · funext e j
    exact caller_terminal c a ac physical fallback copies b h e j

#axiomsClean caller_inputs
#axiomsClean caller_terminal
#axiomsClean caller_target
#specNonCircular InitialLayout avoiding [execute,Real.Call.result,Real.Call.target]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.Host
