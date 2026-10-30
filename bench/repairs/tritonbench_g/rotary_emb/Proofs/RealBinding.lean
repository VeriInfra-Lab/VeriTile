import bench.repairs.tritonbench_g.rotary_emb.Proofs.TypedBinding

namespace VeriTile.Bench.Repairs.RotaryEmb.RealBinding
open VeriTile.Triton FP.Structural Checked
noncomputable section
set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- An explicit mathematical interpretation of current physical FP32 words.
The map is arbitrary, including on NaN and infinity encodings; no FP-to-Real
arithmetic conformance is assumed. Registers and all other regions are kept. -/
def state {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (decode : Native.Word → ℝ) (m : M) (surrounding : State ℝ) : State ℝ :=
  { surrounding with mem := fun r o =>
      if r="Q" then .mk .real (decode (api.read m (Binding.tensorKey (c.tensor 0) o)))
      else if r="K" then .mk .real (decode (api.read m (Binding.tensorKey (c.tensor 1) o)))
      else if r="Cos" then .mk .real (decode (api.read m (Binding.tensorKey (c.tensor 2) o)))
      else if r="Sin" then .mk .real (decode (api.read m (Binding.tensorKey (c.tensor 3) o)))
      else surrounding.mem r o }

def logical {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ) (m : M)
    (i : Fin 4) (o : Nat) : ℝ := decode (TypedBinding.logical api c objects m i o)

theorem direct {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ) (m : M)
    (accepted : check c=Bool.true) (i : Fin 4) (o : Nat) (bound : o<extent c.shape i) :
    decode (api.read m (Binding.tensorKey (c.tensor i) o))=logical api c objects decode m i o :=
  congrArg decode (TypedBinding.direct api c accepted objects m i o bound)

/-- The interleaved mathematical formula over logical elements of the actual
Tensor objects. Even and odd columns retain independent input coefficients. -/
def logicalPair {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ) (m : M)
    (w : Work) (odd : Bool) : ℝ :=
  let a := logical api c objects decode m (role w.key) (c.shape.cell w Bool.false)
  let b := logical api c objects decode m (role w.key) (c.shape.cell w Bool.true)
  let d := 2*w.pair+if odd then 1 else 0
  let cosine := logical api c objects decode m 2 (c.shape.coefficient Bool.false w.token d)
  let sine := logical api c objects decode m 3 (c.shape.coefficient Bool.true w.token d)
  if odd then a*sine+b*cosine else a*cosine-b*sine

theorem data_bound (c : Call) (accepted : check c=Bool.true) (w : Work)
    (active : c.shape.Active w) (odd : Bool) : c.shape.cell w odd<extent c.shape (role w.key) := by
  have h := Host.data_offset_bound c.shape (shape_valid c accepted) w active odd
  cases hk : w.key <;> simpa [hk,extent,expectedShape,role,Shape.heads,Nat.mul_assoc] using h

theorem coefficient_bound (c : Call) (accepted : check c=Bool.true) (w : Work)
    (active : c.shape.Active w) (sine odd : Bool) :
    c.shape.coefficient sine w.token (2*w.pair+if odd then 1 else 0)<
      extent c.shape (coefficientRole sine) := by
  have h := Host.coefficient_offset_bound c.shape (shape_valid c accepted) w active sine odd
  cases sine <;> simpa [extent,expectedShape,coefficientRole] using h

theorem value_logical {M : Type} (api : Binding.API M Native.Word Physical.Byte) (c : Call)
    (objects : TypedBinding.Objects c) (decode : Native.Word → ℝ) (m : M)
    (accepted : check c=Bool.true) (surrounding : State ℝ) (key odd : Bool)
    (i : Typed.Lane (Typed.layout c.shape))
    (live : Typed.active (Typed.layout c.shape) surrounding key i) :
    Typed.value realOps (Typed.layout c.shape) (state api c decode m surrounding) key odd i=
      .mk .real (logicalPair api c objects decode m (Typed.work c.shape surrounding key i) odd) := by
  let w := Typed.work c.shape surrounding key i
  have active : c.shape.Active w := (Typed.active_work c.shape (shape_valid c accepted) surrounding key i).mp live
  have hd (b : Bool) := direct api c objects decode m accepted (role w.key)
    (c.shape.cell w b) (data_bound c accepted w active b)
  have hc (sine : Bool) := direct api c objects decode m accepted (coefficientRole sine)
    (c.shape.coefficient sine w.token (2*w.pair+if odd then 1 else 0))
    (coefficient_bound c accepted w active sine odd)
  have hd0 := hd Bool.false
  have hd1 := hd Bool.true
  have hc0 := hc Bool.false
  have hc1 := hc Bool.true
  simp only [Typed.value,Typed.offset_cell]
  cases key <;> cases odd <;>
    simp [logicalPair,state,Typed.region,Typed.layout,Typed.coefficient,Typed.token,
      Typed.head,Typed.dimension,Typed.work,realOps,role,coefficientRole,Shape.coefficient,
      w,Nat.mul_comm] at hd0 hd1 hc0 hc1 ⊢ <;>
    rw [hd0,hd1,hc0,hc1]

#specNonCircular logicalPair avoiding [Kernels.kernel,Typed.execution,Typed.reference,Typed.value]
#axiomsClean direct
#axiomsClean value_logical
end
end VeriTile.Bench.Repairs.RotaryEmb.RealBinding
