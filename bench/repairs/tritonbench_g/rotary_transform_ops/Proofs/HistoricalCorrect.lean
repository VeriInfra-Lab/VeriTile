import bench.repairs.tritonbench_g.rotary_transform_ops.Spec
import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.Bounds
namespace VeriTile.Bench.Repairs.RotaryTransformOps
open Physical Host
/-- Every valid logical column of the normal completed host invocation.
Metadata is decoded from original caller bytes, not supplied as a Config. -/
theorem correct_real (call : Call) (callerDevice : Nat) (initial : Bytes) (m : Memory ℝ)
    (hc : (originalConfiguration call.shape call.abi initial).Valid)
    (batch row head col : Nat) (hb : batch<call.shape.batches)
    (hr : row<(originalConfiguration call.shape call.abi initial).length batch)
    (hh : head<call.shape.heads) (hd : col<call.shape.width) :
    (mathematical call callerDevice initial m).memory call.shape.output
      (call.shape.address ((originalConfiguration call.shape call.abi initial).start batch+row) head col)=
    realFormula (originalConfiguration call.shape call.abi initial) m batch row head col := by
  let c := originalConfiguration call.shape call.abi initial
  rw [mathematical_memory call callerDevice initial m hc,source_eq_run _ hc]
  change run c realOps (prepare c.shape m) c.shape.output (c.shape.address (c.start batch+row) head col)=realFormula c m batch row head col
  by_cases rot : col<2*c.shape.half
  · obtain ⟨hp,back⟩ := column_inverse c.shape col rot
    let w : Work := ⟨batch,row,head,c.shape.pairOf col⟩
    have active : c.Active w := ⟨hb,hr,hh,hp⟩
    have result := run_readback c hc realOps (prepare c.shape m) w active (c.shape.second col)
    rw [prepare_pair,real_pair_formula c m w active] at result
    simpa only [w,Config.cell,back] using result
  · have tail : 2*c.shape.half≤col := by omega
    rw [run_tail c hc realOps _ _ _ _ hh hd tail,
      prepare_tail c.shape m _ _ _ (row_bound c hc batch row hb hr) hh hd tail]
    simp only [realFormula,rot,↓reduceIte]

structure RealClaim (call : Call) (callerDevice : Nat) (initial : Bytes) (m : Memory ℝ)
    (result : Returned ℝ) : Prop where
  returned : result.valueIdentity=(if call.shape.inplace then call.inputIdentity else call.freshIdentity)
  shape : result.shape=call.shape
  region : result.valueRegion=call.shape.output
  dtype : result.fp32=true
  device : result.device=call.device ∧ result.deviceRestored=callerDevice
  source : result.memory=source (originalConfiguration call.shape call.abi initial) realOps (prepare call.shape m)
  values : ∀ b r h d,b<call.shape.batches → r<(originalConfiguration call.shape call.abi initial).length b →
    h<call.shape.heads → d<call.shape.width →
    result.memory call.shape.output (call.shape.address ((originalConfiguration call.shape call.abi initial).start b+r) h d)=
      realFormula (originalConfiguration call.shape call.abi initial) m b r h d
  frame : ∀ q a,¬copied call.shape q a →
    (∀ w,(originalConfiguration call.shape call.abi initial).Active w → ¬Writes (originalConfiguration call.shape call.abi initial) w q a) →
    result.memory q a=m q a

/-- The real payload view is derived pointwise from the same initial words.
`realize` is only an input scalar interpretation, never an output callback. -/
def realView (realize : Native.Word→ℝ) (initial : Bytes) : Memory ℝ := fun r a => realize (load initial r a)
theorem whole_real (call : Call) (initial : Bytes) (domain : Admitted call initial) (callerDevice : Nat) (realize : Native.Word→ℝ) :
    RealClaim call callerDevice initial (realView realize initial)
      (mathematical call callerDevice initial (realView realize initial)) := by
  refine ⟨rfl,rfl,rfl,rfl,⟨rfl,rfl⟩,mathematical_memory call callerDevice initial _ domain.config,?_,?_⟩
  · exact correct_real call callerDevice initial _ domain.config
  · intro q a noCopy outside
    rw [mathematical_memory call callerDevice initial _ domain.config,source_eq_run _ domain.config,
      run_frame _ realOps _ q a outside]
    simp [prepare,noCopy]
#axiomsClean whole_real
#axiomsClean correct_real
end VeriTile.Bench.Repairs.RotaryTransformOps
