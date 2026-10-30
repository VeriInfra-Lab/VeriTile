import bench.tritonbench_g.rotary_transform.Proofs.Spec
import bench.tritonbench_g.rotary_transform.Proofs.Bounds
namespace VeriTile.Bench.Repairs.RotaryTransform
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

#axiomsClean correct_real
end VeriTile.Bench.Repairs.RotaryTransform
