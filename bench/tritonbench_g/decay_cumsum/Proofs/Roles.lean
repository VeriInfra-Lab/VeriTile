import bench.tritonbench_g.decay_cumsum.Proofs.Source
import Mathlib.Tactic

namespace VeriTile.Bench.Repairs.DecayCumsum.Roles
open VeriTile.Triton

def inputs : Kind → List RegionName
  | .forward => ["g"]
  | .prepare => ["q","k","g"]
  | .backward => ["dqi","dqo","dki","dko","q","k","g"]
def outputs : Kind → List RegionName
  | .forward => ["go"]
  | .prepare => ["qg","kg"]
  | .backward => ["dqo","dko","dg"]
def buffers : Kind → List RegionName
  | .forward => ["g","go"]
  | .prepare => ["q","k","g","qg","kg"]
  | .backward => ["dqi","dqo","dki","dko","q","k","g","dg"]

theorem input_member (kind : Kind) (r : RegionName) (hr : r∈inputs kind) : r∈buffers kind := by
  cases kind <;> simp_all [inputs,buffers] <;> tauto
theorem output_member (kind : Kind) (r : RegionName) (hr : r∈outputs kind) : r∈buffers kind := by
  cases kind <;> simp_all [outputs,buffers] <;> tauto

end VeriTile.Bench.Repairs.DecayCumsum.Roles
