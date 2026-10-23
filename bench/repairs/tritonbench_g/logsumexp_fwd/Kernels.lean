import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.LogsumexpFwd
open VeriTile.Triton
noncomputable section

/-- The separately repaired production body. `scale` is the actual runtime
FP32 parameter register, so native nonfinite scales are not erased to Real
literals. The standard parameter declaration supplies typing without emitting
an assignment; Grid.programState binds the actual launch argument.
Program IDs retain the original signed-int64 casts. -/
def rowBlock (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) : ComputeKernel :=
  triton {
  tl.region x = tl.float32, z = tl.float32
  tl.param scale : tl.float32
  i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
  o_d = i_d * $(B) + tl.arange(0, $(B))
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-float("inf"))
  if HAS_SCALE {
    b_x = b_x * scale
  }
  b_x = tl.where(m_d, b_x, -float("inf"))
  b_m = tl.max(b_x, 0)
  positive_inf = tl.bitcast(tl.full([], 2139095040, dtype=tl.uint32), tl.float32)
  b_m = tl.where((b_m == positive_inf) | (b_m == -float("inf")), 0.0, b_m)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}

/-- Avoid an unnecessary full-vector select in the unscaled specialization.
The masked load already supplies -infinity to every inactive lane. Scaling
retains mask restoration because arbitrary scales include nonfinite values. -/
def rowBlockCandidate (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) : ComputeKernel :=
  triton {
  tl.region x = tl.float32, z = tl.float32
  tl.param scale : tl.float32
  i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
  o_d = i_d * $(B) + tl.arange(0, $(B))
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-float("inf"))
  if HAS_SCALE {
    b_x = b_x * scale
    b_x = tl.where(m_d, b_x, -float("inf"))
  }
  b_m = tl.max(b_x, 0)
  positive_inf = tl.bitcast(tl.full([], 2139095040, dtype=tl.uint32), tl.float32)
  b_m = tl.where((b_m == positive_inf) | (b_m == -float("inf")), 0.0, b_m)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}

def blockLimit : Nat := 64*1024
def warpChoices : List Nat := [1,2,4,8,16,32]
def autotuneKey : List String := ["D"]

end
end VeriTile.Bench.Repairs.LogsumexpFwd
