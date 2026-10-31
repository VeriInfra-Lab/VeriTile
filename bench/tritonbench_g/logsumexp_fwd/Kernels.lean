import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

/-! Complete original Python function body and the existing Python
unscaled-mask candidate. Pointer/scalar/constexpr information is
supplied by the call outside the body. In particular, the positive infinity
literal is copied directly; no bitcast statement replaces the Python source. -/
namespace VeriTile.Bench.Repairs.LogsumexpFwd
open VeriTile.Triton
noncomputable section

def callSignature : CallSignature where
  pointers := [("x",.fp32),("z",.fp32)]
  scalars := [("scale",.fp32)]
  constexpr := ["D","B","HAS_SCALE"]

def rowBlock (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
    b_x = tl.where(m_d, b_x, -float('inf'))
    b_m = tl.max(b_x, 0)
    positive_inf = float('inf')
    b_m = tl.where((b_m == positive_inf) | (b_m == -float('inf')), 0.0, b_m)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)
  }

def rowBlockCandidate (x z : RegionName) (D B : Nat) (HAS_SCALE : Bool) : ComputeKernel :=
  with_triton_signature callSignature in
  triton {
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
        b_x = tl.where(m_d, b_x, -float('inf'))
    b_m = tl.max(b_x, 0)
    positive_inf = float('inf')
    b_m = tl.where((b_m == positive_inf) | (b_m == -float('inf')), 0.0, b_m)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)
  }

def blockLimit : Nat := 64*1024
def warpChoices : List Nat := [1,2,4,8,16,32]
def autotuneKey : List String := ["D"]

end
end VeriTile.Bench.Repairs.LogsumexpFwd
