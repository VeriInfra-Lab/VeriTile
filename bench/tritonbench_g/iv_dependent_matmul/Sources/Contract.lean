import bench.tritonbench_g.iv_dependent_matmul.Sources.Mathematical_HostDefinitions

namespace VeriTile.Bench.Repairs.IvDependentMatmul.Contract
open TritonBenchG.IvDependentMatmul Whole

/-- Checked block admission mirrors Python's positive power-of-two bit test. -/
def Block (n : Nat) : Prop := 0<n ∧ n≤2^20 ∧ n &&& (n-1)=0
instance (n : Nat) : Decidable (Block n) := inferInstanceAs (Decidable (_ ∧ _ ∧ _))

/-- Source envelopes are explicit preallocation guards, not allocator laws. -/
structure Admitted (r : Request) : Prop where
  rowsBlock : Block r.blockRows
  columnsBlock : Block r.blockColumns
  innerBlock : Block r.blockInner
  rowsControl : r.rows+r.blockRows<2^63
  columnsControl : r.columns+r.blockColumns<2^63
  innerControl : r.inner+4*r.blockInner+3<2^63
  inputBytes : 4*(r.rows*r.inner)<2^63
  weightBytes : 4*(r.inner*r.columns)<2^63
  outputBytes : 4*(r.rows*r.columns)<2^63
  grid : programs r<2^31

def check (r : Request) (defaultFP32 cuda : Bool) : Except String Unit :=
  if defaultFP32=Bool.false ∨ cuda=Bool.false then .error "profile"
  else if ¬Block r.blockRows ∨ ¬Block r.blockColumns ∨ ¬Block r.blockInner then .error "blocks"
  else if max (r.rows+r.blockRows) (max (r.columns+r.blockColumns) (r.inner+4*r.blockInner+3))≥2^63 then .error "control"
  else if 4*max (r.rows*r.inner) (max (r.inner*r.columns) (r.rows*r.columns))≥2^63 then .error "bytes"
  else if programs r≥2^31 then .error "grid"
  else .ok ()

def Preconditions (r : Request) (defaultFP32 cuda : Bool) : Prop :=
  defaultFP32=Bool.true ∧ cuda=Bool.true ∧ Admitted r

end VeriTile.Bench.Repairs.IvDependentMatmul.Contract
