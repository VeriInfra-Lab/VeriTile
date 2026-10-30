import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Whole
import Mathlib.Data.Real.Basic

namespace VeriTile.Bench.Repairs.RotaryEmbNopad
noncomputable section

/-- Independent mathematical pair formula over arbitrary real caller memory. -/
def realExpected (r : Request) (m : Memory ℝ) (w : Work) (odd : Bool) : ℝ :=
  let a := m (.other r.device) (r.data w.key w.token w.head w.pair)
  let b := m (.other r.device) (r.data w.key w.token w.head (w.pair+r.half))
  let cosine := m (.other r.device) (r.cosBase+w.token*r.cosWidth+w.pair)
  let sine := m (.other r.device) (r.sinBase+w.token*r.sinWidth+w.pair)
  if odd then a*sine+b*cosine else a*cosine-b*sine

def realOps : Ops ℝ := ⟨(·*·),(·-·),(·+·)⟩

theorem real_pair (r : Request) (m : Memory ℝ) (w : Work) (odd : Bool) :
    component (sourceValues realOps r w m) odd=realExpected r m w odd := by
  cases odd <;> simp [component,sourceValues,sourcePair,realOps,realExpected,Request.dim,Request.coefficient]

def columnWork (r : Request) (key : Bool) (token head column : Nat) : Work :=
  ⟨key,token,head,if column<r.half then column else column-r.half⟩

def second (r : Request) (column : Nat) : Bool := decide (r.half≤column)

theorem column_owner (r : Request) (key : Bool) (token head column : Nat)
    (ht : token<r.tokens) (hh : head<r.heads key) (hd : column<r.width) :
    r.Active (columnWork r key token head column) ∧
    r.dim (columnWork r key token head column).pair (second r column)=column := by
  simp only [Request.width] at hd
  by_cases low : column<r.half
  · simp [columnWork,Request.Active,second,Request.dim,low,show ¬r.half≤column by omega,ht,hh]
  · have high : r.half≤column := by omega
    simp [columnWork,Request.Active,second,Request.dim,low,high,ht,hh]
    omega

/-- All feature columns, including either split half, use the original input
and common row coefficient. Cache destinations are specified separately. -/
def realColumn (r : Request) (m : Memory ℝ) (key : Bool) (token head column : Nat) : ℝ :=
  realExpected r m (columnWork r key token head column) (second r column)

#specNonCircular realExpected avoiding [sourcePair,sourceValues,sourceStep,execute,source,Physical.native]
#specNonCircular realColumn avoiding [sourcePair,sourceValues,sourceStep,execute,source,Physical.native]
#axiomsClean real_pair
#axiomsClean column_owner
end
end VeriTile.Bench.Repairs.RotaryEmbNopad
