import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealModelBindings
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealStepCarry

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
open Host
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

/-- Finite tensor data are connected to initial memory only at source read
addresses. Continuation carry is required exactly when INIT is false. -/
structure Pins (c : Call) (a : Reference.Inputs R) (xs : RealInputs.Inputs c.BM c.D c.NK) : Prop where
  query : ∀ (r : Fin c.BM) (d : Fin c.D), c.params.evenM≠0 ∨ a.tile*c.BM+r.val<c.NQ →
    a.read "Q" ((Reference.qptr c a).address [r.val,d.val])=some (xs.query r d)
  key : ∀ (k : Fin c.NK) (d : Fin c.D),
    a.read "K" (Reference.batch c a*c.k.batch+Reference.kvhead c a*c.k.head+d.val*c.k.dim+k.val*c.k.row)=some (xs.key k d)
  value : ∀ (k : Fin c.NK) (d : Fin c.D),
    a.read "V" (Reference.batch c a*c.v.batch+Reference.kvhead c a*c.v.head+k.val*c.v.row+d.val*c.v.dim)=some (xs.value k d)
  center : ∀ (r : Fin c.BM), c.init=0 → a.read "M" (Reference.carryAddress c a r.val)=some (xs.center r)
  mass : ∀ (r : Fin c.BM), c.init=0 → a.read "L" (Reference.carryAddress c a r.val)=some (xs.denominator r)
  weighted : ∀ (r : Fin c.BM) (d : Fin c.D), c.init=0 →
    a.read "O" ((Reference.optr c a).address [r.val,d.val])=some (xs.output r d)

def queryValues (c : Call) (tile : Nat) (xs : RealInputs.Inputs c.BM c.D c.NK) : TileIndex [c.BM,c.D] → ℝ :=
  fun i=>if c.params.evenM≠0 ∨ tile*c.BM+i.1.val<c.NQ then xs.query i.1 i.2.1 else 0

def keyValues (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) (physical : Nat) : TileIndex [c.D,c.BN] → ℝ :=
  fun i=>if h : physical+i.2.1.val<c.NK then xs.key ⟨physical+i.2.1.val,h⟩ i.1 else 0

def valueValues (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) (physical : Nat) : TileIndex [c.BN,c.D] → ℝ :=
  fun i=>if h : physical+i.1.val<c.NK then xs.value ⟨physical+i.1.val,h⟩ i.2.1 else 0

theorem query_load (c : Call) (a : Reference.Inputs R) (xs : RealInputs.Inputs c.BM c.D c.NK)
    (pins : Pins c a xs) : Reference.query c ops a=fun i=>some (queryValues c a.tile xs i) := by
  funext ⟨r,d,tail⟩
  cases tail
  by_cases he : c.params.evenM≠0
  · have pin:=pins.query r d (Or.inl he)
    simpa [Reference.query,Reference.loadBlock,Reference.qptr,ops,algebra,queryValues,he] using pin
  · by_cases hr : a.tile*c.BM+r.val<c.NQ
    · have pin:=pins.query r d (Or.inr hr)
      simpa [Reference.query,Reference.loadBlock,Reference.qptr,ops,algebra,queryValues,he,hr] using pin
    · simp [Reference.query,Reference.loadBlock,Reference.qptr,ops,algebra,queryValues,he,hr]

theorem key_load (c : Call) (a : Reference.Inputs R) (xs : RealInputs.Inputs c.BM c.D c.NK)
    (pins : Pins c a xs) (cur : Nat) (aligned : c.BN∣cur) (running : (cur : Int)<Reference.upper c a)
    (started : (Reference.lower c a).toNat≤cur) :
    Reference.loadBlock ops a (SourceSafety.keyAt c a cur) [c.D,c.BN]
      (if c.params.evenN≠0 then [] else [0,1])=
      fun i=>some (keyValues c xs (SourceSafety.physicalKey c a cur) i) := by
  funext ⟨d,k,tail⟩
  cases tail
  by_cases he : c.params.evenN≠0
  · have physical:=SourceSafety.physical_le c a cur started
    have upper:=SourceSafety.upper_bound c a
    have whole:=SourceSafety.full_block c cur aligned (by omega) he
    have bound : SourceSafety.physicalKey c a cur+k.val<c.NK := by have:=k.isLt; omega
    have pin:=pins.key ⟨_,bound⟩ d
    simpa [Reference.loadBlock,SourceSafety.keyAt,Reference.kptr,ops,algebra,keyValues,he,bound] using pin
  · by_cases bound : SourceSafety.physicalKey c a cur+k.val<c.NK
    · have pin:=pins.key ⟨_,bound⟩ d
      simpa [Reference.loadBlock,SourceSafety.keyAt,Reference.kptr,ops,algebra,keyValues,he,bound] using pin
    · simp [Reference.loadBlock,SourceSafety.keyAt,Reference.kptr,ops,algebra,keyValues,he,bound]

theorem value_load (c : Call) (a : Reference.Inputs R) (xs : RealInputs.Inputs c.BM c.D c.NK)
    (pins : Pins c a xs) (cur : Nat) (aligned : c.BN∣cur) (running : (cur : Int)<Reference.upper c a)
    (started : (Reference.lower c a).toNat≤cur) :
    Reference.loadBlock ops a (SourceSafety.valueAt c a cur) [c.BN,c.D]
      (if c.params.evenN≠0 then [] else [0,1])=
      fun i=>some (valueValues c xs (SourceSafety.physicalKey c a cur) i) := by
  funext ⟨k,d,tail⟩
  cases tail
  by_cases he : c.params.evenN≠0
  · have physical:=SourceSafety.physical_le c a cur started
    have upper:=SourceSafety.upper_bound c a
    have whole:=SourceSafety.full_block c cur aligned (by omega) he
    have bound : SourceSafety.physicalKey c a cur+k.val<c.NK := by have:=k.isLt; omega
    have pin:=pins.value ⟨_,bound⟩ d
    simpa [Reference.loadBlock,SourceSafety.valueAt,Reference.vptr,ops,algebra,valueValues,he,bound] using pin
  · by_cases bound : SourceSafety.physicalKey c a cur+k.val<c.NK
    · have pin:=pins.value ⟨_,bound⟩ d
      simpa [Reference.loadBlock,SourceSafety.valueAt,Reference.vptr,ops,algebra,valueValues,he,bound] using pin
    · simp [Reference.loadBlock,SourceSafety.valueAt,Reference.vptr,ops,algebra,valueValues,he,bound]

/-- Pure finite seed; no machine state, pointer, or execution result appears. -/
def seed (c : Call) (xs : RealInputs.Inputs c.BM c.D c.NK) : RealLoopFormula.Carry c.BM c.D where
  center:=fun r=>if c.init≠0 then none else some (xs.center r)
  denominator:=fun r=>if c.init≠0 then 1 else xs.denominator r
  numerator:=fun i=>if c.init≠0 then 0 else xs.output i.1 i.2.1

theorem initial_encoded (c : Call) (a : Reference.Inputs R) (xs : RealInputs.Inputs c.BM c.D c.NK)
    (pins : Pins c a xs) : Encodes c (Reference.initial c ops a) (seed c xs) := by
  by_cases hi : c.init=0
  · refine ⟨?_,?_,?_⟩
    · funext ⟨r,tail⟩; cases tail
      simp [Reference.initial,seed,hi,pins.center r hi,ops,algebra]
    · funext ⟨r,tail⟩; cases tail
      simp [Reference.initial,seed,hi,pins.mass r hi,ops,algebra]
    · funext ⟨r,d,tail⟩; cases tail
      simpa [Reference.initial,seed,hi,Reference.optr,ops,algebra] using pins.weighted r d hi
  · simp [Encodes,Reference.initial,seed,hi,ops,algebra]

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Real
