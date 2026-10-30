import bench.tritonbench_g.chunk_gla_fwd.Proofs.LocalReal
import bench.tritonbench_g.chunk_gla_fwd.Proofs.SumBlocks

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Logical
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaFwd
open scoped BigOperators
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 4000000

structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  gate : Nat → ℝ
  value : Nat → ℝ
  history : Nat → ℝ

def featureAddress (T K h t k : Nat) := h*(T*K)+t*K+k

def term (T K : Nat) (a : Inputs) (h t u k : Nat) : ℝ :=
  a.query (featureAddress T K h t k) * a.key (featureAddress T K h u k) *
    Real.exp (a.gate (featureAddress T K h t k)-a.gate (featureAddress T K h u k))

/-- The independent attention contraction, over exactly the logical K features. -/
def contraction (T K : Nat) (a : Inputs) (scale : ℝ) (h t u : Nat) : ℝ :=
  (∑ k ∈ Finset.range K, term T K a h t u k)*scale

/-- A returned attention cell. The complete rectangular result is zero above
its causal diagonal, including unused columns of a final time chunk. -/
def attention (T K BT : Nat) (a : Inputs) (scale : ℝ) (h t j : Nat) : ℝ :=
  if j ≤ t % BT then contraction T K a scale h t (t/BT*BT+j) else 0

/-- History is an explicit input with one K×V matrix per time chunk. -/
def output (T K V BT : Nat) (a : Inputs) (scale : ℝ) (h t v : Nat) : ℝ :=
  (∑ k ∈ Finset.range K,
    a.query (featureAddress T K h t k) * scale * Real.exp (a.gate (featureAddress T K h t k)) *
      a.history (Geometry.historyAddress (Geometry.ceilDiv T BT) K V h (t/BT) k v)) +
  ∑ j ∈ Finset.range (t%BT+1), attention T K BT a scale h t j *
    a.value (Geometry.outputAddress T V h (t/BT*BT+j) v)

def interInputs (a : Inputs) : RealInterMath.Inputs := ⟨a.query,a.key,a.gate⟩
def intraInputs (a : Inputs) : RealIntraMath.Inputs := ⟨a.query,a.key,a.gate⟩
def splitInputs (a : Inputs) : RealSplitMath.Inputs := ⟨a.query,a.key,a.gate⟩

theorem inter_expected (p : RealInterMath.Layout) (l : RealInterMath.Launch)
    (a : Inputs) (scale : ℝ) (i j : Nat)
    (hs : p.headStride=p.T*p.K) (ht : p.timeStride=p.K) (hbk : 0<p.BK)
    (hr : RealInterMath.rowStart p l+i<p.T)
    (hc : RealInterMath.columnStart p l+j<p.T) :
    RealInterMath.expected p l (interInputs a) scale i j =
      contraction p.T p.K a scale l.head (RealInterMath.rowStart p l+i)
        (RealInterMath.columnStart p l+j) := by
  unfold RealInterMath.expected RealInterMath.partialSum
  rw [show RealInterMath.numBlocks p=Geometry.ceilDiv p.K p.BK from rfl,
    SumBlocks.truncated _ p.K p.BK hbk]
  · unfold contraction
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro k hk
    simp only [RealInterMath.term,RealInterMath.matrixValue,hr,hc,Finset.mem_range.mp hk,
      and_self,ite_true,interInputs,RealInterMath.matrixAddress,hs,ht,term,featureAddress]
  · intro k hk
    simp [RealInterMath.term,RealInterMath.matrixValue,Nat.not_lt.mpr hk]

theorem intra_expected (p : RealIntraMath.Layout) (l : RealIntraMath.Launch)
    (a : Inputs) (scale : ℝ) (i j : Nat)
    (hs : p.headStride=p.T*p.K) (ht : p.timeStride=p.K) :
    RealIntraMath.expected p l (intraInputs a) scale i j =
      if j ≤ i then contraction p.T p.K a scale l.head (RealIntraMath.rowStart p l+i)
        (RealIntraMath.rowStart p l+j) else 0 := by
  simp only [RealIntraMath.expected,RealIntraMath.term,RealIntraMath.matrixAddress,
    RealIntraMath.keyAddress,hs,ht,intraInputs,contraction,term,featureAddress]

theorem split_expected_sum (p : RealSplitMath.Layout) (l : RealSplitMath.Launch)
    (a : Inputs) (scale : ℝ) (i j : Nat)
    (hs : p.headStride=p.T*p.K) (ht : p.timeStride=p.K) (hbk : 0<p.BK) :
    (∑ f ∈ Finset.range (Geometry.ceilDiv p.K p.BK),
      RealSplitMath.expected p {l with feature:=f} (splitInputs a) scale i j) =
      if j ≤ i then contraction p.T p.K a scale l.head (RealSplitMath.rowStart p l+i)
        (RealSplitMath.rowStart p l+j) else 0 := by
  by_cases hij : j ≤ i
  · simp only [RealSplitMath.expected,hij,ite_true]
    rw [← Finset.sum_mul]
    unfold contraction
    congr 1
    let f := fun k => if k<p.K then
      term p.T p.K a l.head (RealSplitMath.rowStart p l+i) (RealSplitMath.rowStart p l+j) k else 0
    have heq : (∑ b ∈ Finset.range (Geometry.ceilDiv p.K p.BK), ∑ e : Fin p.BK, f (b*p.BK+e.val)) =
        ∑ k ∈ Finset.range p.K, f k :=
      SumBlocks.truncated f p.K p.BK hbk (by intro k hk; simp [f,Nat.not_lt.mpr hk])
    calc
      _ = ∑ b ∈ Finset.range (Geometry.ceilDiv p.K p.BK), ∑ e : Fin p.BK, f (b*p.BK+e.val) := by
        apply Finset.sum_congr rfl
        intro b _
        apply Finset.sum_congr rfl
        intro e _
        simp only [f,RealSplitMath.featureIndex,RealSplitMath.term,RealSplitMath.matrixAddress,
          RealSplitMath.keyAddress,RealSplitMath.rowStart,hs,ht,splitInputs,term,featureAddress]
      _ = ∑ k ∈ Finset.range p.K, f k := heq
      _ = _ := by
        apply Finset.sum_congr rfl
        intro k hk
        simp [f,Finset.mem_range.mp hk]
  · simp [RealSplitMath.expected,hij]

#specNonCircular attention avoiding [exec]
#specNonCircular output avoiding [exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.Logical
