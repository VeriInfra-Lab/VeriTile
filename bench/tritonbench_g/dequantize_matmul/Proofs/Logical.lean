import bench.tritonbench_g.dequantize_matmul.Proofs.Source

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 3000000

structure Input (α : Type) (s : Shape) where
  a : Fin s.M → Fin s.K → α
  b : Fin s.K → Fin s.N → BitVec 8
  scale : Fin s.N → α

structure Represents (x : Input α s) (bufs : Buffers) (m : Memory (Cell α)) : Prop where
  a : ∀ r k,m (bufs.a.addr (r.val*s.K+k.val))=.fp32 (x.a r k)
  b : ∀ k n,m (bufs.b.addr (k.val*s.N+n.val))=.int8 (x.b k n)
  scale : ∀ n,m (bufs.scale.addr n.val)=.fp32 (x.scale n)

/-- The mathematical specification uses only the logical input arrays. -/
def mathematical (x : Input ℝ s) (r : Fin s.M) (n : Fin s.N) : ℝ :=
  (List.ofFn (fun k : Fin s.K => x.a r k * ((x.b k n).toInt * x.scale n))).sum

/-- A selected-plan floating reference; the plan is fixed by ABI/configuration,
not selected by inspecting a matrix result. -/
def wordReference (a : Arithmetic α) (x : Input α s) (plan : LibraryPlan s)
    (r : Fin s.M) (n : Fin s.N) : α :=
  a.store ((plan.output r.val n.val).eval a (x.a r)
    (fun k => a.store (a.mul (a.signed8 (x.b k n)) (a.load (x.scale n)))) )

theorem result_words (a : Arithmetic α) (x : Input α s) (b : Buffers)
    (plan : LibraryPlan s) (m : Memory (Cell α)) (rep : Represents x b m)
    (r : Fin s.M) (n : Fin s.N) :
    resultValue a s b plan m r.val n.val=wordReference a x plan r n := by
  have ha : (fun k : Fin s.K => (m (b.a.addr (r.val*s.K+k.val))).float a.zero)=x.a r := by
    funext k; rw [rep.a r k]; rfl
  have hb : (fun k : Fin s.K => (dequantValue a s b m k.val n.val).float a.zero)=
      (fun k => a.store (a.mul (a.signed8 (x.b k n)) (a.load (x.scale n)))) := by
    funext k
    simp only [dequantValue,rep.b k n,rep.scale n,Cell.signed,Cell.float]
  simp only [resultValue,wordReference,ha,hb]

theorem reference_real (x : Input ℝ s) (plan : LibraryPlan s)
    (r : Fin s.M) (n : Fin s.N) :
    wordReference realArithmetic x plan r n=mathematical x r n := by
  change (plan.output r.val n.val).eval realArithmetic (x.a r)
    (fun k => (x.b k n).toInt*x.scale n)=_
  exact Plan.real_complete _ (plan.complete r.val r.isLt n.val n.isLt) _ _

theorem real_complete (x : Input ℝ s) (b : Buffers) (tile : Tile)
    (plan : LibraryPlan s) (sep : Separate b) (initial : Memory (Cell ℝ))
    (rep : Represents x b initial) (r : Fin s.M) (n : Fin s.N) :
    run realArithmetic s b tile plan initial (b.out.addr (r.val*s.N+n.val))=
      .fp32 (mathematical x r n) := by
  rw [output _ _ _ _ _ sep _ _ _ r.isLt n.isLt,
    result_words _ x b plan initial rep r n,reference_real]

#axiomsClean result_words
#axiomsClean reference_real
#axiomsClean real_complete
end
end VeriTile.Bench.Repairs.DequantizeMatmul
