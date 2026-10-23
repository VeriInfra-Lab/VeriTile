import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.HistoricalAggregate

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- Complete public API over exact-real scalar semantics: returned O and the
(dQ,dK,dV) tuple at every logical coordinate, exact saved context/metadata,
and both current-memory complement frames. The source's FP32 numerical
relation to this mathematical result is explicitly separate in FPEquiv. -/
specification sourceCorrect (abi : LocalRuntime ℝ σ) (p : Plans) (s : Shape)
    (domain : ShapeContract s) (b : Bases) (tree : SumPlan) (trees : OutputKind → SumPlan)
    (completeF : tree.Covers (OutputKind.forward.shards s))
    (completeB : ∀ kind,kind≠.forward → (trees kind).Covers (kind.shards s))
    (forwardLayout : ForwardLayout s b) (backwardLayout : BackwardLayout s b)
    (saved : SavedInputs s b) (forwardState backwardState : σ) (x : LogicalInputs ℝ s)
    (forwardInputs : ForwardMaterialized s b (abi.observe forwardState) x)
    (backwardInputs : Materialized s b (abi.observe backwardState) x) :
    let f := forwardAPI abi realOps p s b tree forwardLayout saved forwardState
    let g := backwardAPI abi realOps p s b trees backwardLayout f.context backwardState
    (∀ i,readReturned s .forward f.output (abi.observe f.state) i=mathematicalResult s x .forward i) ∧
    (∀ kind,kind≠.forward → ∀ i,
      readReturned s kind (g.handle kind) (abi.observe g.state) i=mathematicalResult s x kind i) ∧
    (f.context=⟨saved.query,saved.key,saved.value⟩ ∧
      f.output=outputHandle s .forward forwardLayout.output saved.query.device ∧
      g.query=outputHandle s .query (backwardLayout.output .query) saved.query.device ∧
      g.key=outputHandle s .key (backwardLayout.output .key) saved.query.device ∧
      g.value=outputHandle s .value (backwardLayout.output .value) saved.query.device) ∧
    (∀ address,
      (address<forwardLayout.scratch ∨ forwardLayout.scratch+OutputKind.forward.count s≤address) →
      (address<forwardLayout.output ∨ forwardLayout.output+OutputKind.forward.cells s≤address) →
      abi.observe f.state address=abi.observe forwardState address) ∧
    (∀ address,
      (∀ cell∈allCells s true,cellsAddress backwardLayout.scratch cell≠address) →
      (∀ cell∈allCells s false,cellsAddress backwardLayout.output cell≠address) →
      abi.observe g.state address=abi.observe backwardState address) := by
  dsimp only
  refine ⟨?_,?_,⟨rfl,rfl,rfl,rfl,rfl⟩,?_,?_⟩
  · exact forward_api_real abi p s domain b tree completeF forwardLayout saved forwardState x forwardInputs
  · intro kind active i
    exact backward_api_real abi p s domain b trees completeB backwardLayout _ backwardState x backwardInputs kind active i
  · exact forward_api_frame abi realOps p s b tree forwardLayout saved forwardState
  · exact backward_api_frame abi realOps p s b trees backwardLayout _ backwardState

/-- The returned gradient tensor words in exact-real semantics pair with every
input direction to give the derivative of the independent public forward
objective. This proves the VJP interpretation, rather than naming formulas
"gradients" without differentiating the forward operation. -/
specification returned_vjp (abi : LocalRuntime ℝ σ) (p : Plans) (s : Shape)
    (domain : ShapeContract s) (b : Bases) (trees : OutputKind → SumPlan)
    (complete : ∀ kind,kind≠.forward → (trees kind).Covers (kind.shards s))
    (layout : BackwardLayout s b) (context : SavedContext) (state : σ)
    (x : LogicalInputs ℝ s) (bound : Materialized s b (abi.observe state) x)
    (batch : Fin s.B) (head : Fin s.H)
    (dq dk : Fin s.T → Fin s.DK → ℝ) (dv : Fin s.T → Fin s.DV → ℝ) :
    let bh : Fin (s.B*s.H) := ⟨batch.val*s.H+head.val,by
      have hb := batch.isLt;have hh := head.isLt;nlinarith⟩
    let result := backwardAPI abi realOps p s b trees layout context state
    HasDerivAt (fun z : ℝ => lossSpec s.T s.DK s.DV head
      (fun t e => x.q bh t e+z*dq t e) (fun t e => x.k bh t e+z*dk t e)
      (fun t e => x.v bh t e+z*dv t e) (x.g bh))
      ((∑ t : Fin s.T,∑ e : Fin s.DK,dq t e*
          readReturned s .query (result.handle .query) (abi.observe result.state) ⟨batch,head,t,e⟩)+
       (∑ t : Fin s.T,∑ e : Fin s.DK,dk t e*
          readReturned s .key (result.handle .key) (abi.observe result.state) ⟨batch,head,t,e⟩)+
       (∑ t : Fin s.T,∑ e : Fin s.DV,dv t e*
          readReturned s .value (result.handle .value) (abi.observe result.state) ⟨batch,head,t,e⟩)) 0 := by
  dsimp only
  simp_rw [backward_api_real abi p s domain b trees complete layout context state x bound .query (by decide),
    backward_api_real abi p s domain b trees complete layout context state x bound .key (by decide),
    backward_api_real abi p s domain b trees complete layout context state x bound .value (by decide)]
  exact independent_vjp s.T s.DK s.DV head _ _ dq dk _ _ dv

#axiomsClean sourceCorrect
#axiomsClean returned_vjp
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention
