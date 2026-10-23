import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelStorage

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
open Native
noncomputable section
set_option maxHeartbeats 4000000

theorem forwardCommit_read (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (tile : Tile) (output : Nat) (final : State (Expr α)) (state : σ)
    (i : TileIndex [128,s.BV]) (hi : activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i) :
    abi.observe (forwardCommit abi a p s tile output final state) (output+forwardAddress s tile i)=
      evaluate a p ((final.mem "o" (forwardAddress s tile i)).read .real) := by
  have h := writeTile_read abi (forwardBase s tile output) s.T s.DV (tile.chunk*128)
    (tile.value*s.BV) (fun j=>evaluate a p ((final.mem "o" (forwardAddress s tile j)).read .real)) state i hi
  simpa only [forwardCommit,forwardBase,forwardAddress,tileOffset,Nat.add_assoc] using h

theorem keyCommit_read (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (tile : Tile) (region : RegionName) (output : Nat) (final : State (Expr α)) (state : σ)
    (i : TileIndex [64,s.BK]) (hi : activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i) :
    abi.observe (keyCommit abi a p s tile region output final state) (output+keyAddress s tile i)=
      evaluate a p ((final.mem region (keyAddress s tile i)).read .real) := by
  have h := writeTile_read abi (keyBase s tile output) s.T s.DK (tile.chunk*64)
    (tile.key*s.BK) (fun j=>evaluate a p ((final.mem region (keyAddress s tile j)).read .real)) state i hi
  simpa only [keyCommit,keyBase,keyAddress,tileOffset,Nat.add_assoc] using h

theorem valueCommit_read (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (tile : Tile) (output : Nat) (final : State (Expr α)) (state : σ)
    (i : TileIndex [64,s.BV]) (hi : activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i) :
    abi.observe (valueCommit abi a p s tile output final state) (output+valueAddress s tile i)=
      evaluate a p ((final.mem "dv" (valueAddress s tile i)).read .real) := by
  have h := writeTile_read abi (valueBase s tile output) s.T s.DV (tile.chunk*64)
    (tile.value*s.BV) (fun j=>evaluate a p ((final.mem "dv" (valueAddress s tile j)).read .real)) state i hi
  simpa only [valueCommit,valueBase,valueAddress,tileOffset,Nat.add_assoc] using h

theorem key_member (s : Shape) (tile : Tile) (base : Nat) (i : TileIndex [64,s.BK])
    (hi : activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i) :
    base+keyAddress s tile i∈keyFootprint s tile base := by
  refine ⟨i,hi,?_⟩
  simp only [keyBase,keyAddress,tileOffset,Nat.add_assoc]

theorem value_member (s : Shape) (tile : Tile) (base : Nat) (i : TileIndex [64,s.BV])
    (hi : activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i) :
    base+valueAddress s tile i∈valueFootprint s tile base := by
  refine ⟨i,hi,?_⟩
  simp only [valueBase,valueAddress,tileOffset,Nat.add_assoc]

theorem forward_words (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (b : Bases) (tile : Tile) (valid : ProgramDomain s 128 tile)
    (output : Nat) (state : σ) :
    ∃ final,forwardRun abi a p s b tile output state=some final ∧
      (∀ i : TileIndex [128,s.BV],activeTile s.T s.DV (tile.chunk*128) (tile.value*s.BV) i →
        abi.observe final (output+forwardAddress s tile i)=
          forwardLane a.ops (plans s p) s (currentInputs a.ops s b (abi.observe state)) tile i.1.val i.2.1.val) ∧
      (∀ address,address∉forwardFootprint s tile output → abi.observe final address=abi.observe state address) := by
  obtain ⟨source,runSource,readSource⟩ := forward_exec s b (abi.observe state) tile valid
  refine ⟨forwardCommit abi a p s tile output source state,?_,?_,?_⟩
  · simp only [forwardRun,runSource,bind,Option.bind,pure]
  · intro i hi
    rw [forwardCommit_read abi a p s tile output source state i hi,readSource i hi.1 hi.2,Cell.read_same]
    exact eval_forwardValues a p s b (abi.observe state) tile valid i.1 i.2.1
  · intro address outside
    exact writeTile_frame abi _ _ _ _ _ _ state address outside

theorem backward_words (abi : LocalRuntime α σ) (a : Scalar α) (p : DotTrees)
    (s : Shape) (b : Bases) (tile : Tile) (valid : ProgramDomain s 64 tile)
    (keyPositive : 0<s.BK) (valuePositive : 0<s.BV)
    (dq dk dv : Nat) (separate : BackwardSeparate s tile dq dk dv) (state : σ) :
    ∃ final,backwardRun abi a p s b tile dq dk dv state=some final ∧
      (∀ i : TileIndex [64,s.BK],activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        abi.observe final (dq+keyAddress s tile i)=
          queryLane a.ops (plans s p) s (currentInputs a.ops s b (abi.observe state)) tile i.1.val i.2.1.val) ∧
      (∀ i : TileIndex [64,s.BK],activeTile s.T s.DK (tile.chunk*64) (tile.key*s.BK) i →
        abi.observe final (dk+keyAddress s tile i)=
          (keyValueLane a.ops (plans s p) s (currentInputs a.ops s b (abi.observe state)) tile i.1.val i.2.1.val 0).1) ∧
      (∀ i : TileIndex [64,s.BV],activeTile s.T s.DV (tile.chunk*64) (tile.value*s.BV) i →
        abi.observe final (dv+valueAddress s tile i)=
          (keyValueLane a.ops (plans s p) s (currentInputs a.ops s b (abi.observe state)) tile i.1.val 0 i.2.1.val).2) ∧
      (∀ address,address∉keyFootprint s tile dq ∪ keyFootprint s tile dk ∪ valueFootprint s tile dv →
        abi.observe final address=abi.observe state address) := by
  obtain ⟨source,runSource,readQ,readK,readV⟩ := backward_exec s b (abi.observe state) tile
  let afterQ := keyCommit abi a p s tile "dq" dq source state
  let afterK := keyCommit abi a p s tile "dk" dk source afterQ
  refine ⟨valueCommit abi a p s tile dv source afterK,?_,?_,?_,?_,?_⟩
  · simp only [backwardRun,decoded_backward s b (abi.observe state) tile valid,
      runSource,bind,Option.bind,pure,afterQ,afterK]
  · intro i hi
    have member := key_member s tile dq i hi
    have outsideV := Set.disjoint_left.mp separate.queryValue member
    have outsideK := Set.disjoint_left.mp separate.queryKey member
    rw [show abi.observe (valueCommit abi a p s tile dv source afterK) (dq+keyAddress s tile i)=
      abi.observe afterK (dq+keyAddress s tile i) from writeTile_frame abi _ _ _ _ _ _ _ _ outsideV]
    rw [show abi.observe afterK (dq+keyAddress s tile i)=abi.observe afterQ (dq+keyAddress s tile i)
      from writeTile_frame abi _ _ _ _ _ _ _ _ outsideK]
    rw [keyCommit_read abi a p s tile "dq" dq source state i hi,readQ i hi.1 hi.2,Cell.read_same]
    exact eval_queryValues a p s b (abi.observe state) tile valid i.1 i.2.1
  · intro i hi
    have outsideV := Set.disjoint_left.mp separate.keyValue (key_member s tile dk i hi)
    rw [show abi.observe (valueCommit abi a p s tile dv source afterK) (dk+keyAddress s tile i)=
      abi.observe afterK (dk+keyAddress s tile i) from writeTile_frame abi _ _ _ _ _ _ _ _ outsideV]
    rw [keyCommit_read abi a p s tile "dk" dk source afterQ i hi,readK i hi.1 hi.2,Cell.read_same]
    exact congrArg Prod.fst (eval_backwardValues a p s b (abi.observe state) tile valid i.1 i.2.1 ⟨0,valuePositive⟩)
  · intro i hi
    rw [valueCommit_read abi a p s tile dv source afterK i hi,readV i hi.1 hi.2,Cell.read_same]
    exact congrArg Prod.snd (eval_backwardValues a p s b (abi.observe state) tile valid i.1 ⟨0,keyPositive⟩ i.2.1)
  · intro address outside
    have hq : address∉keyFootprint s tile dq := fun h=>outside (Or.inl (Or.inl h))
    have hk : address∉keyFootprint s tile dk := fun h=>outside (Or.inl (Or.inr h))
    have hv : address∉valueFootprint s tile dv := fun h=>outside (Or.inr h)
    exact (writeTile_frame abi _ _ _ _ _ _ _ _ hv).trans
      ((writeTile_frame abi _ _ _ _ _ _ _ _ hk).trans (writeTile_frame abi _ _ _ _ _ _ _ _ hq))

#axiomsClean forward_words
#axiomsClean backward_words
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
