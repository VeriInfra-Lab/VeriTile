import bench.tritonbench_g.kldiv_triton.Proofs.Prepare

namespace VeriTile.Bench.Repairs.KldivTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open L2NormBwd.Physical (Matrix)
open Physical (ForwardLayout BackwardLayout rowValues jobs job_member)
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (rounds)
noncomputable section
open Classical
set_option autoImplicit false
set_option maxHeartbeats 6000000

def point {α κ : Type} (a : Storage.Adapter α κ) (flag : Bool) (y t : α) : α :=
  a.project (Tile.loss a.model flag (a.model.alg.fp32Load (a.embed y)) (a.model.alg.fp32Load (a.embed t)))
def chunk {α κ : Type} (a : Storage.Adapter α κ) (flag : Bool) (N B k : Nat) (ys ts : Nat → α) : α :=
  a.project (Tile.chunkSum a.model flag N B k (fun i => a.embed (ys i)) (fun i => a.embed (ts i)))
def derivative {α κ : Type} (a : Storage.Adapter α κ) (flag : Bool) (t : α) : α :=
  a.project (Tile.gradient a.model flag (a.model.alg.fp32Load (a.embed t)))

theorem chunk_congr {α κ : Type} (a : Storage.Adapter α κ) (flag : Bool) (N B k : Nat)
    (ys ts us vs : Nat → α) (hy : ∀ i,i<N → ys i=us i) (ht : ∀ i,i<N → ts i=vs i) :
    chunk a flag N B k ys ts=chunk a flag N B k us vs := by
  unfold chunk Tile.chunkSum
  congr 2
  funext j
  by_cases active : k*B+j.1.val<N
  · simp only [Tile.laneLoss,Tile.loaded,active,if_true,hy _ active,ht _ active]
  · simp only [Tile.laneLoss,active,if_false]

theorem flat_bounds (m : Matrix) (i : Nat) (active : i<m.view.numel) :
    i/m.cols<m.rows ∧ i%m.cols<m.cols ∧ i/m.cols*m.cols+i%m.cols=i := by
  rw [m.numel] at active
  have positive : 0<m.cols := by
    by_contra h
    have empty : m.cols=0 := by omega
    simp [empty] at active
  exact ⟨(Nat.div_lt_iff_lt_mul positive).mpr (by simpa [Nat.mul_comm] using active),
    Nat.mod_lt i positive,by simpa [Nat.mul_comm] using Nat.div_add_mod i m.cols⟩

theorem coord_bound (m : Matrix) (r c : Nat) (hr : r<m.rows) (hc : c<m.cols) : r*m.cols+c<m.view.numel := by
  rw [m.numel]
  have h := Nat.mul_le_mul_right m.cols (show r+1≤m.rows by omega)
  simp only [Nat.add_mul,Nat.one_mul] at h
  omega

theorem PreparedMatrix.raw {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {input output : Matrix} {before after : World M} (h : PreparedMatrix ops neg input output before after)
    (r c : Nat) (hr : r < input.rows) (hc : c < input.cols) :
    rowValues ops after.memory output r c=logical ops neg before.memory input.view (r*input.cols+c) := by
  have v := h.values (r*input.cols+c) (coord_bound input r c hr hc)
  rw [←h.cols] at v
  unfold logical at v
  simp only [h.positive,Negation.encode,Bool.false_eq_true,if_false] at v
  rw [output.logical_key r c (by rw [h.cols];exact hc)] at v
  simpa only [h.cols] using v

theorem Prepared.transport {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {input output : View} {initial before after : World M} (h : Prepared ops neg input output initial before)
    (stable : Stable ops before after) : Prepared ops neg input output initial after := by
  refine ⟨h.shape,h.positive,h.backed,stable.live _ h.live,h.owner,?_,h.storage⟩
  intro i hi
  rw [stable.logical neg output h.live h.backed i (by simpa [View.numel,h.shape] using hi)]
  exact h.values i hi

theorem PreparedMatrix.transport {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {input output : Matrix} {initial before after : World M} (h : PreparedMatrix ops neg input output initial before)
    (stable : Stable ops before after) : PreparedMatrix ops neg input output initial after :=
  ⟨h.toPrepared.transport stable,h.rows,h.cols,h.unit⟩

theorem Prepared.rebase {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {input output : View} {initial before after : World M} (h : Prepared ops neg input output before after)
    (stable : Stable ops initial before) (live : initial.live input.storage) (backed : input.backed) :
    Prepared ops neg input output initial after := by
  refine ⟨h.shape,h.positive,h.backed,h.live,h.owner,?_,?_⟩
  · intro i hi
    rw [h.values i hi,stable.logical neg input live backed i hi]
  · rcases h.storage with same | fresh
    · exact Or.inl same
    · exact Or.inr (fun old ho => fresh old (stable.live old ho))

theorem PreparedMatrix.rebase {M α β : Type} {ops : Ops M α β} {neg : Negation α}
    {input output : Matrix} {initial before after : World M} (h : PreparedMatrix ops neg input output before after)
    (stable : Stable ops initial before) (live : initial.live input.view.storage) (backed : input.view.backed) :
    PreparedMatrix ops neg input output initial after :=
  ⟨h.toPrepared.rebase stable live backed,h.rows,h.cols,h.unit⟩

def outputCols (call : ForwardCall) : Nat :=
  if call.mode=.none then call.prediction.cols else rounds call.prediction.cols (blockSize call.prediction.cols)

def forwardExpected {M α β κ : Type} (ops : Ops M α β) (neg : Negation α)
    (adapter : Storage.Adapter α κ) (call : ForwardCall) (initial : World M) (i : Nat) : α :=
  let r := i/outputCols call
  let c := i%outputCols call
  let ys := fun c => logical ops neg initial.memory call.prediction.view (r*call.prediction.cols+c)
  let ts := fun c => logical ops neg initial.memory call.target.view (r*call.target.cols+c)
  if call.mode=.none then point adapter call.flag (ys c) (ts c)
  else chunk adapter call.flag call.prediction.cols (blockSize call.prediction.cols) c ys ts

/-- The produced values come from full checked source executions. The logical
snapshot formula is derived from ordinary view preparation coordinate laws. -/
theorem forward_kernel_values {M α β κ : Type} [Inhabited κ]
    (ops : Ops M α β) (neg : Negation α) (adapter : Storage.Adapter α κ)
    (call : ForwardCall) (initial before : World M) (y t output : Matrix)
    (hy : PreparedMatrix ops neg call.prediction y initial before)
    (ht : PreparedMatrix ops neg call.target t initial before)
    (sameRows : call.target.rows=call.prediction.rows) (sameCols : call.target.cols=call.prediction.cols)
    (outRows : output.rows=call.prediction.rows) (outCols : output.cols=outputCols call)
    (positive : output.view.negative=Bool.false) (unique : output.injective)
    (i : Nat) (active : i<output.view.numel) :
    logical ops neg
      (Storage.forwardKernel ops adapter before.memory
        ⟨y,t,output,blockSize call.prediction.cols,call.flag,modeCode call.mode⟩ (block_positive _))
      output.view i=forwardExpected ops neg adapter call initial i := by
  obtain ⟨hr,hc,index⟩ := flat_bounds output i active
  let r := i/output.cols
  let c := i%output.cols
  have hr' : r<call.prediction.rows := by simpa only [r,outRows] using hr
  have targetRows : r<call.target.rows := by rw [sameRows];exact hr'
  unfold logical
  simp only [positive,Negation.encode,Bool.false_eq_true,if_false]
  have keyeq : output.view.key i=output.key r c :=
    (congrArg output.view.key index.symm).trans (output.logical_key _ _ hc)
  rw [keyeq]
  rw [Storage.forward_readback ops adapter before.memory _ (block_positive _) unique _ _ hr hc]
  change Storage.forwardValue ops adapter before.memory
    ⟨y,t,output,blockSize call.prediction.cols,call.flag,modeCode call.mode⟩ (block_positive _) r c=_
  unfold forwardExpected
  rw [←outCols]
  change _ = if call.mode=.none then
    point adapter call.flag (logical ops neg initial.memory call.prediction.view (r*call.prediction.cols+c))
      (logical ops neg initial.memory call.target.view (r*call.target.cols+c)) else
    chunk adapter call.flag call.prediction.cols (blockSize call.prediction.cols) c
      (fun j => logical ops neg initial.memory call.prediction.view (r*call.prediction.cols+j))
      (fun j => logical ops neg initial.memory call.target.view (r*call.target.cols+j))
  by_cases mode : call.mode=.none
  · rw [if_pos mode]
    have activeCol : c<call.prediction.cols := by simpa only [outCols,outputCols,mode,if_true,c] using hc
    rw [Storage.forward_element ops adapter before.memory _ (block_positive _) hy.unit ht.unit (by simp [mode,modeCode])
      r c (by simpa only [hy.cols] using activeCol)
      (by simpa only [hy.rows] using hr') (by simpa only [ht.rows] using targetRows)
      (by simpa only [ht.cols,sameCols] using activeCol)]
    unfold point
    rw [hy.raw r c hr' activeCol,ht.raw r c targetRows (by rw [sameCols];exact activeCol)]
  · rw [if_neg mode]
    have activeChunk : c<rounds y.cols (blockSize call.prediction.cols) := by
      simpa only [c,outCols,outputCols,mode,if_false,hy.cols] using hc
    have code : modeCode call.mode≠0 := by cases h : call.mode <;> simp_all [modeCode]
    rw [Storage.forward_chunk ops adapter before.memory _ (block_positive _) hy.unit ht.unit code r c activeChunk
      (by simpa only [hy.rows] using hr') (by simpa only [ht.rows] using targetRows)
      (by simp only [ht.cols,hy.cols,sameCols])]
    change chunk adapter call.flag y.cols (blockSize call.prediction.cols) c
      (rowValues ops before.memory y r) (rowValues ops before.memory t r)=_
    rw [hy.cols]
    apply chunk_congr
    · intro j hj;exact hy.raw r j hr' hj
    · intro j hj;exact ht.raw r j targetRows (by rw [sameCols];exact hj)

#axiomsClean forward_kernel_values
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.Host
