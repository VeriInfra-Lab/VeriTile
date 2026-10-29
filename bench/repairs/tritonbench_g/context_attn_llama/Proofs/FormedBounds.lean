import bench.repairs.tritonbench_g.context_attn_llama.Proofs.CheckedNative

namespace VeriTile.Bench.Repairs.ContextAttnLlama.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
noncomputable section
set_option maxHeartbeats 5000000

def paddedView (rows heads dims : Nat) : VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3 :=
  ⟨rows,heads,dims,heads*dims,dims,1,Bool.false⟩

theorem payload_fits (rows heads dims : Nat) (v : View) (budget : payloadBudget rows heads dims v)
    (row head dim : Nat) (hr : row < rows) (hh : head < heads) (hd : dim < dims) :
    row*(heads*dims)+head*dims+dim < 2^61 := by
  have bound := VeriTile.Bench.Repairs.ContextAttnFwd.FormedBounds.bound3 (paddedView rows heads dims) v.storageOffset
    row head dim ⟨by simpa [paddedView] using budget.1,budget.2.1,budget.2.2,by norm_num [paddedView]⟩ hr hh hd
  simpa [paddedView] using bound

theorem formed_rows (c : Call) (d : Dimensions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c d extent) (limit : c.maxInput≤requestedMax)
    (idx : GridIndex c.grid) (lane : Fin c.layout.BM) :
    c.start (Native.batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) lane.val < d.queryRows+requestedMax+128 := by
  have cap := (g.2.2.2.2.2.2.1 ⟨Native.batchAt c idx,Host.batch_bound c idx⟩).1
  change c.start (Native.batchAt c idx)+c.length (Native.batchAt c idx)≤d.queryRows at cap
  have block := Host.block_bound c idx
  have laneBound := lane.isLt
  rcases c.blockM with bm|bm <;> simp only [Call.blocks,bm] at block laneBound <;>
    simp only [Formula.queryRow,bm] <;> omega

theorem formed_key_row (c : Call) (d : Dimensions) (extent : RegionBounds)
    (g : geometryChecks c d extent) (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN)
    (ht : step < Native.countAt c idx) : step*c.layout.BN+j.val < d.tokens+128 := by
  have cap := (g.2.2.2.2.2.2.1 ⟨Native.batchAt c idx,Host.batch_bound c idx⟩).2.2.1
  have stop := Native.stop_le (c.launchAt idx).block c.layout.BM (Native.seqAt c idx) (c.prompt (Native.batchAt c idx))
    (by dsimp only [Native.seqAt]; omega)
  have stepped := (Native.ceil_step _ c.layout.BM step c.blockM).mp ht
  have lane := j.isLt
  change c.length (Native.batchAt c idx)+c.prompt (Native.batchAt c idx)≤d.tokens at cap
  change Native.stopIndex _ _ _ _≤c.length (Native.batchAt c idx)+c.prompt (Native.batchAt c idx) at stop
  rcases c.blockM with bm|bm <;> simp only [c.blockN,bm] at stepped lane stop ⊢ <;> omega

theorem formed_page (c : Call) (d : Dimensions) (extent : RegionBounds)
    (g : geometryChecks c d extent) (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN) :
    Native.mathPage c idx step j < max d.cacheRows 1 := by
  unfold Native.mathPage
  split_ifs with h
  · have fullBound := lt_of_lt_of_le h (Native.end_le _ _ _ _ (by dsimp only [Native.seqAt]; omega))
    have page := (g.2.2.2.2.2.2.1 ⟨Native.batchAt c idx,Host.batch_bound c idx⟩).2.2.2.2 ⟨_,fullBound⟩
    exact lt_of_lt_of_le page (le_max_left _ _)
  · omega

/-- Q/Out pointers are formed even in wholly inactive programs with no loop. -/
theorem query_formed (c : Call) (d : Dimensions) (v : Views) (A : FlatAlloc) (B : Bytes.Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (guard : InputContract c d v A.extent requestedMax start sequence prompt request pages)
    (before : Bytes.Memory) (observed : Bytes.Observed c B before start sequence prompt request pages)
    (storage : StorageBinding c v B) (needed : Bytes.Needed c)
    (idx : GridIndex c.grid) (i : Fin c.layout.BM) (dim : Fin c.layout.D)
    (role : Native.DataRole) (choice : role=.query ∨ role=.output) :
    (Native.qOffsetBits c widths role idx i dim).toInt=(Native.qOffset c role idx i dim : Int) ∧
    (Native.word (4*Native.qOffset c role idx i dim)).toInt=(4*Native.qOffset c role idx i dim : Nat) := by
  have abi := abi c d v A B requestedMax start sequence prompt request pages guard before observed storage needed
  have qr := formed_rows c d A.extent requestedMax guard.2.1 guard.2.2.1.2.1 idx i
  have qBound := payload_fits _ _ _ v.q (guard.2.2.1.2.2.2.2.1 v.q (by simp)) _ _ dim.val
    qr (Host.head_bound c idx) dim.isLt
  rcases guard.2.1.1 with ⟨qrow,qhead,qdim,orow,ohead,odim,_⟩
  have bound : Native.qOffset c role idx i dim < 2^61 := by
    rcases choice with rfl|rfl <;>
      simpa [Native.qOffset,Native.DataRole.row,Native.DataRole.head,Native.DataRole.dim,
        qrow,qhead,qdim,orow,ohead,odim] using qBound
  have eq := Native.query_modular c.layout (c.launchAt idx) widths.start _ _ _ (role.row c) (role.head c) (role.dim c)
    (Native.block_abi c A widths (nativeBytes B) abi idx) (Native.batchHead_abi c A widths (nativeBytes B) abi idx)
    c.blockM abi.heads abi.group (abi.start _ (Host.batch_bound c idx)) i.isLt (Native.dim_abi c dim dim.isLt)
  change Native.qOffsetBits c widths role idx i dim=Native.word (Native.qOffset c role idx i dim) at eq
  rw [eq]
  exact ⟨Native.signed _ (by omega),Native.signed _ (by omega)⟩

/-- Exact signed offsets and four-byte scaling for every formed payload and
page-table lane, including false-mask lanes, follow from the checked budgets. -/
theorem all_formed (c : Call) (d : Dimensions) (v : Views) (A : FlatAlloc) (B : Bytes.Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (guard : InputContract c d v A.extent requestedMax start sequence prompt request pages)
    (before : Bytes.Memory) (observed : Bytes.Observed c B before start sequence prompt request pages)
    (storage : StorageBinding c v B) (needed : Bytes.Needed c)
    (idx : GridIndex c.grid) (step : Nat) (i : Fin c.layout.BM) (j : Fin c.layout.BN) (dim : Fin c.layout.D)
    (ht : step < Native.countAt c idx) :
    (∀ role,role=Native.DataRole.query ∨ role=.output →
      (Native.qOffsetBits c widths role idx i dim).toInt=(Native.qOffset c role idx i dim : Int) ∧
      (Native.word (4*Native.qOffset c role idx i dim)).toInt=(4*Native.qOffset c role idx i dim : Nat)) ∧
    (∀ role,role=Native.DataRole.key ∨ role=.value →
      (Native.kvOffsetBits c widths role idx step j dim).toInt=(Native.kvOffset c role idx step j dim : Int) ∧
      (Native.word (4*Native.kvOffset c role idx step j dim)).toInt=(4*Native.kvOffset c role idx step j dim : Nat)) := by
  have abi := abi c d v A B requestedMax start sequence prompt request pages guard before observed storage needed
  have qr := formed_rows c d A.extent requestedMax guard.2.1 guard.2.2.1.2.1 idx i
  have kr := formed_page c d A.extent guard.2.1 idx step j
  have qBound := payload_fits _ _ _ v.q (guard.2.2.1.2.2.2.2.1 v.q (by simp)) _ _ dim.val
    qr (Host.head_bound c idx) dim.isLt
  have kvHead : Formula.head c.layout (c.launchAt idx)/c.layout.kvGroup < c.kvHeads := by
    apply (Nat.div_lt_iff_lt_mul guard.2.1.2.2.2.2.1).mpr
    rw [←guard.2.1.2.2.2.2.2.1]
    exact Host.head_bound c idx
  have kBound := payload_fits _ _ _ v.k (guard.2.2.1.2.2.2.2.2.1 v.k (by simp)) _ _ dim.val kr kvHead dim.isLt
  rcases guard.2.1.1 with ⟨qrow,qhead,qdim,orow,ohead,odim,krow,khead,kdim,vrow,vhead,vdim⟩
  constructor
  · intro role choice
    have bound : Native.qOffset c role idx i dim < 2^61 := by
      rcases choice with rfl|rfl <;>
        simpa [Native.qOffset,Native.DataRole.row,Native.DataRole.head,Native.DataRole.dim,
          qrow,qhead,qdim,orow,ohead,odim] using qBound
    have eq := Native.query_modular c.layout (c.launchAt idx) widths.start _ _ _ (role.row c) (role.head c) (role.dim c)
      (Native.block_abi c A widths (nativeBytes B) abi idx) (Native.batchHead_abi c A widths (nativeBytes B) abi idx)
      c.blockM abi.heads abi.group (abi.start _ (Host.batch_bound c idx)) i.isLt (Native.dim_abi c dim dim.isLt)
    change Native.qOffsetBits c widths role idx i dim=Native.word (Native.qOffset c role idx i dim) at eq
    rw [eq]
    exact ⟨Native.signed _ (by omega),Native.signed _ (by omega)⟩
  · intro role choice
    have bound : Native.kvOffset c role idx step j dim < 2^61 := by
      rcases choice with rfl|rfl <;>
        simpa [Native.kvOffset,Native.DataRole.row,Native.DataRole.head,Native.DataRole.dim,
          krow,khead,kdim,vrow,vhead,vdim] using kBound
    have eq := Native.value_modular c.layout (c.launchAt idx) (Native.mathPage c idx step j) dim
      (role.row c) (role.head c) (role.dim c) (Native.batchHead_abi c A widths (nativeBytes B) abi idx)
      abi.heads abi.group (Native.dim_abi c dim dim.isLt)
    unfold Native.kvOffsetBits
    rw [Native.pageBits_exact c A widths (nativeBytes B) abi idx step j ht,eq]
    exact ⟨Native.signed (Native.kvOffset c role idx step j dim) (by omega),Native.signed _ (by omega)⟩

theorem page_formed (c : Call) (d : Dimensions) (v : Views) (A : FlatAlloc) (B : Bytes.Windows A.extent)
    (requestedMax : Nat) (start sequence prompt request pages : Nat → Int)
    (guard : InputContract c d v A.extent requestedMax start sequence prompt request pages)
    (before : Bytes.Memory) (observed : Bytes.Observed c B before start sequence prompt request pages)
    (storage : StorageBinding c v B) (needed : Bytes.Needed c)
    (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN) (ht : step < Native.countAt c idx) :
    let off := c.layout.reqRow*c.request (Native.batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j.val)
    (Native.pageOffsetBits c.layout (c.launchAt idx) widths.request (c.request (Native.batchAt c idx)) step j).toInt=(off : Int) ∧
    (Native.word (4*off)).toInt=(4*off : Nat) := by
  dsimp only
  have abi := abi c d v A B requestedMax start sequence prompt request pages guard before observed storage needed
  have req := (guard.2.1.2.2.2.2.2.2.1 ⟨Native.batchAt c idx,Host.batch_bound c idx⟩).2.2.2.1
  have token := formed_key_row c d A.extent guard.2.1 idx step j ht
  have mul := Nat.mul_le_mul_right d.tokens (Nat.le_sub_one_of_lt req)
  have budget := guard.2.2.1.2.2.2.2.2.2.2.1
  have bound : c.layout.reqRow*c.request (Native.batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j.val) < 2^61 := by
    rw [guard.2.1.2.1,guard.2.1.2.2.1,Nat.one_mul,Nat.mul_comm d.tokens]
    change c.request (Native.batchAt c idx) < d.requests at req
    change c.request (Native.batchAt c idx)*d.tokens≤(d.requests-1)*d.tokens at mul
    omega
  rw [Native.page_modular c.layout (c.launchAt idx) widths.request _ step j
    (by rw [c.blockN]; exact c.blockM) (abi.request _ (Host.batch_bound c idx)) j.isLt]
  exact ⟨Native.signed (c.layout.reqRow*c.request (Native.batchAt c idx)+c.layout.reqToken*(step*c.layout.BN+j.val))
    (by omega),Native.signed _ (by omega)⟩

/-- The widened source retains its original minimum and BN-step loop. Its
end-expression and even the final signed backedge are within int64; the
historical counted-loop bridge therefore needs no overflow workaround here. -/
theorem direct_control (block bm sequence prompt cur : Nat)
    (hb : block < 2^31) (hm : bm=64 ∨ bm=128) (hs : sequence < 2^31)
    (hp : prompt≤sequence) (hc : cur≤sequence+bm) :
    (Native.smin (Native.word (block*bm)+Native.word bm+Native.word prompt) (Native.word sequence)).toInt=
      (min (block*bm+bm+prompt) sequence : Nat) ∧
    (Native.word cur+Native.word bm).toInt=(cur+bm : Nat) := by
  simp only [Native.word,BitVec.ofNat_add_ofNat]
  change (Native.smin (Native.word (block*bm+bm+prompt)) (Native.word sequence)).toInt=_ ∧
    (Native.word (cur+bm)).toInt=_
  have endBound : block*bm+bm+prompt < 2^63 := by rcases hm with rfl|rfl <;> omega
  rw [Native.smin_word _ _ endBound (by omega)]
  refine ⟨Native.signed _ (lt_of_le_of_lt (min_le_right _ _) (by omega)),Native.signed _ ?_⟩
  rcases hm with rfl|rfl <;> omega

#axiomsClean all_formed
#axiomsClean query_formed
#axiomsClean page_formed
#axiomsClean direct_control
end
end VeriTile.Bench.Repairs.ContextAttnLlama.Checked
