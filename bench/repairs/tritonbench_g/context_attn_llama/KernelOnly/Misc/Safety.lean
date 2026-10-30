import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.Allocation
import bench.repairs.tritonbench_g.context_attn_llama.KernelOnly.Misc.NativeOffsets

namespace VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Safety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnLlama Host Formula Program
open Checked Integer
noncomputable section
set_option maxHeartbeats 4000000

def widths : Widths := ⟨.int32,.int32,.int32,.int32,.int32⟩

theorem index_bound (r : Request) (heap : ByteMemory) (role : IndexRole) (i n : Nat)
    (eq : indexValues r heap role i=(n : Int)) : n<2^31 := by
  have bound := (load32 heap ((role.tensor r).metadata.dataPtr.toNat+4*i)).toInt_lt
  change indexValues r heap role i<2^31 at bound
  rw [eq] at bound
  exact_mod_cast bound

/-- All finite machine facts are derived from ordinary descriptors, guard
predicates and the actual signed INT32 loads; no allocator/host-success input. -/
theorem machine (r : Request) (heap : ByteMemory) (h : Contract r heap) : Machine r.call widths := by
  have maxBound := h.checks.2.2.1.1
  have limit := h.checks.2.2.1.2.1
  have gridBound := h.checks.2.2.1.2.2.2.1
  have qBudget := h.checks.2.2.1.2.2.2.2.1 (views r).q (by simp)
  have groupPositive := h.checks.2.1.2.2.2.1
  have grouped := h.checks.2.1.2.2.2.2.2.1
  have channel := r.call.channels
  have dimPos : 0<r.call.layout.D := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at channel
    rcases channel with e|e|e|e|e <;> omega
  have headsLe := Nat.mul_le_mul_left r.call.layout.heads dimPos
  have headsBound : r.call.layout.heads<2^63 := by
    simp only [Nat.mul_one] at headsLe
    have := qBudget.2.1
    omega
  have groupLe := Nat.mul_le_mul_right r.call.layout.kvGroup groupPositive
  have groupBound : r.call.layout.kvGroup<2^63 := by rw [Nat.one_mul,←grouped] at groupLe; omega
  constructor
  · rcases r.call.blockM with bm|bm <;> simp only [Call.blocks,bm] <;> omega
  · omega
  · exact headsBound
  · exact groupBound
  · intro n hn
    rcases h.checks.2.1.1 with ⟨qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd⟩
    have kBudget := h.checks.2.2.1.2.2.2.2.2.1 (views r).k (by simp)
    have tokens := h.checks.2.2.1.2.2.2.2.2.2.2.2
    simp only [qr,qh,qd,orr,oh,od,kr,kh,kd,vr,vh,vd,h.checks.2.1.2.1,h.checks.2.1.2.2.1,
      List.mem_cons,List.not_mem_nil,or_false] at hn
    have q := qBudget.2
    have k := kBudget.2
    rcases hn with e|e|e|e|e|e|e|e|e|e|e|e|e|e <;> subst n <;> omega
  · intro b hb; exact index_bound r heap .start b _ (h.checks.2.2.2.1 ⟨b,hb⟩).1
  · intro b hb; exact index_bound r heap .sequence b _ (h.checks.2.2.2.1 ⟨b,hb⟩).2.1
  · intro b hb; exact index_bound r heap .prompt b _ (h.checks.2.2.2.1 ⟨b,hb⟩).2.2.1
  · intro b hb; exact index_bound r heap .request b _ (h.checks.2.2.2.1 ⟨b,hb⟩).2.2.2.1
  · intro b hb j hj
    exact index_bound r heap .pages _ _ ((h.checks.2.2.2.1 ⟨b,hb⟩).2.2.2.2 ⟨j,hj⟩)

theorem payload_fits (rows heads dims : Nat) (v : View) (budget : payloadBudget rows heads dims v)
    (row head dim : Nat) (hr : row<rows) (hh : head<heads) (hd : dim<dims) :
    row*(heads*dims)+head*dims+dim<2^61 := by
  have rowBound := Nat.mul_le_mul_right (heads*dims) (Nat.le_sub_one_of_lt hr)
  have headBound := Nat.mul_le_mul_right dims (Nat.le_sub_one_of_lt hh)
  have dimBound := Nat.le_sub_one_of_lt hd
  have total := budget.1
  omega

theorem formed_rows (c : Call) (d : Dimensions) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c d extent) (limit : c.maxInput≤requestedMax)
    (idx : GridIndex c.grid) (lane : Fin c.layout.BM) :
    c.start (Integer.batchAt c idx)+Formula.queryRow c.layout (c.launchAt idx) lane.val < d.queryRows+requestedMax+128 := by
  have cap := (g.2.2.2.2.2.2.1 ⟨Integer.batchAt c idx,Host.batch_bound c idx⟩).1
  change c.start (Integer.batchAt c idx)+c.length (Integer.batchAt c idx)≤d.queryRows at cap
  have block := Host.block_bound c idx
  have laneBound := lane.isLt
  rcases c.blockM with bm|bm <;> simp only [Call.blocks,bm] at block laneBound <;>
    simp only [Formula.queryRow,bm] <;> omega

theorem formed_key_row (c : Call) (d : Dimensions) (extent : RegionBounds)
    (g : geometryChecks c d extent) (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN)
    (ht : step < Integer.countAt c idx) : step*c.layout.BN+j.val < d.tokens+128 := by
  have cap := (g.2.2.2.2.2.2.1 ⟨Integer.batchAt c idx,Host.batch_bound c idx⟩).2.2.1
  have stop := Integer.stop_le (c.launchAt idx).block c.layout.BM (Integer.seqAt c idx) (c.prompt (Integer.batchAt c idx))
    (by dsimp only [Integer.seqAt]; omega)
  have stepped := (Integer.ceil_step _ c.layout.BM step c.blockM).mp ht
  have lane := j.isLt
  change c.length (Integer.batchAt c idx)+c.prompt (Integer.batchAt c idx)≤d.tokens at cap
  change Integer.stopIndex _ _ _ _≤c.length (Integer.batchAt c idx)+c.prompt (Integer.batchAt c idx) at stop
  rcases c.blockM with bm|bm <;> simp only [c.blockN,bm] at stepped lane stop ⊢ <;> omega

theorem formed_page (c : Call) (d : Dimensions) (extent : RegionBounds)
    (g : geometryChecks c d extent) (idx : GridIndex c.grid) (step : Nat) (j : Fin c.layout.BN) :
    Integer.mathPage c idx step j < max d.cacheRows 1 := by
  unfold Integer.mathPage
  split_ifs with h
  · have fullBound := lt_of_lt_of_le h (Integer.end_le _ _ _ _ (by dsimp only [Integer.seqAt]; omega))
    have page := (g.2.2.2.2.2.2.1 ⟨Integer.batchAt c idx,Host.batch_bound c idx⟩).2.2.2.2 ⟨_,fullBound⟩
    exact lt_of_lt_of_le page (le_max_left _ _)
  · omega

theorem query_formed (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (idx : GridIndex r.call.grid) (i : Fin r.call.layout.BM) (dim : Fin r.call.layout.D)
    (role : Integer.DataRole) (choice : role=.query ∨ role=.output) :
    (Integer.qOffsetBits r.call widths role idx i dim).toInt=(Integer.qOffset r.call role idx i dim : Int) ∧
    (Integer.word (4*Integer.qOffset r.call role idx i dim)).toInt=(4*Integer.qOffset r.call role idx i dim : Nat) := by
  have abi := machine r heap h
  have qr := formed_rows r.call r.dimensions (extent r) r.requestedMax h.checks.2.1 h.checks.2.2.1.2.1 idx i
  have qBound := payload_fits _ _ _ (views r).q (h.checks.2.2.1.2.2.2.2.1 (views r).q (by simp)) _ _ dim.val
    qr (Host.head_bound r.call idx) dim.isLt
  rcases h.checks.2.1.1 with ⟨qrow,qhead,qdim,orow,ohead,odim,_⟩
  have bound : Integer.qOffset r.call role idx i dim < 2^61 := by
    rcases choice with rfl|rfl <;>
      simpa [Integer.qOffset,Integer.DataRole.row,Integer.DataRole.head,Integer.DataRole.dim,
        qrow,qhead,qdim,orow,ohead,odim] using qBound
  have eq := Integer.query_modular r.call.layout (r.call.launchAt idx) widths.start _ _ _ (role.row r.call) (role.head r.call) (role.dim r.call)
    (Integer.block_abi r.call widths abi idx) (Integer.batchHead_abi r.call widths abi idx)
    r.call.blockM abi.heads abi.group (abi.start _ (Host.batch_bound r.call idx)) i.isLt (Integer.dim_abi r.call dim dim.isLt)
  change Integer.qOffsetBits r.call widths role idx i dim=Integer.word (Integer.qOffset r.call role idx i dim) at eq
  rw [eq]
  exact ⟨Integer.signed _ (by omega),Integer.signed _ (by omega)⟩


theorem all_formed (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (idx : GridIndex r.call.grid) (step : Nat) (i : Fin r.call.layout.BM) (j : Fin r.call.layout.BN) (dim : Fin r.call.layout.D)
    (ht : step < Integer.countAt r.call idx) :
    (∀ role,role=Integer.DataRole.query ∨ role=.output →
      (Integer.qOffsetBits r.call widths role idx i dim).toInt=(Integer.qOffset r.call role idx i dim : Int) ∧
      (Integer.word (4*Integer.qOffset r.call role idx i dim)).toInt=(4*Integer.qOffset r.call role idx i dim : Nat)) ∧
    (∀ role,role=Integer.DataRole.key ∨ role=.value →
      (Integer.kvOffsetBits r.call widths role idx step j dim).toInt=(Integer.kvOffset r.call role idx step j dim : Int) ∧
      (Integer.word (4*Integer.kvOffset r.call role idx step j dim)).toInt=(4*Integer.kvOffset r.call role idx step j dim : Nat)) := by
  have abi := machine r heap h
  have qr := formed_rows r.call r.dimensions (extent r) r.requestedMax h.checks.2.1 h.checks.2.2.1.2.1 idx i
  have kr := formed_page r.call r.dimensions (extent r) h.checks.2.1 idx step j
  have qBound := payload_fits _ _ _ (views r).q (h.checks.2.2.1.2.2.2.2.1 (views r).q (by simp)) _ _ dim.val
    qr (Host.head_bound r.call idx) dim.isLt
  have kvHead : Formula.head r.call.layout (r.call.launchAt idx)/r.call.layout.kvGroup < r.call.kvHeads := by
    apply (Nat.div_lt_iff_lt_mul h.checks.2.1.2.2.2.2.1).mpr
    rw [←h.checks.2.1.2.2.2.2.2.1]
    exact Host.head_bound r.call idx
  have kBound := payload_fits _ _ _ (views r).k (h.checks.2.2.1.2.2.2.2.2.1 (views r).k (by simp)) _ _ dim.val kr kvHead dim.isLt
  rcases h.checks.2.1.1 with ⟨qrow,qhead,qdim,orow,ohead,odim,krow,khead,kdim,vrow,vhead,vdim⟩
  constructor
  · intro role choice
    have bound : Integer.qOffset r.call role idx i dim < 2^61 := by
      rcases choice with rfl|rfl <;>
        simpa [Integer.qOffset,Integer.DataRole.row,Integer.DataRole.head,Integer.DataRole.dim,
          qrow,qhead,qdim,orow,ohead,odim] using qBound
    have eq := Integer.query_modular r.call.layout (r.call.launchAt idx) widths.start _ _ _ (role.row r.call) (role.head r.call) (role.dim r.call)
      (Integer.block_abi r.call widths abi idx) (Integer.batchHead_abi r.call widths abi idx)
      r.call.blockM abi.heads abi.group (abi.start _ (Host.batch_bound r.call idx)) i.isLt (Integer.dim_abi r.call dim dim.isLt)
    change Integer.qOffsetBits r.call widths role idx i dim=Integer.word (Integer.qOffset r.call role idx i dim) at eq
    rw [eq]
    exact ⟨Integer.signed _ (by omega),Integer.signed _ (by omega)⟩
  · intro role choice
    have bound : Integer.kvOffset r.call role idx step j dim < 2^61 := by
      rcases choice with rfl|rfl <;>
        simpa [Integer.kvOffset,Integer.DataRole.row,Integer.DataRole.head,Integer.DataRole.dim,
          krow,khead,kdim,vrow,vhead,vdim] using kBound
    have eq := Integer.value_modular r.call.layout (r.call.launchAt idx) (Integer.mathPage r.call idx step j) dim
      (role.row r.call) (role.head r.call) (role.dim r.call) (Integer.batchHead_abi r.call widths abi idx)
      abi.heads abi.group (Integer.dim_abi r.call dim dim.isLt)
    unfold Integer.kvOffsetBits
    rw [Integer.pageBits_exact r.call widths abi idx step j ht,eq]
    exact ⟨Integer.signed (Integer.kvOffset r.call role idx step j dim) (by omega),Integer.signed _ (by omega)⟩


theorem page_formed (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (idx : GridIndex r.call.grid) (step : Nat) (j : Fin r.call.layout.BN) (ht : step < Integer.countAt r.call idx) :
    let off := r.call.layout.reqRow*r.call.request (Integer.batchAt r.call idx)+r.call.layout.reqToken*(step*r.call.layout.BN+j.val)
    (Integer.pageOffsetBits r.call.layout (r.call.launchAt idx) widths.request (r.call.request (Integer.batchAt r.call idx)) step j).toInt=(off : Int) ∧
    (Integer.word (4*off)).toInt=(4*off : Nat) := by
  dsimp only
  have abi := machine r heap h
  have req := (h.checks.2.1.2.2.2.2.2.2.1 ⟨Integer.batchAt r.call idx,Host.batch_bound r.call idx⟩).2.2.2.1
  have token := formed_key_row r.call r.dimensions (extent r) h.checks.2.1 idx step j ht
  have mul := Nat.mul_le_mul_right r.dimensions.tokens (Nat.le_sub_one_of_lt req)
  have budget := h.checks.2.2.1.2.2.2.2.2.2.2.1
  have bound : r.call.layout.reqRow*r.call.request (Integer.batchAt r.call idx)+r.call.layout.reqToken*(step*r.call.layout.BN+j.val) < 2^61 := by
    rw [h.checks.2.1.2.1,h.checks.2.1.2.2.1,Nat.one_mul,Nat.mul_comm r.dimensions.tokens]
    change r.call.request (Integer.batchAt r.call idx) < r.dimensions.requests at req
    change r.call.request (Integer.batchAt r.call idx)*r.dimensions.tokens≤(r.dimensions.requests-1)*r.dimensions.tokens at mul
    omega
  rw [Integer.page_modular r.call.layout (r.call.launchAt idx) widths.request _ step j
    (by rw [r.call.blockN]; exact r.call.blockM) (abi.request _ (Host.batch_bound r.call idx)) j.isLt]
  exact ⟨Integer.signed (r.call.layout.reqRow*r.call.request (Integer.batchAt r.call idx)+r.call.layout.reqToken*(step*r.call.layout.BN+j.val))
    (by omega),Integer.signed _ (by omega)⟩


/-- The widened source retains its original minimum and BN-step loop. Its
end-expression and even the final signed backedge are within int64; the
historical counted-loop bridge therefore needs no overflow workaround here. -/
theorem direct_control (block bm sequence prompt cur : Nat)
    (hb : block < 2^31) (hm : bm=64 ∨ bm=128) (hs : sequence < 2^31)
    (hp : prompt≤sequence) (hc : cur≤sequence+bm) :
    (Integer.smin (Integer.word (block*bm)+Integer.word bm+Integer.word prompt) (Integer.word sequence)).toInt=
      (min (block*bm+bm+prompt) sequence : Nat) ∧
    (Integer.word cur+Integer.word bm).toInt=(cur+bm : Nat) := by
  simp only [Integer.word,BitVec.ofNat_add_ofNat]
  change (Integer.smin (Integer.word (block*bm+bm+prompt)) (Integer.word sequence)).toInt=_ ∧
    (Integer.word (cur+bm)).toInt=_
  have endBound : block*bm+bm+prompt < 2^63 := by rcases hm with rfl|rfl <;> omega
  rw [Integer.smin_word _ _ endBound (by omega)]
  refine ⟨Integer.signed _ (lt_of_le_of_lt (min_le_right _ _) (by omega)),Integer.signed _ ?_⟩
  rcases hm with rfl|rfl <;> omega

#axiomsClean all_formed
#axiomsClean query_formed
#axiomsClean page_formed
#axiomsClean direct_control
end
end VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly.Safety
