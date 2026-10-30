import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.StorageBridge
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.FormedBounds

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom Math Program RealSetup
open FPSuccess (Call)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 24000
abbrev word64 (x : Int) := BitVec.ofInt 64 x

def coordinateBits (row head dim rowStride headStride dimStride : Int) : BitVec 64 :=
  word64 row*word64 rowStride+word64 head*word64 headStride+word64 dim*word64 dimStride

theorem coordinate_modular (row head dim rowStride headStride dimStride : Int) :
    coordinateBits row head dim rowStride headStride dimStride=
      word64 (row*rowStride+head*headStride+dim*dimStride) := by
  simp only [coordinateBits,word64,←BitVec.ofInt_mul,←BitVec.ofInt_add]

theorem signed64 (x : Int) (lo : -(2^63) ≤ x) (hi : x < 2^63) : (word64 x).toInt=x :=
  BitVec.toInt_ofInt_eq_self (by decide) lo hi

theorem signed_offset (x : Int) (lo : 0 ≤ x) (hi : x < 2^61) :
    (word64 x).toInt=x ∧ (word64 4*word64 x).toInt=4*x := by
  refine ⟨signed64 x (by omega) (by omega),?_⟩
  rw [←BitVec.ofInt_mul]
  exact signed64 (4*x) (by omega) (by omega)

theorem payload_fits (rows heads width padded : Nat) (v : View)
    (budget : payloadBudget rows heads width padded v) (row head dim : Nat)
    (hr : row < rows) (hh : head < heads) (hd : dim < padded) :
    row*(heads*width)+head*width+dim < 2^61 := by
  let desc : VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3 :=
    ⟨rows,heads,padded,heads*width,width,1,Bool.false⟩
  have h := VeriTile.Bench.Repairs.ContextAttnFwd.FormedBounds.bound3 desc v.storageOffset row head dim
    ⟨by simpa [desc] using budget.1,budget.2.1,budget.2.2,by norm_num [desc]⟩ hr hh hd
  simpa [desc] using h

theorem block_small (a : Host.Arguments) : Host.block a ≤ 128 := by
  simp only [Host.block]; split <;> omega

theorem padded_small (a : Host.Arguments) (hi : a.q.dims ≤ 256) : Host.paddedDim a ≤ 256 := by
  have clog : Nat.clog 2 a.q.dims ≤ 8 := Nat.clog_le_of_le_pow (by simpa using hi)
  have pow : 2^Nat.clog 2 a.q.dims ≤ 2^8 := Nat.pow_le_pow_right (by decide) clog
  unfold Host.paddedDim
  omega

theorem query_rows (c : Call) (m : metadataChecks c.args c.input) (requestedMax : Nat)
    (maximum : c.args.maxInput ≤ requestedMax) (idx : GridIndex (Host.grid c.args))
    (i : Fin (Host.layout c.args).BM) :
    0 ≤ c.input.start (Host.launchOf c.args idx).batch+queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.val ∧
    (c.input.start (Host.launchOf c.args idx).batch+queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.val).toNat <
      c.args.q.rows+requestedMax+128 := by
  let l := Host.launchOf c.args idx
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have start : 0 ≤ c.input.start l.batch := md.1
  have prompt : 0 ≤ c.input.prompt l.batch := md.2.1
  have seq : c.input.prompt l.batch ≤ c.input.sequence l.batch := md.2.2.1
  have cap : c.input.start l.batch+(c.input.sequence l.batch-c.input.prompt l.batch) ≤ c.args.q.rows := md.2.2.2.2.1
  have bl : l.block < Bounds.ceilBlocks c.args.maxInput (Host.block c.args) :=
    (idx ⟨2,by simp [Host.grid,Grid.rank]⟩).isLt
  have bs := (Bounds.lt_ceilBlocks _ _ _ (Host.block_positive c.args)).mp bl
  have bm := block_small c.args
  have lane : i.val < Host.block c.args := i.isLt
  change 0 ≤ c.input.start l.batch+(l.block*Host.block c.args+i.val : Nat) ∧
    (c.input.start l.batch+(l.block*Host.block c.args+i.val : Nat)).toNat < _
  constructor <;> omega

/-- Signed offsets cover every formed Q/Out lane, even CTAs with no key loop
and padded channels that never access memory. -/
theorem query_formed (c : Call) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c.args c.regions extent) (m : metadataChecks c.args c.input)
    (n : nativeChecks c.args v requestedMax) (idx : GridIndex (Host.grid c.args))
    (i : Fin (Host.layout c.args).BM) (d : Fin (Host.layout c.args).D) :
    let row := c.input.start (Host.launchOf c.args idx).batch+queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.val
    let head := (Host.launchOf c.args idx).head
    (∀ desc∈[c.args.q,c.args.o],
      (coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=desc.address row head d.val ∧
      (word64 4*coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=4*desc.address row head d.val) := by
  dsimp only
  have rows := query_rows c m requestedMax n.2.1 idx i
  have bound := payload_fits _ _ _ _ v.q (n.2.2.2.2.2.1 v.q (by simp))
    _ _ d.val rows.2 (Host.head_lt c.args idx) d.isLt
  rcases g with ⟨strides,hq,hk,grouped,qk,kv,ok,oh,vh,orows,rest⟩
  rcases strides with ⟨qc,kc,vc,oc,rr,rt⟩
  have rowcast := Int.toNat_of_nonneg rows.1
  intro desc member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  have fit : 0 ≤ desc.address (c.input.start (Host.launchOf c.args idx).batch+
      queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.val) (Host.launchOf c.args idx).head d.val ∧
      desc.address (c.input.start (Host.launchOf c.args idx).batch+
      queryRow (Host.layout c.args) (Host.launchOf c.args idx) i.val) (Host.launchOf c.args idx).head d.val < 2^61 := by
    rcases member with rfl|rfl
    all_goals
      simp only [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3.address,
        qc.1,qc.2.1,qc.2.2,oc.1,oc.2.1,oc.2.2,oh,ok,←qk,Nat.cast_one,mul_one]
      rw [←rowcast]
      constructor
      · positivity
      · exact_mod_cast bound
  rw [coordinate_modular]
  exact signed_offset _ fit.1 fit.2

/-- Every reached gather tile forms at most one masked tail beyond sequence. -/
theorem key_column (c : Call) (m : metadataChecks c.args c.input)
    (idx : GridIndex (Host.grid c.args)) (step : Nat) (j : Fin (Host.layout c.args).BN)
    (reached : step < count (Host.layout c.args) (Host.launchOf c.args idx) c.input) :
    step*(Host.layout c.args).BN+j.val < c.args.requestColumns+128 := by
  let p := Host.layout c.args
  let l := Host.launchOf c.args idx
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have cap : c.input.sequence l.batch ≤ c.args.requestColumns := md.2.2.2.2.2.1
  have loop := (Bounds.loop_test p l c.input step (Host.block_positive c.args)).mp reached
  have stop : Bounds.stop p l c.input ≤ c.input.sequence l.batch := by
    unfold Bounds.stop
    split_ifs
    · exact min_le_right _ _
    · have pp : 0 ≤ c.input.prompt l.batch := md.2.1
      have ps : c.input.prompt l.batch ≤ c.input.sequence l.batch := md.2.2.1
      omega
  have bn : p.BN ≤ 128 := block_small c.args
  have jl : j.val < p.BN := j.isLt
  change step*p.BN+j.val < c.args.requestColumns+128
  omega

/-- A false page mask uses exactly zero, whose formed K/V pointers are still
covered even when the cache has zero rows. -/
def gathered (c : Call) (idx : GridIndex (Host.grid c.args)) (step j : Nat) : Int :=
  if ((step*(Host.layout c.args).BN+j : Nat) : Int) < limit (Host.layout c.args) (Host.launchOf c.args idx) c.input then
    token (Host.layout c.args) (Host.launchOf c.args idx) c.input (step*(Host.layout c.args).BN+j) else 0

theorem gathered_bound (c : Call) (m : metadataChecks c.args c.input)
    (strides : sourceStrides c.args) (idx : GridIndex (Host.grid c.args)) (step j : Nat) :
    0 ≤ gathered c idx step j ∧ (gathered c idx step j).toNat < max c.args.k.rows 1 := by
  let l := Host.launchOf c.args idx
  let column := step*(Host.layout c.args).BN+j
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have rq0 : 0 ≤ c.input.request l.batch := md.2.2.2.2.2.2.1
  unfold gathered
  split_ifs with active
  · have seq : (column : Int) < c.input.sequence l.batch := lt_of_lt_of_le active (min_le_right _ _)
    have page := md.2.2.2.2.2.2.2.2 ⟨column,by change column < (c.input.sequence l.batch).toNat; omega⟩
    have eq : token (Host.layout c.args) l c.input column=
        c.input.tokens (c.args.requestColumns*(c.input.request l.batch).toNat+column) := by
      unfold token Host.layout
      rw [strides.2.2.2.2.1,strides.2.2.2.2.2]
      simp only [Nat.cast_one,mul_one]
      congr 1
      rw [←Int.toNat_of_nonneg rq0]
      simp only [←Nat.cast_mul,←Nat.cast_add,Int.toNat_natCast,Nat.mul_comm]
    change 0 ≤ token (Host.layout c.args) l c.input column ∧ _
    rw [eq]
    constructor
    · exact page.1
    · have hi : c.input.tokens (c.args.requestColumns*(c.input.request l.batch).toNat+column) < c.args.k.rows := page.2
      omega
  · simp


theorem cache_formed (c : Call) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c.args c.regions extent) (m : metadataChecks c.args c.input)
    (n : nativeChecks c.args v requestedMax) (idx : GridIndex (Host.grid c.args))
    (step j : Nat) (d : Fin (Host.layout c.args).D) :
    let row := gathered c idx step j
    let head := (Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup
    ∀ desc∈[c.args.k,c.args.v],
      (coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=desc.address row head d.val ∧
      (word64 4*coordinateBits row head d.val desc.rowStride desc.headStride desc.dimStride).toInt=4*desc.address row head d.val := by
  dsimp only
  have rows := gathered_bound c m g.1 idx step j
  rcases g with ⟨strides,hq,hk,grouped,qk,kv,ok,oh,vh,orows,vrows,rest⟩
  have gp : 0 < c.args.q.heads/c.args.k.heads := Nat.div_pos (Nat.le_of_dvd hq grouped) hk
  have kh : (Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup < c.args.k.heads := by
    apply (Nat.div_lt_iff_lt_mul gp).mpr
    simpa only [Nat.mul_div_cancel' grouped] using Host.head_lt c.args idx
  have bound := payload_fits _ _ _ _ v.k (n.2.2.2.2.2.2.1 v.k (by simp))
    _ _ d.val rows.2 kh d.isLt
  rcases strides with ⟨qc,kc,vc,oc,rr,rt⟩
  have rowcast := Int.toNat_of_nonneg rows.1
  intro desc member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  have fit : 0 ≤ desc.address (gathered c idx step j)
      ((Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup) d.val ∧
      desc.address (gathered c idx step j)
      ((Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup) d.val < 2^61 := by
    rcases member with rfl|rfl
    all_goals
      simp only [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.View3.address,
        kc.1,kc.2.1,kc.2.2,vc.1,vc.2.1,vc.2.2,vh,←kv,Nat.cast_one,mul_one]
      rw [←rowcast]
      constructor
      · positivity
      · exact_mod_cast bound
  rw [coordinate_modular]
  exact signed_offset _ fit.1 fit.2

theorem page_formed (c : Call) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c.args c.regions extent) (m : metadataChecks c.args c.input)
    (n : nativeChecks c.args v requestedMax) (idx : GridIndex (Host.grid c.args))
    (step : Nat) (j : Fin (Host.layout c.args).BN)
    (reached : step < count (Host.layout c.args) (Host.launchOf c.args idx) c.input) :
    let off := AddressBounds.tokenAddress (Host.layout c.args) (Host.launchOf c.args idx) c.input
      (step*(Host.layout c.args).BN+j.val)
    (word64 off).toInt=off ∧ (word64 4*word64 off).toInt=4*off := by
  dsimp only
  let l := Host.launchOf c.args idx
  let column := step*(Host.layout c.args).BN+j.val
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have rq0 : 0 ≤ c.input.request l.batch := md.2.2.2.2.2.2.1
  have rq : (c.input.request l.batch).toNat < c.args.requestRows := by
    have hi : c.input.request l.batch < c.args.requestRows := md.2.2.2.2.2.2.2.1
    omega
  have col : column < c.args.requestColumns+128 := key_column c m idx step j reached
  have mul := Nat.mul_le_mul_right c.args.requestColumns (Nat.le_sub_one_of_lt rq)
  have budget := n.2.2.2.2.2.2.2.2.1
  have bound : (c.input.request l.batch).toNat*c.args.requestColumns+column < 2^61 := by omega
  apply signed_offset
  · unfold AddressBounds.tokenAddress Host.layout
    have rr := g.1.2.2.2.2.1
    have rt := g.1.2.2.2.2.2
    rw [rr,rt]
    positivity
  · change (c.args.requestRowStride : Int)*c.input.request l.batch+
      (c.args.requestTokenStride : Int)*column < _
    rw [g.1.2.2.2.2.1,g.1.2.2.2.2.2,←Int.toNat_of_nonneg rq0]
    simpa only [Nat.cast_add,Nat.cast_mul,Nat.cast_one,one_mul,mul_comm] using
      (show (((c.input.request l.batch).toNat*c.args.requestColumns+column : Nat) : Int)<2^61 by exact_mod_cast bound)

theorem min_exact (a b : Int) (ha : (word64 a).toInt=a) (hb : (word64 b).toInt=b) :
    (if (word64 a).slt (word64 b) then word64 a else word64 b).toInt=min a b := by
  have test : (word64 a).slt (word64 b)=decide (a<b) := by simp only [BitVec.slt,ha,hb]
  rw [test]
  by_cases h : a < b
  · simp [h,ha,min_eq_left (le_of_lt h)]
  · simp [h,hb,min_eq_right (le_of_not_gt h)]

/-- Bounds for the actual signed min, block mask, loop stop and BN-sized final
backedge. Integer metadata is read from the original int32 byte words. -/
theorem direct_control (c : Call) (v : Views) (requestedMax : Nat)
    (m : metadataChecks c.args c.input) (n : nativeChecks c.args v requestedMax)
    (B : Bytes.Windows extent) (before : Bytes.Memory) (observed : Bytes.Observed c B before)
    (idx : GridIndex (Host.grid c.args)) :
    let l := Host.launchOf c.args idx
    let p := Host.layout c.args
    let upper := ((l.block+1)*p.BM : Nat)+c.input.prompt l.batch
    let sequence := c.input.sequence l.batch
    ((if (word64 upper).slt (word64 sequence) then word64 upper else word64 sequence).toInt=min upper sequence) ∧
    ∀ cur : Nat,(cur : Int) ≤ sequence+p.BN → (word64 cur+word64 p.BN).toInt=(cur+p.BN : Nat) := by
  dsimp only
  let l := Host.launchOf c.args idx
  let p := Host.layout c.args
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have prompt0 : 0 ≤ c.input.prompt l.batch := md.2.1
  have ps : c.input.prompt l.batch ≤ c.input.sequence l.batch := md.2.2.1
  have int32 : c.input.sequence l.batch < 2^31 := by
    change c.input=RealSetup.inputs (Bytes.layoutState B before) c.regions at observed
    rw [observed]
    change (Bytes.load32 before (B.base c.regions.sequence+4*l.batch)).toInt < 2^31
    exact BitVec.toInt_lt
  have block : l.block < 65535 := lt_of_lt_of_le
    (idx ⟨2,by simp [Host.grid,Grid.rank]⟩).isLt n.2.2.2.2.1
  have bm : p.BM ≤ 128 := block_small c.args
  have bn : p.BN ≤ 128 := block_small c.args
  have bm0 : 0 < p.BM := Host.block_positive c.args
  have blockProduct : (l.block+1)*p.BM ≤ 65535*128 := Nat.mul_le_mul (by omega) bm
  have upper := signed64 (((l.block+1)*p.BM : Nat)+c.input.prompt l.batch) (by omega) (by omega)
  have sequence := signed64 (c.input.sequence l.batch) (by omega) (by omega)
  constructor
  · exact min_exact _ _ upper sequence
  · intro cur hcur
    change (cur : Int) ≤ c.input.sequence l.batch+p.BN at hcur
    change (word64 cur+word64 p.BN).toInt=(cur+p.BN : Nat)
    rw [←BitVec.ofInt_add]
    simpa only [Nat.cast_add] using signed64 ((cur : Int)+(p.BN : Int)) (by omega) (by omega)


theorem grouped_head (c : Call) (v : Views) (requestedMax : Nat)
    (n : nativeChecks c.args v requestedMax) (idx : GridIndex (Host.grid c.args)) :
    ((word64 (Host.launchOf c.args idx).head).sdiv (word64 (Host.layout c.args).kvGroup)).toInt=
      ((Host.launchOf c.args idx).head/(Host.layout c.args).kvGroup : Nat) := by
  have hh := Host.head_lt c.args idx
  have heads := n.2.2.2.1
  have group : (Host.layout c.args).kvGroup ≤ c.args.q.heads := Nat.div_le_self _ _
  have eq := VeriTile.Bench.Repairs.ContextAttnFwd.Native.sdiv_eq
    (Host.launchOf c.args idx).head (Host.layout c.args).kvGroup (by omega) (by omega)
  simp only [word64,BitVec.ofInt_natCast]
  rw [eq]
  apply VeriTile.Bench.Repairs.ContextAttnFwd.Native.signed
  have h := Nat.div_le_self (Host.launchOf c.args idx).head (Host.layout c.args).kvGroup
  omega

theorem pointer_exact (B : Bytes.Windows extent) (rg : RegionName) (off : Nat)
    (inside : off < extent rg) (j : Fin 4) :
    (BitVec.ofNat 64 (B.base rg)+BitVec.ofNat 64 4*BitVec.ofNat 64 off+BitVec.ofNat 64 j.val).toNat=
      B.base rg+4*off+j.val := by
  have room := B.fits rg
  have lane := j.isLt
  simp only [←BitVec.ofNat_mul,←BitVec.ofNat_add,BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  omega


theorem source_masks (c : Call) (v : Views) (extent : RegionBounds) (requestedMax : Nat)
    (g : geometryChecks c.args c.regions extent) (m : metadataChecks c.args c.input)
    (n : nativeChecks c.args v requestedMax) (B : Bytes.Windows extent) (before : Bytes.Memory)
    (observed : Bytes.Observed c B before) (idx : GridIndex (Host.grid c.args))
    (i : Fin (Host.layout c.args).BM) (d : Fin (Host.layout c.args).D) :
    let l := Host.launchOf c.args idx
    let row := queryRow (Host.layout c.args) l i.val
    ((word64 row).slt (word64 (c.input.sequence l.batch)-word64 (c.input.prompt l.batch))=
      decide ((row : Int)<c.input.sequence l.batch-c.input.prompt l.batch)) ∧
    ((word64 d.val).slt (word64 c.args.q.dims)=decide (d.val<c.args.q.dims)) ∧
    ∀ column : Nat,(column : Int) ≤ c.input.sequence l.batch+128 →
      (word64 column).sle (word64 row+word64 (c.input.prompt l.batch))=
        decide ((column : Int) ≤ row+c.input.prompt l.batch) := by
  dsimp only
  let l := Host.launchOf c.args idx
  let row := queryRow (Host.layout c.args) l i.val
  have md := m.1 ⟨l.batch,Host.batch_lt c.args idx⟩
  have pp : 0 ≤ c.input.prompt l.batch := md.2.1
  have ps : c.input.prompt l.batch ≤ c.input.sequence l.batch := md.2.2.1
  have seq32 : c.input.sequence l.batch < 2^31 := by
    change c.input=RealSetup.inputs (Bytes.layoutState B before) c.regions at observed
    rw [observed]
    change (Bytes.load32 before (B.base c.regions.sequence+4*l.batch)).toInt < 2^31
    exact BitVec.toInt_lt
  have block : l.block < 65535 := lt_of_lt_of_le
    (idx ⟨2,by simp [Host.grid,Grid.rank]⟩).isLt n.2.2.2.2.1
  have bm := block_small c.args
  have iBound : i.val < Host.block c.args := i.isLt
  have product : l.block*Host.block c.args ≤ 65535*128 := Nat.mul_le_mul (by omega) bm
  have rowBound : row < 65535*128+128 := by change l.block*Host.block c.args+i.val < _; omega
  have geometry := g
  rcases geometry with ⟨_,_,_,_,_,_,_,_,_,_,_,_,widthHi,_⟩
  have dimBound : d.val < 256 := lt_of_lt_of_le d.isLt (padded_small c.args widthHi)
  have rowExact := signed64 (row : Int) (by omega) (by omega)
  have lenExact := signed64 (c.input.sequence l.batch-c.input.prompt l.batch) (by omega) (by omega)
  refine ⟨?_,?_,?_⟩
  · change (word64 row).slt (word64 (c.input.sequence l.batch)-word64 (c.input.prompt l.batch))=
      decide ((row : Int)<c.input.sequence l.batch-c.input.prompt l.batch)
    rw [BitVec.sub_eq_add_neg,←BitVec.ofInt_neg,←BitVec.ofInt_add]
    rw [←Int.sub_eq_add_neg]
    simp only [BitVec.slt,rowExact,lenExact]
  · rw [BitVec.slt,signed64 d.val (by omega) (by omega),signed64 c.args.q.dims (by omega) (by omega)]
    simp
  · intro column hcolumn
    change (column : Int) ≤ c.input.sequence l.batch+128 at hcolumn
    change (word64 column).sle (word64 row+word64 (c.input.prompt l.batch))=
      decide ((column : Int) ≤ (row : Int)+c.input.prompt l.batch)
    rw [←BitVec.ofInt_add]
    simp only [BitVec.sle,signed64 column (by omega) (by omega),
      signed64 ((row : Int)+c.input.prompt l.batch) (by omega) (by omega)]


theorem padded_eight (a : Host.Arguments) (hi : a.q.dims ≤ 256) : Host.paddedDim a % 8=0 := by
  have hc : Nat.clog 2 a.q.dims ≤ 8 := Nat.clog_le_of_le_pow (by simpa using hi)
  generalize he : Nat.clog 2 a.q.dims=k at *
  interval_cases k <;> norm_num [Host.paddedDim,he]

theorem block_eight (a : Host.Arguments) : Host.block a % 8=0 := by
  unfold Host.block
  split <;> norm_num

#axiomsClean padded_eight
#axiomsClean block_eight
#axiomsClean source_masks
#axiomsClean grouped_head
#axiomsClean pointer_exact
#axiomsClean cache_formed
#axiomsClean page_formed
#axiomsClean direct_control
#axiomsClean query_formed
#axiomsClean key_column
#axiomsClean gathered_bound
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.Checked
