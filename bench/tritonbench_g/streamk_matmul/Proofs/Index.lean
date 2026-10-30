import bench.tritonbench_g.streamk_matmul.Proofs.ByteHost

namespace VeriTile.Bench.Repairs.StreamK.Checked.Index
open Contract TensorBinding Whole001 Source VeriTile.Bench.TritonBenchG.StreamK
noncomputable section
set_option maxHeartbeats 4000000

/-- The native signed index expression, retaining all i64 operations. -/
def offset (row strideRow col strideCol : Nat) : BitVec 64 :=
  BitVec.ofNat 64 row*BitVec.ofNat 64 strideRow+BitVec.ofNat 64 col*BitVec.ofNat 64 strideCol

theorem offset_nat (row strideRow col strideCol : Nat) (h : row*strideRow+col*strideCol<2^63) :
    (offset row strideRow col strideCol).toNat=row*strideRow+col*strideCol := by
  simp only [offset,BitVec.toNat_add,BitVec.toNat_mul,BitVec.toNat_ofNat]
  simp only [←Nat.mul_mod,←Nat.add_mod]
  exact Nat.mod_eq_of_lt (by omega)

theorem offset_signed (row strideRow col strideCol : Nat) (h : row*strideRow+col*strideCol<2^63) :
    (offset row strideRow col strideCol).toInt=row*strideRow+col*strideCol := by
  have hn := offset_nat row strideRow col strideCol h
  rw [BitVec.toInt_eq_toNat_of_lt (by rw [hn]; omega),hn]
  simp

theorem lanes_bounded (p : Parameters) (tile : Nat) (ht : tile<Plan.tiles p.h) (lane : Lane p) :
    row p tile lane<ceilDiv p.h.m p.h.bm*p.h.bm ∧
    col p tile lane<ceilDiv p.h.n p.h.bn*p.h.bn := by
  have hn : 0<ceilDiv p.h.n p.h.bn := by
    unfold Plan.tiles at ht
    by_contra hn
    have hz : ceilDiv p.h.n p.h.bn=0 := by omega
    simp [hz] at ht
  have grid := VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.GroupedGrid.bounds
    (ceilDiv p.h.m p.h.bm) (ceilDiv p.h.n p.h.bn) 8 tile hn (by decide) ht
  have lr := lane.1.isLt
  have lc := lane.2.isLt
  constructor
  · change _*p.h.bm+lane.1.val<ceilDiv p.h.m p.h.bm*p.h.bm
    have hm := Nat.mul_le_mul_right p.h.bm (Nat.succ_le_of_lt grid.1)
    rw [Nat.succ_mul] at hm
    change (let first := tile/(8*ceilDiv p.h.n p.h.bn)*8
      first+tile%min (ceilDiv p.h.m p.h.bm-first) 8)*p.h.bm+p.h.bm≤_ at hm
    omega
  · change _*p.h.bn+lane.2.val<ceilDiv p.h.n p.h.bn*p.h.bn
    have hm := Nat.mul_le_mul_right p.h.bn (Nat.succ_le_of_lt grid.2)
    rw [Nat.succ_mul] at hm
    change ((tile%(8*ceilDiv p.h.n p.h.bn))/min (ceilDiv p.h.m p.h.bm-tile/(8*ceilDiv p.h.n p.h.bn)*8) 8)*p.h.bn+p.h.bn≤_ at hm
    dsimp only
    omega

/-- Covers inactive tail pointers and the one-past loop pointer advances, not
only the masked dereferences. The assumptions are integer loop/lane bounds. -/
theorem pointers_signed (p : Parameters) (valid : ABI.Valid p)
    (tile : Nat) (ht : tile<Plan.tiles p.h) (lane : Lane p) (k : Nat)
    (hk : k≤((Plan.select p.h).iterations+1)*p.h.bk) :
    (offset (row p tile lane) p.strideAM k p.strideAK).toInt=
      row p tile lane*p.strideAM+k*p.strideAK ∧
    (offset k p.strideBK (col p tile lane) p.strideBN).toInt=
      k*p.strideBK+col p tile lane*p.strideBN := by
  have bounds := lanes_bounded p tile ht lane
  constructor
  · apply offset_signed
    have rowBound := Nat.mul_le_mul_right p.strideAM (Nat.le_of_lt bounds.1)
    have kBound := Nat.mul_le_mul_right p.strideAK hk
    have envelope := valid.pointerA
    omega
  · apply offset_signed
    have colBound := Nat.mul_le_mul_right p.strideBN (Nat.le_of_lt bounds.2)
    have kBound := Nat.mul_le_mul_right p.strideBK hk
    have envelope := valid.pointerB
    omega

def nativeLoad (read : ValueRead α) (zero : α) (base rows cols sr sc row col : Nat) : α :=
  if (BitVec.ofNat 64 row).toInt<(BitVec.ofNat 64 rows).toInt ∧
     (BitVec.ofNat 64 col).toInt<(BitVec.ofNat 64 cols).toInt then
    read (ByteExecution.address base (offset row sr col sc).toNat)
  else zero

theorem load_eq (read : ValueRead α) (zero : α) (base rows cols sr sc row col : Nat)
    (contiguous : Geometry.Contiguous rows cols sr sc)
    (rowsBound : rows<2^63) (colsBound : cols<2^63) (rowBound : row<2^63) (colBound : col<2^63)
    (bytes : 4*(rows*cols)<2^63) (window : base+4*(rows*cols)≤2^64) :
    nativeLoad read zero base rows cols sr sc row col=
      if h:row<rows ∧ col<cols then read (base+4*(row*sr+col*sc)) else zero := by
  unfold nativeLoad
  rw [Envelope.signed64 row rowBound,Envelope.signed64 col colBound,
    Envelope.signed64 rows rowsBound,Envelope.signed64 cols colsBound]
  simp only [Nat.cast_lt]
  by_cases active : row<rows ∧ col<cols
  · rw [if_pos active,dif_pos active]
    have live := Geometry.input_mask_safe rows cols sr sc row col contiguous active.1 active.2
    rw [offset_nat _ _ _ _ (by omega),ByteExecution.address_eq base _ (rows*cols) live window]
  · rw [if_neg active,dif_neg active]

/-- The input readers consumed by every fixed MMA operand are the actual
original physical windows, with signed masks and finite pointer arithmetic. -/
theorem input_loads (a b : Tensor) (config : Config) (h : InputContract a.metadata b.metadata config)
    (nonempty : ¬Empty a.metadata b.metadata) (read : ValueRead α) (zero : α)
    (tile : Nat) (ht : tile<Plan.tiles (plan a.metadata b.metadata config))
    (lane : Lane (parameters a b config h.profile)) (k : Nat)
    (hk : k<advancedK (plan a.metadata b.metadata config)) :
    let p := parameters a b config h.profile
    nativeLoad read zero (Host.base a) p.h.m p.h.k p.strideAM p.strideAK (row p tile lane) k=
      loadA p zero (fun i => read (Host.base a+4*i.val)) (row p tile lane) k ∧
    nativeLoad read zero (Host.base b) p.h.k p.h.n p.strideBK p.strideBN k (col p tile lane)=
      loadB p zero (fun i => read (Host.base b+4*i.val)) k (col p tile lane) := by
  dsimp only
  let p := parameters a b config h.profile
  have valid := Envelope.valid a b config h nonempty
  change ABI.Valid p at valid
  have lanes := lanes_bounded p tile ht lane
  obtain ⟨control,stream,bytes,wa,wb⟩ := execution_nonempty a.metadata b.metadata config h nonempty
  have ba := bytes.2.2.2.2.1
  have bb := bytes.2.2.2.2.2.1
  have kb : k<2^63 := by
    have advance := valid.advanceA
    change ((Plan.select p.h).iterations+1)*p.h.bk*1<2^63 at advance
    change k<((Plan.select p.h).iterations+1)*p.h.bk at hk
    omega
  have rb : row p tile lane<2^63 := lt_trans lanes.1 valid.tiledRows
  have cb : col p tile lane<2^63 := lt_trans lanes.2 valid.tiledCols
  have numa : a.metadata.numel=p.h.m*p.h.k := matrix_numel a.metadata h.inputs.2.2.1
  have numb : b.metadata.numel=p.h.k*p.h.n := by
    rw [matrix_numel b.metadata h.inputs.2.2.2.1,←h.inputs.2.2.2.2.2.2.2.2.2.2.2.2.2]
    rfl
  change nativeLoad read zero (Host.base a) p.h.m p.h.k p.strideAM p.strideAK (row p tile lane) k=
    loadA p zero (fun i => read (Host.base a+4*i.val)) (row p tile lane) k ∧
    nativeLoad read zero (Host.base b) p.h.k p.h.n p.strideBK p.strideBN k (col p tile lane)=
    loadB p zero (fun i => read (Host.base b+4*i.val)) k (col p tile lane)
  have dimM : p.h.m<2^63 := lt_trans valid.dimensions.1 (by norm_num)
  have dimN : p.h.n<2^63 := lt_trans valid.dimensions.2.1 (by norm_num)
  have dimK : p.h.k<2^63 := lt_trans valid.dimensions.2.2 (by norm_num)
  constructor
  · by_cases empty : a.metadata.numel=0
    · have hz : p.h.m*p.h.k=0 := numa.symm.trans empty
      have inactive : ¬(row p tile lane<p.h.m ∧ k<p.h.k) := by rcases Nat.mul_eq_zero.mp hz with hz|hz <;> omega
      simp only [nativeLoad,Envelope.signed64 _ rb,Envelope.signed64 _ kb,
        Envelope.signed64 _ dimM,
        Envelope.signed64 _ dimK,Nat.cast_lt,if_neg inactive,loadA,dif_neg inactive]
    · have window := (wa.resolve_left empty).2
      rw [numa] at window
      exact load_eq read zero (Host.base a) _ _ _ _ _ _ p.inputA dimM dimK rb kb ba window
  · by_cases empty : b.metadata.numel=0
    · have hz : p.h.k*p.h.n=0 := numb.symm.trans empty
      have inactive : ¬(k<p.h.k ∧ col p tile lane<p.h.n) := by rcases Nat.mul_eq_zero.mp hz with hz|hz <;> omega
      simp only [nativeLoad,Envelope.signed64 _ kb,Envelope.signed64 _ cb,
        Envelope.signed64 _ dimK,
        Envelope.signed64 _ dimN,Nat.cast_lt,if_neg inactive,loadB,dif_neg inactive]
    · have window := (wb.resolve_left empty).2
      rw [numb] at window
      exact load_eq read zero (Host.base b) _ _ _ _ _ _ p.inputB dimK dimN kb cb bb window

theorem fold_same (xs : List β) (f g : α → β → α)
    (same : ∀ x∈xs,∀ seed,f seed x=g seed x) (seed : α) : xs.foldl f seed=xs.foldl g seed := by
  induction xs generalizing seed with
  | nil => rfl
  | cons x tail ih =>
    simp only [List.foldl_cons,same x (by simp)]
    exact ih (fun y hy => same y (by simp [hy])) _

/-- Actual block instruction graph reading the physical A/B pointers through
signed masks and i64 pointer arithmetic. This is executable operand routing,
not a numerical assumption. -/
def nativeBlock (a b : Tensor) (config : Config) (profile : Profile config)
    (ops : Instructions α) (read : ValueRead α) (tile iter : Nat)
    (lane : Lane (parameters a b config profile)) (seed : α) : α :=
  let p := parameters a b config profile
  (List.range ((p.h.bk+7)/8)).foldl (fun acc group =>
    ops.mma8
      (fun j => nativeLoad read ops.zero (Host.base a) p.h.m p.h.k p.strideAM p.strideAK (row p tile lane)
        ((iter%(Plan.select p.h).iterations)*p.h.bk+group*8+j.val))
      (fun j => nativeLoad read ops.zero (Host.base b) p.h.k p.h.n p.strideBK p.strideBN
        ((iter%(Plan.select p.h).iterations)*p.h.bk+group*8+j.val) (col p tile lane)) acc) seed

theorem nativeBlock_eq (a b : Tensor) (config : Config) (h : InputContract a.metadata b.metadata config)
    (nonempty : ¬Empty a.metadata b.metadata) (ops : Instructions α) (read : ValueRead α)
    (tile iter : Nat) (ht : tile<Plan.tiles (plan a.metadata b.metadata config))
    (hI : 0<(Plan.select (plan a.metadata b.metadata config)).iterations)
    (lane : Lane (parameters a b config h.profile)) (seed : α) :
    nativeBlock a b config h.profile ops read tile iter lane seed=
      block (parameters a b config h.profile) ops
        (fun i => read (Host.base a+4*i.val)) (fun i => read (Host.base b+4*i.val)) tile iter lane seed := by
  let p := parameters a b config h.profile
  unfold nativeBlock block
  dsimp only
  apply fold_same
  intro group hg acc
  have hg' : group<(p.h.bk+7)/8 := List.mem_range.mp hg
  have blockLength := Arithmetic.native_block_length p
  have groupBound := Nat.mul_le_mul_right 8 (Nat.succ_le_of_lt hg')
  change 0<(Plan.select p.h).iterations at hI
  have iterBound := Nat.mod_lt iter hI
  have iterMul := Nat.mul_le_mul_right p.h.bk (Nat.succ_le_of_lt iterBound)
  have positive := p.blockK
  have operandBound : ∀ j : Fin 8,
      (iter%(Plan.select p.h).iterations)*p.h.bk+group*8+j.val<advancedK p.h := by
    intro j
    have hj := j.isLt
    rw [Nat.succ_mul] at groupBound iterMul
    change _<((Plan.select p.h).iterations+1)*p.h.bk
    rw [Nat.add_mul,Nat.one_mul]
    omega
  congr 1
  · funext j
    exact (input_loads a b config h nonempty read ops.zero tile ht lane _ (operandBound j)).1
  · funext j
    exact (input_loads a b config h nonempty read ops.zero tile ht lane _ (operandBound j)).2

/-- Structural operand bridge carried by the public native contract. -/
def Conformance (a b : Tensor) (config : Config) (profile : Profile config)
    (ops : Instructions α) (read : ValueRead α) : Prop :=
  ∀ tile iter, tile<Plan.tiles (plan a.metadata b.metadata config) →
    0<(Plan.select (plan a.metadata b.metadata config)).iterations →
    ∀ lane seed,nativeBlock a b config profile ops read tile iter lane seed=
      block (parameters a b config profile) ops
        (fun i => read (Host.base a+4*i.val)) (fun i => read (Host.base b+4*i.val)) tile iter lane seed

theorem conformance (a b : Tensor) (config : Config) (h : InputContract a.metadata b.metadata config)
    (nonempty : ¬Empty a.metadata b.metadata) (ops : Instructions α) (read : ValueRead α) :
    Conformance a b config h.profile ops read := by
  intro tile iter ht hi lane seed
  exact nativeBlock_eq a b config h nonempty ops read tile iter ht hi lane seed

#axiomsClean nativeBlock_eq
#axiomsClean conformance
#axiomsClean pointers_signed
#axiomsClean input_loads
end
end VeriTile.Bench.Repairs.StreamK.Checked.Index
