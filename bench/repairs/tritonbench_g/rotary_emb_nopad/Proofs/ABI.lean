import bench.repairs.tritonbench_g.rotary_emb_nopad.Proofs.Whole

namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Checked

theorem formed_pointer (c : Call) (index : Indices) (accepted : check c index=true)
    (w : Work) (active : c.request.Active w) (i : Fin 5)
    (pi : present c.request ⟨i.val,by omega⟩=true) (offset : Nat)
    (inside : offset≤lastOffset c.request ⟨i.val,by omega⟩) :
    4*offset+4<2^63 ∧ 4*base c.request ⟨i.val,by omega⟩+4*offset+4≤2^64 := by
  have checked := launch_checked c index accepted w active
  simp only [launchCheck,Bool.and_eq_true,decide_eq_true_eq] at checked
  have bound := checked.2 i pi
  constructor <;> omega

theorem tensor_extent (c : Call) (index : Indices) (accepted : check c index=true)
    (i : Fin 7) (pi : present c.request i=true) (offset : Nat)
    (inside : offset<wordExtent c.request i) :
    (c.tensor i).base=4*base c.request i ∧ (c.tensor i).base%4=0 ∧
    4*(base c.request i+offset)+4≤(c.tensor i).allocationBase+(c.tensor i).allocationBytes ∧
    4*(base c.request i+offset)+4≤2^64 := by
  have facts := tensor_facts c index accepted i pi
  have nonempty : (c.tensor i).shape.prod≠0 := by
    rw [facts.1]
    intro zero
    simp [wordExtent,zero] at inside
  have f := facts.2.2.resolve_left nonempty
  have bytes : 4*wordExtent c.request i≤(c.tensor i).allocationBytes := by
    have fit := f.2.2.2.1
    rw [facts.1,facts.2.1] at fit
    unfold wordExtent
    split_ifs at * <;> omega
  exact ⟨f.2.1,by omega,by omega,by omega⟩

theorem input_pointer (c : Call) (index : Indices) (accepted : check c index=true)
    (w : Work) (active : c.request.Active w) (odd : Bool) :
    4*c.request.data w.key w.token w.head (c.request.dim w.pair odd)+4≤
      (c.tensor (role w.key)).allocationBase+(c.tensor (role w.key)).allocationBytes ∧
    4*c.request.data w.key w.token w.head (c.request.dim w.pair odd)+4≤2^64 := by
  have range := input_range c.request w active odd
  have bound := tensor_extent c index accepted (role w.key)
    (by cases w.key <;> simp [present,role])
    (c.request.data w.key w.token w.head (c.request.dim w.pair odd)-base c.request (role w.key)) (by omega)
  have eq : base c.request (role w.key)+(c.request.data w.key w.token w.head (c.request.dim w.pair odd)-base c.request (role w.key))=
      c.request.data w.key w.token w.head (c.request.dim w.pair odd) := by omega
  rw [eq] at bound
  exact bound.2.2

theorem destination_pointer (c : Call) (index : Indices) (accepted : check c index=true)
    (w : Work) (active : c.request.Active w) (odd : Bool) :
    4*destination c.request index w odd+4≤
      (c.tensor (destinationRole c.request w.key)).allocationBase+
      (c.tensor (destinationRole c.request w.key)).allocationBytes ∧
    4*destination c.request index w odd+4≤2^64 := by
  have range := destination_range c index accepted w active odd
  change base c.request (destinationRole c.request w.key)≤destination c.request index w odd ∧
    destination c.request index w odd<base c.request (destinationRole c.request w.key)+wordExtent c.request (destinationRole c.request w.key) at range
  have bound := tensor_extent c index accepted (destinationRole c.request w.key)
    (destination_present c.request w.key)
    (destination c.request index w odd-base c.request (destinationRole c.request w.key)) (by omega)
  have eq : base c.request (destinationRole c.request w.key)+
      (destination c.request index w odd-base c.request (destinationRole c.request w.key))=destination c.request index w odd := by omega
  rw [eq] at bound
  exact bound.2.2

theorem padded_data (r : Request) (key odd : Bool) (token head pair : Nat)
    (ht : token<paddedTokens r) (hh : head<r.qHeads) (hp : pair<r.blockPairs) :
    r.data key token head (r.dim pair odd)≤base r (role key)+lastOffset r (role key) := by
  have token := Nat.mul_le_mul_right (r.heads key*r.width) (show token≤paddedTokens r-1 by omega)
  have head := Nat.mul_le_mul_right r.width (show head≤r.qHeads-1 by omega)
  have dim : r.dim pair odd≤r.half+r.blockPairs-1 := by cases odd <;> simp [Request.dim] <;> omega
  cases key <;> simp [Request.data,Request.heads,role,base,lastOffset,Nat.add_mul,Nat.mul_assoc] at * <;> omega

theorem padded_coefficient (r : Request) (sine : Bool) (token pair : Nat)
    (ht : token<paddedTokens r) (hp : pair<r.blockPairs) :
    r.coefficient sine token pair≤base r (if sine then 3 else 2)+lastOffset r (if sine then 3 else 2) := by
  have cos := Nat.mul_le_mul_right r.cosWidth (show token≤paddedTokens r-1 by omega)
  have sin := Nat.mul_le_mul_right r.sinWidth (show token≤paddedTokens r-1 by omega)
  cases sine <;> simp [Request.coefficient,base,lastOffset] at * <;> omega


theorem padded_cache (r : Request) (odd : Bool) (block head slot pair : Nat)
    (hb : block<r.blocks) (hh : head<r.qHeads) (hs : slot<r.blockSize) (hp : pair<r.blockPairs) :
    r.cacheBase+((block*r.kHeads+head)*r.blockSize+slot)*r.width+r.dim pair odd≤
      base r 4+lastOffset r 4 := by
  have blockBound := Nat.mul_le_mul_right (r.kHeads*r.blockSize*r.width) (show block≤r.blocks-1 by omega)
  have headBound := Nat.mul_le_mul_right (r.blockSize*r.width) (show head≤r.qHeads-1 by omega)
  have slotBound := Nat.mul_le_mul_right r.width (show slot≤r.blockSize-1 by omega)
  have dim : r.dim pair odd≤r.half+r.blockPairs-1 := by cases odd <;> simp [Request.dim] <;> omega
  change r.cacheBase+((block*r.kHeads+head)*r.blockSize+slot)*r.width+r.dim pair odd≤
    r.cacheBase+((r.blocks-1)*r.kHeads*r.blockSize*r.width+(r.qHeads-1)*r.blockSize*r.width+
      (r.blockSize-1)*r.width+r.half+r.blockPairs-1)
  simp only [Nat.add_mul,Nat.mul_assoc]
  simp only [Nat.mul_assoc] at blockBound headBound
  omega

theorem metadata_arithmetic (c : Call) (index : Indices) (accepted : check c index=true)
    (w : Work) (active : c.request.Active w) (cached : c.request.cached=true) :
    (c.request.past index w.token : Int)=index (c.request.lengthBase+2*w.token)-1 ∧
    0 ≤ index (c.request.lengthBase+2*w.token)-1 ∧
    index (c.request.lengthBase+2*w.token)-1<2^63 ∧
    ((c.request.past index w.token/c.request.blockSize : Nat) : Int)=
      (index (c.request.lengthBase+2*w.token)-1)/(c.request.blockSize : Int) ∧
    (c.request.slot index w.token : Int)=
      (index (c.request.lengthBase+2*w.token)-1)%(c.request.blockSize : Int) := by
  have checked := metadata_checked c index accepted w active
  simp only [metadataCheck,cached,↓reduceIte,Bool.and_eq_true,decide_eq_true_eq] at checked
  have length := checked.1 ⟨w.token,active.1⟩
  dsimp only at length
  have nonnegative : 0 ≤ index (c.request.lengthBase+2*w.token)-1 := by omega
  have past : (c.request.past index w.token : Int)=index (c.request.lengthBase+2*w.token)-1 :=
    Int.toNat_of_nonneg nonnegative
  exact ⟨past,nonnegative,by omega,by rw [Int.natCast_ediv,past],by rw [Request.slot,Int.natCast_emod,past]⟩

#axiomsClean padded_cache
#axiomsClean metadata_arithmetic
#axiomsClean formed_pointer
#axiomsClean tensor_extent
#axiomsClean input_pointer
#axiomsClean destination_pointer
#axiomsClean padded_data
#axiomsClean padded_coefficient
end VeriTile.Bench.Repairs.RotaryEmbNopad.Checked
