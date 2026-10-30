import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.FiniteSource

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.SaturatingTile
open VeriTile.Triton FP.Structural
open Contract
noncomputable section

/-- PTX signed32 saturation, used by INT8 MMA. The candidate starts each tile
dot at zero and uses a separate add.u32 for the arbitrary running carry. -/
def sat32 (x : Int) : Int := max (-2147483648) (min 2147483647 x)

theorem sat32_eq (x : Int) (lo : -2147483648≤x) (hi : x≤2147483647) : sat32 x=x := by
  simp [sat32,min_eq_right hi,max_eq_right lo]

theorem tile_inner_bound (p : Config) (h : ConfigContract p) : p.blockK≤65536 := by
  have hm := h.1.1
  have hmk := h.2.2.1.2.1
  norm_num at hmk
  nlinarith

theorem signed8_product (a b : Int) (ha : FP.IntegerDot.signed8 a) (hb : FP.IntegerDot.signed8 b) :
    |a*b|≤16384 := by
  have aa : |a|≤128 := abs_le.mpr ⟨ha.1,by have := ha.2; omega⟩
  have ab : |b|≤128 := abs_le.mpr ⟨hb.1,by have := hb.2; omega⟩
  calc |a*b|=|a| * |b| := abs_mul a b
    _ ≤128*128 := mul_le_mul aa ab (abs_nonneg b) (by decide)
    _ =16384 := by decide

theorem sum_bound (f : Nat → Int) (n : Nat) (h : ∀ k,k<n → |f k|≤16384) :
    |LogicalSum.sum f n|≤16384*n := by
  induction n with
  | zero => simp [LogicalSum.sum]
  | succ n ih =>
    have previous := ih (fun k hk => h k (by omega))
    have last := h n (by omega)
    calc |LogicalSum.sum f (n+1)|=|LogicalSum.sum f n+f n| := rfl
      _ ≤|LogicalSum.sum f n|+|f n| := abs_add_le _ _
      _ ≤16384*(n+1) := by push_cast; omega

/-- Every subset-prefix of an admitted tile dot is within signed32. This
bound uses only the checked tile sizes and signed8 operand type. -/
theorem prefix_range (p : Config) (legal : ConfigContract p) (f : Nat → Int)
    (products : ∀ k,k<p.blockK → |f k|≤16384) (n : Nat) (hn : n≤p.blockK) :
    -1073741824≤LogicalSum.sum f n ∧ LogicalSum.sum f n≤1073741824 := by
  have hs := abs_le.mp (sum_bound f n (fun k hk => products k (by omega)))
  have hk := tile_inner_bound p legal
  have ni : (n : Int)≤65536 := by omega
  constructor <;> omega

/-- Partitioned MMA recurrence. Each entry is the number of products consumed
by one instruction; the theorem covers any instruction partition and order
whose products satisfy the signed8 bound. -/
def mmaBlocks (f : Nat → Int) : List Nat → Nat → Int → Int
  | [],_,carry => carry
  | width::rest,offset,carry =>
    mmaBlocks f rest (offset+width) (sat32 (carry+LogicalSum.sum (fun k => f (offset+k)) width))

theorem mmaBlocks_eq (p : Config) (legal : ConfigContract p) (f : Nat → Int)
    (products : ∀ k,k<p.blockK → |f k|≤16384) (partition : List Nat) (offset : Nat)
    (within : offset+partition.sum≤p.blockK) :
    mmaBlocks f partition offset (LogicalSum.sum f offset)=LogicalSum.sum f (offset+partition.sum) := by
  induction partition generalizing offset with
  | nil => simp [mmaBlocks]
  | cons width rest ih =>
    simp only [mmaBlocks]
    rw [←LogicalSum.sum_add]
    have next : offset+width≤p.blockK := by simp only [List.sum_cons] at within; omega
    have range := prefix_range p legal f products (offset+width) next
    rw [sat32_eq _ (by omega) (by omega),ih (offset+width) (by simp only [List.sum_cons] at within; omega)]
    simp only [List.sum_cons,Nat.add_assoc]

theorem tile_mma_exact (p : Config) (legal : ConfigContract p) (f : Nat → Int)
    (products : ∀ k,k<p.blockK → |f k|≤16384) (partition : List Nat)
    (cover : partition.sum=p.blockK) :
    mmaBlocks f partition 0 0=LogicalSum.sum f p.blockK := by
  simpa only [LogicalSum.sum,Nat.zero_add,cover] using mmaBlocks_eq p legal f products partition 0 (by omega)

def tileProduct (p : Config) (a : TileIndex [p.blockM,p.blockK] → Int)
    (b : TileIndex [p.blockK,p.blockN] → Int) (i : TileIndex [p.blockM,p.blockN]) (k : Nat) : Int :=
  if hk : k<p.blockK then a (i.1,⟨k,hk⟩,PUnit.unit)*b (⟨k,hk⟩,i.2.1,PUnit.unit) else 0

/-- The actual typed integer-dot value equals any saturating MMA partition
of the admitted zero-seeded tile. Thus saturation is proved inactive locally,
rather than deferred as a whole-dot numerical equality. -/
theorem source_dot_eq (p : Config) (legal : ConfigContract p)
    (a : TileIndex [p.blockM,p.blockK] → Int) (b : TileIndex [p.blockK,p.blockN] → Int)
    (ha : ∀ i,FP.IntegerDot.signed8 (a i)) (hb : ∀ i,FP.IntegerDot.signed8 (b i))
    (i : TileIndex [p.blockM,p.blockN]) (partition : List Nat) (cover : partition.sum=p.blockK) :
    (FP.IntegerDot.wordDot [] a b i).toInt=mmaBlocks (tileProduct p a b i) partition 0 0 := by
  have products : ∀ k,k<p.blockK → |tileProduct p a b i k|≤16384 := by
    intro k hk
    simp only [tileProduct,dif_pos hk]
    exact signed8_product _ _ (ha _) (hb _)
  rw [tile_mma_exact p legal _ products partition cover]
  have range := prefix_range p legal _ products p.blockK (Nat.le_refl _)
  have dot : FP.IntegerDot.wordDot [] a b i=BitVec.ofInt 32 (LogicalSum.sum (tileProduct p a b i) p.blockK) := by
    rcases i with ⟨r,c,u⟩; cases u
    change (List.finRange p.blockK).foldl (fun acc t =>
      acc+BitVec.ofInt 32 (a (r,t,PUnit.unit))*BitVec.ofInt 32 (b (t,c,PUnit.unit)))
      (BitVec.ofInt 32 (0 : Int))=_
    rw [FP.IntegerDot.fold_is_low32]
    congr 1
    have eq (acc : Int) (t : Fin p.blockK) :
        acc+a (r,t,PUnit.unit)*b (t,c,PUnit.unit)=acc+tileProduct p a b (r,c,PUnit.unit) t.val := by
      simp only [tileProduct,dif_pos t.isLt]
    simp only [eq,LogicalSum.fin_fold,Int.zero_add]
  rw [dot,BitVec.toInt_ofInt_eq_self (by decide) (by norm_num; omega) (by norm_num; omega)]

theorem wrap_signed8 (x : Int) : FP.IntegerDot.signed8 (BlockWord.wrap 8 x) := by
  exact ⟨BitVec.le_toInt _,BitVec.toInt_lt⟩

/-- Both actual masked-load paths produce precisely the signed8 operand
domain required by source_dot_eq, including zero-padded lanes. -/
theorem loaded_domains (x : FiniteExecution.Launch) (wrapped : Bool)
    (s : State FiniteExecution.Word) (k : Nat) :
    (∀ i,FP.IntegerDot.signed8 (FiniteExecution.naturalA x wrapped s k i)) ∧
    (∀ i,FP.IntegerDot.signed8 (FiniteExecution.naturalB x wrapped s k i)) := by
  exact ⟨fun _ => wrap_signed8 _,fun _ => wrap_signed8 _⟩

/-- The explicit PTX add is the finite source's modular carry update even
when the full-K result overflows. No saturation law is assumed here. -/
theorem add_u32 (carry dot : Int) :
    (BitVec.ofInt 32 carry+BitVec.ofInt 32 dot).toInt=BlockWord.wrap 32 (carry+dot) := by
  rw [←BitVec.ofInt_add]
  rfl

/-- Concrete original lowering mismatch: all signed8 inputs equal -128, with
4096 K-blocks of 32 products (K=131072). The admitted source may overflow. -/
theorem overflow_witness :
    (BitVec.ofInt 32 (4096*32*((-128 : Int)*(-128)))).toInt=-2147483648 ∧
    sat32 (4096*32*((-128 : Int)*(-128)))=2147483647 := by decide

#axiomsClean tile_inner_bound
#axiomsClean signed8_product
#axiomsClean tile_mma_exact
#axiomsClean source_dot_eq
#axiomsClean loaded_domains
#axiomsClean add_u32
#axiomsClean overflow_witness
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.SaturatingTile
