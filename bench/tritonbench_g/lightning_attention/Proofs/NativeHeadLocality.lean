import bench.tritonbench_g.lightning_attention.Proofs.NativeLocality

namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeLocality
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]

structure HeadCovered (scope : RegionName → Nat → Prop)
    (p : Parameters) (head : Nat) : Prop where
  query : ∀ t a, t<p.N → a<p.D → scope "query" (head*p.N*p.D+t*p.D+a)
  key : ∀ t a, t<p.N → a<p.D → scope "key" (head*p.N*p.D+t*p.D+a)
  value : ∀ t a, t<p.N → a<p.E → scope "value" (head*p.N*p.E+t*p.E+a)
  incoming : ∀ t a, t<p.N → a<p.E → scope "incoming" (head*p.N*p.E+t*p.E+a)
  queryGradient : ∀ t a, t<p.N → a<p.D → scope "queryGradient" (head*p.N*p.D+t*p.D+a)
  keyGradient : ∀ t a, t<p.N → a<p.D → scope "keyGradient" (head*p.N*p.D+t*p.D+a)
  valueGradient : ∀ t a, t<p.N → a<p.E → scope "valueGradient" (head*p.N*p.E+t*p.E+a)

theorem intraQ_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    intraQ M p head block a = intraQ M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.query _ _ hi i.1.isLt)

theorem intraK_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    intraK M p head block a = intraK M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.key _ _ hi i.2.1.isLt)

theorem intraV_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    intraV M p head block a = intraV M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.value _ _ hi i.1.isLt)

theorem intraG_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    intraG M p head block a = intraG M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.incoming _ _ hi i.2.1.isLt)

theorem subK_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    subK M p head block a = subK M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.key _ _ hi i.2.1.isLt)

theorem subVT_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    subVT M p head block a = subVT M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.value _ _ hi i.1.isLt)

theorem subQT_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    subQT M p head block a = subQT M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.query _ _ hi i.1.isLt)

theorem subG_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    subG M p head block a = subG M p head block b := by
  funext i
  exact read_agree scope M a b h _ _ _ (fun hi => hc.incoming _ _ hi i.2.1.isLt)

theorem gradientRows_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (r : RegionName) (N width head row rows : Nat)
    (hc : ∀ t c, t<N → c<width → scope r (head*N*width+t*width+c))
    (a b : Memory α) (h : Agree scope a b) :
    gradientRows M r N width head row rows a = gradientRows M r N width head row rows b := by
  funext i
  unfold gradientRows
  split
  · rename_i hi; rw [h r _ (hc _ _ hi i.2.1.isLt)]
  · rfl

theorem gradientTrans_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (r : RegionName) (N width head row rows : Nat)
    (hc : ∀ t c, t<N → c<width → scope r (head*N*width+t*width+c))
    (a b : Memory α) (h : Agree scope a b) :
    gradientTrans M r N width head row rows a = gradientTrans M r N width head row rows b := by
  funext i
  unfold gradientTrans
  split
  · rename_i hi; rw [h r _ (hc _ _ hi i.1.isLt)]
  · rfl

theorem writeRows_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (r : RegionName) (N width head start rows : Nat) (values : Matrix α rows width)
    (a b : Memory α) (h : Agree scope a b) :
    Agree scope (writeRows M r N width head start rows values a)
      (writeRows M r N width head start rows values b) :=
  scatter_agree scope _ _ _ _ _ a b h

theorem writeTrans_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (r : RegionName) (N width head start rows : Nat) (values : Matrix α width rows)
    (a b : Memory α) (h : Agree scope a b) :
    Agree scope (writeTrans M r N width head start rows values a)
      (writeTrans M r N width head start rows values b) :=
  scatter_agree scope _ _ _ _ _ a b h

theorem intraMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head block : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) :
    Agree scope (intraMemory M D p head block a) (intraMemory M D p head block b) := by
  simp only [intraMemory,intraDQ,intraDKT,intraDV,intraDS,intraST,
    intraQ_agree scope M p head hc a b h,intraK_agree scope M p head hc a b h,
    intraV_agree scope M p head hc a b h,intraG_agree scope M p head hc a b h]
  exact writeRows_agree scope _ _ _ _ _ _ _ _ _ _
    (writeTrans_agree scope _ _ _ _ _ _ _ _ _ _
      (writeRows_agree scope _ _ _ _ _ _ _ _ _ _ h))

theorem queryUpdate_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head row : Nat) (hc : HeadCovered scope p head)
    (carry : Matrix α p.E p.D) (a b : Memory α) (h : Agree scope a b) :
    Agree scope (queryUpdate M D p head row carry a) (queryUpdate M D p head row carry b) := by
  simp only [queryUpdate,subG_agree scope M p head hc a b h,
    gradientRows_agree scope M _ _ _ _ _ _ hc.queryGradient a b h]
  exact writeRows_agree scope _ _ _ _ _ _ _ _ _ _ h

theorem keyValueUpdate_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head row : Nat) (hc : HeadCovered scope p head)
    (carry : Matrix α p.D p.E) (a b : Memory α) (h : Agree scope a b) :
    Agree scope (keyValueUpdate M D p head row carry a) (keyValueUpdate M D p head row carry b) := by
  simp only [keyValueUpdate,subVT_agree scope M p head hc a b h,
    subK_agree scope M p head hc a b h,
    gradientRows_agree scope M _ _ _ _ _ _ hc.valueGradient a b h,
    gradientTrans_agree scope M _ _ _ _ _ _ hc.keyGradient a b h]
  exact writeRows_agree scope _ _ _ _ _ _ _ _ _ _
    (writeTrans_agree scope _ _ _ _ _ _ _ _ _ _ h)

theorem queryBlockMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head block : Nat) (hc : HeadCovered scope p head)
    (carry : Matrix α p.E p.D) (a b : Memory α) (h : Agree scope a b) :
    Agree scope (queryBlockMemory M D p head block carry a) (queryBlockMemory M D p head block carry b) := by
  unfold queryBlockMemory
  split
  · exact h
  · exact fold_rel (Agree scope) _ _ _ a b h
      (fun _ _ _ hab => queryUpdate_agree scope M D p head _ hc carry _ _ hab)

theorem keyValueBlockMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head block : Nat) (hc : HeadCovered scope p head)
    (carry : Matrix α p.D p.E) (a b : Memory α) (h : Agree scope a b) :
    Agree scope (keyValueBlockMemory M D p head block carry a) (keyValueBlockMemory M D p head block carry b) := by
  unfold keyValueBlockMemory
  split
  · exact h
  · exact fold_rel (Agree scope) _ _ _ a b h
      (fun _ _ _ hab => keyValueUpdate_agree scope M D p head _ hc carry _ _ hab)

theorem queryBlockDelta_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head block : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) :
    queryBlockDelta M D p head block a = queryBlockDelta M D p head block b := by
  simp only [queryBlockDelta,subVT_agree scope M p head hc a b h,
    subK_agree scope M p head hc a b h]

theorem keyValueBlockDelta_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head block : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) :
    keyValueBlockDelta M D p head block a = keyValueBlockDelta M D p head block b := by
  simp only [keyValueBlockDelta,subQT_agree scope M p head hc a b h,
    subG_agree scope M p head hc a b h]

theorem queryCarry_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    queryCarry M D p head a block = queryCarry M D p head b block := by
  induction block with
  | zero => rfl
  | succ block ih =>
    simp only [queryCarry,ih,queryBlockDelta_agree scope M D p head block hc a b h]

theorem keyValueCarry_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b : Memory α) (h : Agree scope a b) (block : Nat) :
    keyValueCarry M D p head a block = keyValueCarry M D p head b block := by
  induction block with
  | zero => rfl
  | succ block ih =>
    simp only [keyValueCarry,ih,keyValueBlockDelta_agree scope M D p head block hc a b h]

theorem queryScanMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b x y : Memory α) (h : Agree scope a b) (hxy : Agree scope x y) :
    Agree scope (queryScanMemory M D p head a x) (queryScanMemory M D p head b y) := by
  apply fold_rel (Agree scope) _ _ _ _ _ hxy
  intro j u v huv
  rw [queryCarry_agree scope M D p head hc a b h j]
  exact queryBlockMemory_agree scope M D p head j hc _ u v huv

theorem keyValueScanMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b x y : Memory α) (h : Agree scope a b) (hxy : Agree scope x y) :
    Agree scope (keyValueScanMemory M D p head a x) (keyValueScanMemory M D p head b y) := by
  apply fold_rel (Agree scope) _ _ _ _ _ hxy
  intro j u v huv
  rw [keyValueCarry_agree scope M D p head hc a b h j]
  exact keyValueBlockMemory_agree scope M D p head j hc _ u v huv

theorem interMemory_agree (scope : RegionName → Nat → Prop) (M : Algebra α)
    (D : DotPrimitive α) (p : Parameters) (head : Nat) (hc : HeadCovered scope p head)
    (a b x y : Memory α) (h : Agree scope a b) (hxy : Agree scope x y) :
    Agree scope (interMemory M D p head a x) (interMemory M D p head b y) :=
  keyValueScanMemory_agree scope M D p head hc a b _ _ h
    (queryScanMemory_agree scope M D p head hc a b x y h hxy)

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.NativeLocality

