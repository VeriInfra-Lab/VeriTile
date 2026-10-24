import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Admission
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.Ownership
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
noncomputable section
namespace Checked

theorem flat3_injective (R C h r j h' r' j' : Nat)
    (hr : r<R) (hr' : r'<R) (hj : j<C) (hj' : j'<C)
    (eq : h*(R*C)+r*C+j=h'*(R*C)+r'*C+j') : h=h' ∧ r=r' ∧ j=j' := by
  have e := digit_injective C (h*R+r) j (h'*R+r') j' hj hj'
    (by simpa [Nat.add_mul,Nat.mul_assoc] using eq)
  have f := digit_injective R h r h' r' hr hr' e.1
  exact ⟨f.1,f.2,e.2⟩

theorem flat4_injective (S R C h s r j h' s' r' j' : Nat)
    (hs : s<S) (hs' : s'<S) (hr : r<R) (hr' : r'<R) (hj : j<C) (hj' : j'<C)
    (eq : h*(S*(R*C))+s*(R*C)+r*C+j=h'*(S*(R*C))+s'*(R*C)+r'*C+j') :
    h=h' ∧ s=s' ∧ r=r' ∧ j=j' := by
  have e := flat3_injective R C (h*S+s) r j (h'*S+s') r' j' hr hr' hj hj'
    (by simpa [Nat.add_mul,Nat.mul_assoc,Nat.add_assoc] using eq)
  have f := digit_injective S h s h' s' hs hs' e.1
  exact ⟨f.1,f.2,e.2.1,e.2.2⟩

theorem offset_injective (c : Shape) (a b : Address) (ha : a.Valid c) (hb : b.Valid c)
    (region : a.region=b.region) (eq : a.offset c=b.offset c) : a=b := by
  rcases a with ⟨r,h,s,i,j⟩
  rcases b with ⟨r',h',s',i',j'⟩
  cases region
  cases r <;> simp only [Address.Valid,Address.offset] at ha hb eq
  all_goals first
  | have e := flat3_injective c.T c.K h i j h' i' j' ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2 eq
    rcases e with ⟨rfl,rfl,rfl⟩
    simp_all
  | have e := flat3_injective c.T c.V h i j h' i' j' ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2 eq
    rcases e with ⟨rfl,rfl,rfl⟩
    simp_all
  | have e := flat3_injective c.K c.V h i j h' i' j' ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2 eq
    rcases e with ⟨rfl,rfl,rfl⟩
    simp_all
  | have e := c.history_offset_injective h s i j h' s' i' j'
      ha.2.1 hb.2.1 ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2 eq
    rcases e with ⟨rfl,rfl,rfl,rfl⟩
    rfl
  | have e := flat4_injective c.heads c.T c.V s h i j s' h' i' j'
      ha.1 hb.1 ha.2.2.1 hb.2.2.1 ha.2.2.2 hb.2.2.2
      (by simpa [Shape.partialOffset,Shape.valueOffset,Shape.valueSize,Shape.valueHead,Nat.mul_assoc,Nat.add_assoc] using eq)
    rcases e with ⟨rfl,rfl,rfl,rfl⟩
    rfl

def regionShape (c : Shape) : Region → List Nat
  | .q | .k | .dq | .dk => [c.B,c.H,c.T,c.K]
  | .v | .incoming | .output | .dv => [c.B,c.H,c.T,c.V]
  | .seed | .terminal | .finalState | .initialGradient => [c.B,c.H,c.K,c.V]
  | .history | .dhistory => [c.B,c.H,c.chunks*c.K,c.V]
  | .dvPart => [c.keyBlocks,c.B,c.H,c.T,c.V]

theorem regionShape_prod (c : Shape) (r : Region) : (regionShape c r).prod=r.capacity c := by
  cases r <;> simp [regionShape,Region.capacity,Shape.keySize,Shape.valueSize,Shape.stateSize,
    Shape.historySize,Shape.partialSize,Shape.heads,Shape.keyHead,Shape.valueHead,Shape.stateHead,
    Shape.historyHead,Nat.mul_assoc]

/-- The dense logical interval law is an ordinary contiguous Tensor API fact.
It admits arbitrary strides on singleton axes and does not assert any value. -/
def DenseMetadata (t : Contract.Tensor) : Prop :=
  Contract.lastOffset t t.metadata.shape=t.offset+(t.metadata.numel:Int)-1

theorem tensor_span (t : Contract.Tensor) (shape : List Nat) (device : VeriTile.Torch.Device)
    (checked : Contract.TensorContract t shape device Bool.false Bool.true)
    (ordinary : DenseMetadata t) (positive : 0<shape.prod) :
    t.metadata.dataPtr.toNat%4=0 ∧
    t.base≤t.metadata.dataPtr.toNat ∧
    t.metadata.dataPtr.toNat+4*shape.prod≤t.base+t.bytes ∧
    t.metadata.dataPtr.toNat+4*shape.prod≤2^64 := by
  rcases checked with ⟨_,sh,_,_,_,_,_,_,_,_,back⟩
  rcases back with ⟨_,off,align,upper,pointer,_,last⟩
  unfold DenseMetadata at ordinary
  rw [sh] at ordinary
  simp only [VeriTile.Torch.TensorMetadata.numel,sh] at ordinary
  rw [ordinary] at last
  have bytes := Nat.div_mul_le_self t.bytes 4
  have ptr : t.metadata.dataPtr.toNat=t.base+4*t.offset.toNat := by omega
  have span : t.metadata.dataPtr.toNat+4*shape.prod≤t.base+t.bytes := by omega
  refine ⟨?_,by omega,span,by omega⟩
  rw [ptr]
  omega

def regionNumber : Region → Nat
  | .q => 0 | .k => 1 | .v => 2 | .seed => 3 | .incoming => 4 | .terminal => 5
  | .history => 6 | .finalState => 7 | .output => 8 | .dhistory => 9 | .dq => 10
  | .dk => 11 | .dvPart => 12 | .dv => 13 | .initialGradient => 14
theorem regionNumber_injective : Function.Injective regionNumber := by
  intro a b h
  cases a <;> cases b <;> simp_all [regionNumber]

def ghostSpan (c : Shape) := c.keySize+c.valueSize+c.stateSize+c.historySize+c.partialSize+1
theorem capacity_lt_ghost (c : Shape) (r : Region) : r.capacity c<ghostSpan c := by
  cases r <;> simp only [Region.capacity,ghostSpan] <;> omega

/-- Present descriptors are actual Tensor observations. Freshness is at the
whole backing-allocation level: allocator facts for new results, and the
checked disjointness clause for a supplied d_h0. Read-only aliases are allowed.
Absent optional arguments use derived ghost addresses and perform no load. -/
structure Storage (c : Shape) where
  tensor : Region → Contract.Tensor
  device : VeriTile.Torch.Device
  present : Region → Bool
  zeroInput : Region → Bool
  zeroOnly : ∀ r,zeroInput r=Bool.true → r∈[Region.seed,Region.incoming,Region.terminal]
  checked : ∀ r,present r=Bool.true →
    Contract.TensorContract (tensor r) (regionShape c r) device Bool.false Bool.true
  dense : ∀ r,present r=Bool.true → DenseMetadata (tensor r)
  fresh : ∀ r s,r≠s → present r=Bool.true → present s=Bool.true →
    (r∉inputRegions ∨ s∉inputRegions) → Contract.disjoint (tensor r) (tensor s)

def Storage.base (a : Storage c) (r : Region) : Nat :=
  if a.present r then (a.tensor r).metadata.dataPtr.toNat/4 else 2^64+regionNumber r*ghostSpan c

theorem Storage.live_span (a : Storage c) (r : Region) (live : a.present r=Bool.true)
    (positive : 0<r.capacity c) :
    4*a.base r=(a.tensor r).metadata.dataPtr.toNat ∧
    (a.tensor r).base≤4*a.base r ∧
    4*(a.base r+r.capacity c)≤(a.tensor r).base+(a.tensor r).bytes ∧
    a.base r+r.capacity c≤2^62 := by
  have p : 0<(regionShape c r).prod := by rwa [regionShape_prod]
  obtain ⟨align,lo,hi,up⟩ := tensor_span (a.tensor r) _ _ (a.checked r live) (a.dense r live) p
  rw [regionShape_prod] at hi up
  simp only [Storage.base,live,↓reduceIte]
  have div := Nat.div_add_mod (a.tensor r).metadata.dataPtr.toNat 4
  omega

theorem Storage.cell_live (a : Storage c) (x : Address) (hx : x.Valid c)
    (live : a.present x.region=Bool.true) :
    (a.tensor x.region).base≤4*(a.base x.region+x.offset c) ∧
    4*(a.base x.region+x.offset c)+4≤(a.tensor x.region).base+(a.tensor x.region).bytes ∧
    a.base x.region+x.offset c<2^62 := by
  have bound := x.offset_bound c hx
  obtain ⟨_,lo,hi,up⟩ := a.live_span x.region live (by omega)
  omega

theorem Storage.injective (a : Storage c) (x y : Address) (hx : x.Valid c) (hy : y.Valid c)
    (writable : x.region∉inputRegions ∨ y.region∉inputRegions)
    (eq : a.base x.region+x.offset c=a.base y.region+y.offset c) : x=y := by
  by_cases same : x.region=y.region
  · apply offset_injective c x y hx hy same
    rw [same] at eq
    omega
  have bx := x.offset_bound c hx
  have boundY := y.offset_bound c hy
  cases ex : a.present x.region <;> cases ey : a.present y.region
  · have gx := capacity_lt_ghost c x.region
    have gy := capacity_lt_ghost c y.region
    simp only [Storage.base,ex,ey,Bool.false_eq_true,↓reduceIte] at eq
    have pair := digit_injective (ghostSpan c) (regionNumber x.region) (x.offset c)
      (regionNumber y.region) (y.offset c) (by omega) (by omega) (by omega)
    exact False.elim (same (regionNumber_injective pair.1))
  · have live := (a.cell_live y hy ey).2.2
    simp only [Storage.base,ex,ey,Bool.false_eq_true,↓reduceIte] at eq live
    omega
  · have live := (a.cell_live x hx ex).2.2
    simp only [Storage.base,ex,ey,Bool.false_eq_true,↓reduceIte] at eq live
    omega
  · obtain ⟨lox,hix,_⟩ := a.cell_live x hx ex
    obtain ⟨loy,hiy,_⟩ := a.cell_live y hy ey
    have sep := a.fresh _ _ same ex ey writable
    unfold Contract.disjoint at sep
    omega

def Storage.binding (a : Storage c) : Binding c where
  base := a.base
  zeroInput := a.zeroInput
  zeroOnlyOptional := a.zeroOnly
  injective := a.injective

#axiomsClean offset_injective
#axiomsClean tensor_span
#axiomsClean Storage.injective
end Checked
end
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
