import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.NativeProgram
import bench.repairs.tritonbench_g.context_attn_fwd.Contract

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Checked
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
set_option maxRecDepth 24000
set_option synthInstance.maxSize 512

deriving instance DecidableEq for Tensor3

abbrev View := VeriTile.Bench.Repairs.ContextAttnFwd.Checked.View
structure Views where
  (q k v o start length : View)

def all (v : Views) : List View := [v.q,v.k,v.v,v.o,v.start,v.length]
def reads (v : Views) : List View := [v.q,v.k,v.v,v.start,v.length]
abbrev separate := VeriTile.Bench.Repairs.ContextAttnFwd.Checked.separate

def canonical (t : Tensor3) : Prop := t.rowStride=t.heads*t.dims ∧ t.headStride=t.dims ∧ t.dimStride=1

def tensorChecks (c : Call) (v : Views) : Prop :=
  (∀ x∈[v.q,v.k,v.v,v.o],x.metadata.shape=[c.q.rows,c.q.heads,c.q.dims] ∧
    x.metadata.dtype=.floating .fp32 ∧ x.metadata.isContiguous=Bool.true ∧
    x.strides=[c.q.heads*c.q.dims,c.q.dims,1]) ∧
  (∀ x∈[v.start,v.length],x.metadata.shape=[c.batches] ∧
    x.metadata.dtype=.other "int32" ∧ x.metadata.isContiguous=Bool.true) ∧
  v.q.metadata.device.kind="cuda" ∧
  (∀ x∈all v,x.metadata.layout=.strided ∧ x.metadata.device=v.q.metadata.device ∧
    x.metadata.isNeg=Bool.false ∧ x.conjugate=Bool.false ∧ x.storageBytes%4=0 ∧ x.storageBytes<2^63 ∧
    (x.metadata.numel=0 ∨ x.metadata.dataPtr.toNat%4=0)) ∧
  (c.batches=0 ∨ ∀ x∈reads v,separate v.o x)

/-- Finite descriptor, capacity and nonoverlap checks. Native active access
bounds and injective output ownership are derived below. -/
def geometryChecks (c : Call) (r : Regions) (extent : RegionBounds) : Prop :=
  c.q.dims∈({16,32,64,128} : Finset Nat) ∧ 0 < c.q.heads ∧
  c.k=c.q ∧ c.v=c.q ∧ c.o=c.q ∧ canonical c.q ∧
  (∀ b : Fin c.batches,c.start b+c.length b≤c.q.rows ∧ c.length b≤c.maxInputLen) ∧
  (∀ b z : Fin c.batches,0 < c.length b → 0 < c.length z → b≠z →
    c.start b+c.length b≤c.start z ∨ c.start z+c.length z≤c.start b) ∧
  (∀ rg∈[r.query,r.key,r.value,r.out],c.q.rows*c.q.heads*c.q.dims≤extent rg) ∧
  c.batches≤extent r.start ∧ c.batches≤extent r.sequence

def formedBudget (c : Call) (requestedMax : Nat) (v : View) : Prop :=
  v.storageOffset+(c.q.rows+requestedMax+128-1)*(c.q.heads*c.q.dims)+
    (c.q.heads-1)*c.q.dims+(c.q.dims-1)<2^61 ∧
  c.q.heads*c.q.dims<2^61 ∧ c.q.dims<2^61

def vectorBudget (v : View) (count : Nat) : Prop :=
  v.storageOffset+(count-1)*v.strides[0]?.getD 0<2^61 ∧ v.strides[0]?.getD 0<2^61

def nativeChecks (c : Call) (v : Views) (requestedMax : Nat) : Prop :=
  requestedMax<2^31 ∧ c.maxInputLen≤requestedMax ∧ c.batches<2^31 ∧
  c.q.heads≤65535 ∧ c.blocks≤65535 ∧
  (∀ x∈[v.q,v.k,v.v,v.o],formedBudget c requestedMax x) ∧
  vectorBudget v.start c.batches ∧ vectorBudget v.length c.batches

def metadataChecks (c : Call) (start length : Nat → Int) : Prop :=
  (∀ b : Fin c.batches,start b=(c.start b : Int) ∧ length b=(c.length b : Int)) ∧
  c.maxInputLen=(Finset.univ.sup fun b : Fin c.batches => c.length b)

instance (t : Tensor3) : Decidable (canonical t) := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Call) (v : Views) : Decidable (tensorChecks c v) := by unfold tensorChecks all reads; infer_instance
instance (c : Call) (r : Regions) (extent : RegionBounds) : Decidable (geometryChecks c r extent) := by
  unfold geometryChecks; infer_instance
instance (c : Call) (requestedMax : Nat) (v : View) : Decidable (formedBudget c requestedMax v) := inferInstanceAs (Decidable (_ ∧ _))
instance (v : View) (count : Nat) : Decidable (vectorBudget v count) := inferInstanceAs (Decidable (_ ∧ _))
instance (c : Call) (v : Views) (requestedMax : Nat) : Decidable (nativeChecks c v requestedMax) := by
  unfold nativeChecks; infer_instance
instance (c : Call) (start length : Nat → Int) : Decidable (metadataChecks c start length) := by
  unfold metadataChecks; infer_instance

def InputContract (c : Call) (r : Regions) (v : Views) (extent : RegionBounds)
    (requestedMax : Nat) (start length : Nat → Int) : Prop :=
  tensorChecks c v ∧ geometryChecks c r extent ∧ nativeChecks c v requestedMax ∧ metadataChecks c start length

def checkInputs (c : Call) (r : Regions) (v : Views) (extent : RegionBounds)
    (requestedMax : Nat) (start length : Nat → Int) : Bool :=
  decide (tensorChecks c v) && decide (geometryChecks c r extent) &&
    decide (nativeChecks c v requestedMax) && decide (metadataChecks c start length)

theorem checkInputs_iff (c : Call) (r : Regions) (v : Views) (extent : RegionBounds)
    (requestedMax : Nat) (start length : Nat → Int) :
    checkInputs c r v extent requestedMax start length=Bool.true ↔
      InputContract c r v extent requestedMax start length := by
  simp only [checkInputs,InputContract,Bool.and_eq_true,decide_eq_true_eq,and_assoc]

theorem offset_eq (t : Tensor3) (h : canonical t) (row head dim : Nat) :
    t.offset row head dim=(row*t.heads+head)*t.dims+dim := by
  simp only [Tensor3.offset,h.1,h.2.1,h.2.2,Nat.mul_one]
  ring

theorem bounded (t : Tensor3) (h : canonical t) (extent : Nat)
    (space : t.rows*t.heads*t.dims≤extent) : t.Bounded extent := by
  intro row head dim hr hh hd
  rw [offset_eq t h]
  have hrow := Nat.mul_le_mul_right t.heads (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at hrow
  have hlinear : row*t.heads+head < t.rows*t.heads := by omega
  have hdim := Nat.mul_le_mul_right t.dims (Nat.succ_le_of_lt hlinear)
  simp only [Nat.succ_mul] at hdim
  omega

theorem injective (t : Tensor3) (h : canonical t) :
    ∀ row row' head head' dim dim',row < t.rows → row' < t.rows → head < t.heads → head' < t.heads →
    dim < t.dims → dim' < t.dims → t.offset row head dim=t.offset row' head' dim' →
    row=row' ∧ head=head' ∧ dim=dim' := by
  intro row row' head head' dim dim' _ _ hh hh' hd hd' he
  rw [offset_eq t h,offset_eq t h] at he
  have hD : 0 < t.dims := by omega
  have hH : 0 < t.heads := by omega
  have e : row*t.heads+head=row'*t.heads+head' := by
    have divided := congrArg (· / t.dims) he
    simpa [Nat.add_div,hD,Nat.ne_of_gt hD,Nat.div_eq_of_lt hd,Nat.div_eq_of_lt hd',
      Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd',Nat.not_le.mpr hd,Nat.not_le.mpr hd'] using divided
  have d : dim=dim' := by
    have reduced := congrArg (· % t.dims) he
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hd,Nat.mod_eq_of_lt hd'] using reduced
  have hhEq : head=head' := by
    have reduced := congrArg (· % t.heads) e
    simpa [Nat.add_mod,Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh'] using reduced
  have rr : row=row' := by
    have divided := congrArg (· / t.heads) e
    simpa [Nat.add_div,hH,Nat.ne_of_gt hH,Nat.div_eq_of_lt hh,Nat.div_eq_of_lt hh',
      Nat.mod_eq_of_lt hh,Nat.mod_eq_of_lt hh',Nat.not_le.mpr hh,Nat.not_le.mpr hh'] using divided
  exact ⟨rr,hhEq,d⟩

theorem intervals (c : Call)
    (packed : ∀ b z : Fin c.batches,0 < c.length b → 0 < c.length z → b≠z →
      c.start b+c.length b≤c.start z ∨ c.start z+c.length z≤c.start b) :
    ∀ b b' t t',b < c.batches → b' < c.batches → t < c.length b → t' < c.length b' →
      c.start b+t=c.start b'+t' → b=b' ∧ t=t' := by
  intro b b' t t' hb hb' ht ht' he
  by_cases eq : b=b'
  · exact ⟨eq,by subst b'; omega⟩
  · have sep := packed ⟨b,hb⟩ ⟨b',hb'⟩ (show 0 < c.length b from Nat.zero_lt_of_lt ht)
      (show 0 < c.length b' from Nat.zero_lt_of_lt ht') (fun h => eq (congrArg Fin.val h))
    change c.start b+c.length b≤c.start b' ∨ c.start b'+c.length b'≤c.start b at sep
    rcases sep with h|h <;> omega

#axiomsClean checkInputs_iff
#axiomsClean bounded
#axiomsClean injective
end VeriTile.Bench.Repairs.ContextAttnNopad.Checked
