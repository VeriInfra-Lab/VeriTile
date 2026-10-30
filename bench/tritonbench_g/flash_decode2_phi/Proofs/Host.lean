import bench.tritonbench_g.flash_decode2_phi.Proofs.Geometry
import bench.tritonbench_g.flash_decode2_phi.Proofs.Reference

namespace VeriTile.Bench.Repairs.FlashDecode2.Host
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000

structure Request where
  g : Geometry
  valid : g.Valid

structure Tensor where
  allocation : RegionName
  shape : List Nat
  dtype : String
  bytes : Nat
  device : Nat
  cuda : Bool
  layout : String
  contiguous : Bool
  offset : Nat
  lazyNeg : Bool
  lazyConj : Bool

def Tensor.Valid (t : Tensor) (dims : List Nat) (dtype : String) (owner : Nat) : Prop :=
  t.shape=dims ∧ t.dtype=dtype ∧ t.bytes=4 ∧ t.device=owner ∧
  t.cuda=Bool.true ∧ t.layout="strided" ∧ t.contiguous=Bool.true ∧ t.offset=0 ∧
  t.lazyNeg=Bool.false ∧ t.lazyConj=Bool.false

/-- Successful ordinary same-call validation and binding of the four supplied
buffers. Length observations are pre-launch input facts checked by min/max;
no output, kernel execution, reduction, or arithmetic result is a field. -/
structure Preparation (α : Type) [Inhabited α] (q : Request) where
  before : State α
  mid : Tensor
  log : Tensor
  lengths : Tensor
  out : Tensor
  owner : Nat
  callerDevice : Nat
  midValid : mid.Valid [q.g.batches,q.g.heads,q.g.capacity,q.g.dimension] "float32" owner
  logValid : log.Valid [q.g.batches,q.g.heads,q.g.capacity] "float32" owner
  lengthsValid : lengths.Valid [q.g.batches] "int32" owner
  outValid : out.Valid [q.g.batches,q.g.heads,q.g.dimension] "float32" owner
  disjoint : 0<q.g.rows → out.allocation≠mid.allocation ∧ out.allocation≠log.allocation ∧
    out.allocation≠lengths.allocation ∧ lengths.allocation≠mid.allocation ∧ lengths.allocation≠log.allocation
  lengthValues : ∀ b,b<q.g.batches → 0<q.g.heads →
    let l := (before.mem "L" b).read .nat
    0<l ∧ l≤q.g.capacity*q.g.sequenceBlock ∧ l≤2^31-1

structure Result (α : Type) where
  returned : Option Unit
  suppliedOutput : Tensor
  state : State α
  currentDevice : Nat
  launchDevice : Option Nat

def finish {α : Type} [Inhabited α] {q : Request} (call : Preparation α q) (s : State α) : Result α :=
  ⟨none,call.out,s,call.callerDevice,if q.g.rows=0 then none else some call.owner⟩

def invoke {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation α q) (schedule : List Nat) : Option (Result α) :=
  (Grid.source A q.g.masked q.g.batches q.g.heads q.g.capacity q.g.dimension q.g.sequenceBlock
    q.g.laneBlock schedule call.before).map (finish call)

theorem row_indices (q : Request) (row : Nat) (hr : row<q.g.rows) :
    0<q.g.heads ∧ row/q.g.heads<q.g.batches ∧ row%q.g.heads<q.g.heads := by
  have hH : 0<q.g.heads := by dsimp [Geometry.rows] at hr; nlinarith
  exact ⟨hH,(Nat.div_lt_iff_lt_mul hH).mpr hr,Nat.mod_lt _ hH⟩

theorem row_count {α : Type} [Inhabited α] (q : Request) (call : Preparation α q)
    (row : Nat) (hr : row<q.g.rows) :
    0<count (Grid.setup q.g.batches q.g.heads row call.before) q.g.sequenceBlock ∧
    count (Grid.setup q.g.batches q.g.heads row call.before) q.g.sequenceBlock≤q.g.capacity := by
  obtain ⟨hH,hb,hh⟩ := row_indices q row hr
  obtain ⟨hl,hcap,hint⟩ := call.lengthValues _ hb hH
  have he := count_eq (Grid.setup q.g.batches q.g.heads row call.before) q.g.sequenceBlock
    ((call.before.mem "L" (row/q.g.heads)).read .nat) rfl hl
  rw [he]
  exact ⟨ceil_positive _ _ hl q.valid.2.2.2.1,ceil_capacity _ _ _ q.valid.2.2.2.1 hcap⟩

theorem whole_typed {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation α q) (schedule : List Nat) (all : schedule.Perm (List.range q.g.rows)) :
    ∃ result,invoke A q call schedule=some result ∧ result.returned=none ∧
      result.suppliedOutput=call.out ∧ result.currentDevice=call.callerDevice ∧
      result.launchDevice=(if q.g.rows=0 then none else some call.owner) ∧
      (∀ row (k : TileIndex [q.g.laneBlock]),row<q.g.rows → lane k<q.g.dimension →
        result.state.mem "O" (row*q.g.dimension+lane k)=.mk .real
          (Grid.value A call.before q.g.masked q.g.batches q.g.heads q.g.capacity
            q.g.dimension q.g.sequenceBlock q.g.laneBlock row k)) ∧
      (∀ r a,r≠("O" : RegionName) ∨ q.g.size≤a → result.state.mem r a=call.before.mem r a) := by
  obtain ⟨s,he,hm⟩ := Grid.source_complete A q.g.masked q.g.batches q.g.heads q.g.capacity
    q.g.dimension q.g.sequenceBlock q.g.laneBlock schedule call.before
  refine ⟨finish call s,by simp [invoke,he],rfl,rfl,rfl,rfl,?_,?_⟩
  · intro row k hr hk
    change s.mem "O" _=_
    rw [hm]
    exact Grid.grid_readback A q.g.masked q.g.batches q.g.heads q.g.capacity
      q.g.dimension q.g.sequenceBlock q.g.laneBlock q.valid.2.2.1 schedule call.before row
      (all.mem_iff.mpr (List.mem_range.mpr hr)) k hk
  · intro r a outside
    change s.mem r a=_
    rw [hm]
    apply Grid.grid_frame A q.g.masked q.g.batches q.g.heads q.g.capacity
      q.g.dimension q.g.sequenceBlock q.g.laneBlock q.valid.2.2.1 schedule call.before r a
    rcases outside with h|h
    · exact .inl h
    · right
      intro row hr k hk eq
      have bound := output_offset_bound q.g row k
        (List.mem_range.mp (all.mem_iff.mp hr)) hk
      omega

theorem empty_return {α : Type} [Inhabited α] (A : Algebra α) (q : Request)
    (call : Preparation α q) (schedule : List Nat) (all : schedule.Perm (List.range q.g.rows))
    (empty : q.g.batches=0 ∨ q.g.heads=0) :
    invoke A q call schedule=some ⟨none,call.out,call.before,call.callerDevice,none⟩ := by
  have hz : q.g.rows=0 := by rcases empty with h|h <;> simp [Geometry.rows,h]
  have hs : schedule=[] := by
    apply List.eq_nil_of_length_eq_zero
    simpa [hz] using all.length_eq
  simp [invoke,hs,Grid.source,finish,hz]

#axiomsClean whole_typed
#axiomsClean empty_return
#axiomsClean row_count
#specNonCircular Preparation avoiding [invoke,Grid.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FlashDecode2.Host
