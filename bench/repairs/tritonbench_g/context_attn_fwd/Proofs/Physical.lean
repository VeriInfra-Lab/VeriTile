import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Real
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeAddresses
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeSM90
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Numerical

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Physical
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section
set_option maxHeartbeats 4000000

abbrev Byte := BitVec 8
abbrev Memory := Nat → Byte
def byte (word : Word) (lane : Fin 4) : Byte := BitVec.extractLsb' (8*lane.val) 8 word
def Lane (c : Host.Call) := GridIndex c.grid × TileIndex [c.layout.BM,c.layout.D] × Fin 4
def active (c : Host.Call) (i : Lane c) := c.active i.1 i.2.1
def address (c : Host.Call) (B : Native.Bytes extent) (i : Lane c) : Nat :=
  B.base c.regions.out+4*c.offset i.1 i.2.1+i.2.2.val
def owner (c : Host.Call) (i : Lane c) :=
  NativeSM90.storedOwner c.layout.BM c.layout.D i.2.1.1.val i.2.1.2.1.val
def outputWord (c : Host.Call) (ops : ScalarOps) (t : State Expr) (i : Lane c) : Word :=
  evaluate ops (NativeSM90.plan c.layout.BM c.layout.D)
    ((t.mem c.regions.out (c.offset i.1 i.2.1)).read .real) (owner c i)
def outputByte (c : Host.Call) (ops : ScalarOps) (t : State Expr) (i : Lane c) : Byte :=
  byte (outputWord c ops t i) i.2.2

/-- The finite masked four-byte stores from the successful source graph.
Selection depends only on output addresses, never on floating payloads. -/
def stores (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (t : State Expr) (before : Memory) (a : Nat) : Byte := by
  classical
  exact if h : ∃ i : Lane c,active c i ∧ address c B i=a then outputByte c ops t (Classical.choose h)
  else before a

theorem address_unique (c : Host.Call) (B : Native.Bytes extent) (A : FlatAlloc) (s : BlockState)
    (hc : Geometry.Checked c A.extent) (hm : Geometry.Metadata c s) (sep : FPSuccess.Separate c)
    (i j : Lane c) (hi : active c i) (hj : active c j) (he : address c B i=address c B j) : i=j := by
  have hb := i.2.2.isLt
  have hb' := j.2.2.isLt
  have hoff : c.offset i.1 i.2.1=c.offset j.1 j.2.1 := by unfold address at he; omega
  have hw := RealProof.unique_writer c A s hc hm sep i.1 j.1 i.2.1 j.2.1 hi hj hoff
  have hbyte : i.2.2=j.2.2 := Fin.ext (by unfold address at he; omega)
  exact Prod.ext hw.1 (Prod.ext hw.2 hbyte)

theorem readback (c : Host.Call) (B : Native.Bytes extent) (A : FlatAlloc) (s : BlockState)
    (hc : Geometry.Checked c A.extent) (hm : Geometry.Metadata c s) (sep : FPSuccess.Separate c)
    (ops : ScalarOps) (t : State Expr) (before : Memory) (i : Lane c) (hi : active c i) :
    stores c B ops t before (address c B i)=byte (outputWord c ops t i) i.2.2 := by
  have hex : ∃ j : Lane c,active c j ∧ address c B j=address c B i := ⟨i,hi,rfl⟩
  rw [stores,dif_pos hex]
  have hj := Classical.choose_spec hex
  rw [address_unique c B A s hc hm sep (Classical.choose hex) i hj.1 hi hj.2]
  rfl

theorem frame (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (t : State Expr) (before : Memory) (a : Nat)
    (h : ¬∃ i : Lane c,active c i ∧ address c B i=a) : stores c B ops t before a=before a := by
  rw [stores,dif_neg h]

def reads (c : Host.Call) := [c.regions.query,c.regions.key,c.regions.value,c.regions.start,c.regions.sequence,c.regions.prompt]
/-- Ordinary owned allocations: Out is separate from every read allocation.
No disjointness is required between any pair of read-only regions. -/
def Separate (c : Host.Call) (B : Native.Bytes extent) : Prop :=
  ∀ rg,rg∈reads c → B.base c.regions.out+4*extent c.regions.out≤B.base rg ∨
    B.base rg+4*extent rg≤B.base c.regions.out

theorem input_frame (c : Host.Call) (B : Native.Bytes extent) (hc : Geometry.Checked c extent)
    (sep : Separate c B) (ops : ScalarOps) (t : State Expr) (before : Memory)
    (rg : RegionName) (hr : rg∈reads c) (off : Nat) (ho : off<extent rg) (j : Fin 4) :
    stores c B ops t before (B.base rg+4*off+j.val)=before (B.base rg+4*off+j.val) := by
  apply frame
  rintro ⟨i,hi,he⟩
  have hb := Host.batch_bound c i.1
  have hh := Host.head_bound c i.1
  have hout := Geometry.output_bound c hc _ _ _ _ hb hh hi i.2.1.2.1.isLt
  change c.offset i.1 i.2.1<extent c.regions.out at hout
  have il := i.2.2.isLt
  have jl := j.isLt
  unfold address at he
  rcases sep rg hr with h|h <;> omega

theorem byte_address_finite (c : Host.Call) (B : Native.Bytes extent) (hc : Geometry.Checked c extent)
    (i : Lane c) (hi : active c i) : address c B i<2^64 := by
  have ho := Geometry.output_bound c hc _ _ _ _ (Host.batch_bound c i.1)
    (Host.head_bound c i.1) hi i.2.1.2.1.isLt
  change c.offset i.1 i.2.1<extent c.regions.out at ho
  have hb := B.fits c.regions.out
  have hi := i.2.2.isLt
  unfold address
  omega

theorem scalar_equiv (c : Host.Call) (B : Native.Bytes extent) (reference native : ScalarOps)
    (laws : Numerical.ScalarLaws reference native) (t : State Expr) (before : Memory) :
    stores c B reference t before=stores c B native t before := by
  have he := laws.operations_eq reference native
  rw [he]

/-- No active requests cause no write, including padding, unused rows, and
allocation bytes outside the view. -/
theorem empty_stores (c : Host.Call) (B : Native.Bytes extent) (ops : ScalarOps)
    (t : State Expr) (before : Memory) (empty : ∀ b,b<c.batches → c.length b=0) :
    stores c B ops t before=before := by
  funext a
  apply frame
  rintro ⟨i,hi,_⟩
  have he := empty _ (Host.batch_bound c i.1)
  change _<c.length _ at hi
  rw [he] at hi
  omega

#print axioms readback
#print axioms input_frame
end
end VeriTile.Bench.Repairs.ContextAttnFwd.Physical
