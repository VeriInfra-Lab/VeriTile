import bench.tritonbench_g.attention_llama.Proofs.RealCorrect
import Mathlib.Data.Nat.Bitwise

namespace VeriTile.Bench.Repairs.AttentionLlama.Checked
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.AttentionLlama
open VeriTile.Bench.TritonBenchG.AttentionLlama.RealKernel
open LoopInvariant HostGrid HostResult Correctness
set_option maxRecDepth 6000
set_option maxHeartbeats 1000000

structure Strides where
  (batch head row dim : Nat)
  deriving DecidableEq

def queryStrides (c : Context) : Strides := ⟨c.sqz,c.sqh,c.sqm,c.sqk⟩
def keyStrides (c : Context) : Strides := ⟨c.skz,c.skh,c.skn,c.skk⟩
def valueStrides (c : Context) : Strides := ⟨c.svz,c.svh,c.svk,c.svn⟩
def outputStrides {B : Nat} {c : Context} (o : OutputLayout B c) : Strides := ⟨o.soz,o.soh,o.som,o.son⟩
def Strides.at (s : Strides) (b h r d : Nat) := b*s.batch+h*s.head+r*s.row+d*s.dim

def Strides.span (s : Strides) (B NH rows D : Nat) :=
  if B=0 ∨ NH=0 ∨ rows=0 ∨ D=0 then 0
  else (B-1)*s.batch+(NH-1)*s.head+(rows-1)*s.row+(D-1)*s.dim+1

theorem active_bound (s : Strides) (B NH rows D b h r d : Nat)
    (hb : b<B) (hh : h<NH) (hr : r<rows) (hd : d<D) :
    s.at b h r d<s.span B NH rows D := by
  have bnd:=Nat.mul_le_mul_right s.batch (show b≤B-1 by omega)
  have hnd:=Nat.mul_le_mul_right s.head (show h≤NH-1 by omega)
  have rnd:=Nat.mul_le_mul_right s.row (show r≤rows-1 by omega)
  have dnd:=Nat.mul_le_mul_right s.dim (show d≤D-1 by omega)
  have nz : ¬(B=0 ∨ NH=0 ∨ rows=0 ∨ D=0) := by omega
  simp only [Strides.at,Strides.span,if_neg nz]
  omega

/-- Ordinary backed Tensor spans. No source-index access law is a field. -/
structure Storage (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (bounds : RegionBounds) : Prop where
  query : (queryStrides c).span B c.NH c.H c.D≤bounds c.Q
  key : (keyStrides c).span B c.NH c.N c.D≤bounds c.K
  value : (valueStrides c).span B c.NH c.N c.D≤bounds c.V
  output : (outputStrides o).span B c.NH c.H c.D≤bounds Out

def metadataChecks (c : Context) : Bool := decide (
  (c.D=16 ∨ c.D=32 ∨ c.D=64 ∨ c.D=128) ∧ 0<c.N ∧ c.SP<2^31 ∧
    (c.causal=Bool.true → c.SP=0))

theorem admitted_metadata (c : Context) (h : metadataChecks c=Bool.true) :
    0<c.D ∧ 0<c.N ∧ (c.causal=Bool.true → c.SP=0) := by
  simp only [metadataChecks,decide_eq_true_eq] at h
  exact ⟨by rcases h.1 with h|h|h|h <;> omega,h.2.1,h.2.2.2⟩

def keyRows (c : Context) : Nat := if c.causal then ((c.H+63)/64)*64+128 else c.N+128

def strideExtent (B : Nat) (c : Context) (s : Strides) (rows : Nat) : Nat :=
  B*s.batch+c.NH*s.head+rows*s.row+c.D*s.dim+1

def envelope (B : Nat) (c : Context) (o : OutputLayout B c) : Nat :=
  B+c.NH+c.H+c.N+c.D+c.SP+256+B*c.NH+
  strideExtent B c (queryStrides c) (c.H+128)+
  strideExtent B c (keyStrides c) (keyRows c)+
  strideExtent B c (valueStrides c) (keyRows c)+
  strideExtent B c (outputStrides o) (c.H+128)

inductive Buffer where | query | key | value | output deriving DecidableEq

def buffers : List Buffer := [.query,.key,.value,.output]
def width (fp8 : Bool) : Buffer → Nat | .key | .value=>if fp8 then 1 else 2 | _=>2

structure Pointers (B : Nat) (c : Context) (o : OutputLayout B c) where
  base : Buffer → Nat
  outputBinding : o.base=base .output/2

def launchChecks (B : Nat) (c : Context) (o : OutputLayout B c) : Bool :=
  decide (envelope B c o<2^31 ∧ B*c.NH≤65535)
def pointerChecks (B : Nat) (c : Context) (o : OutputLayout B c) (fp8 : Bool) (p : Pointers B c o) : Bool :=
  buffers.all fun r=>decide (p.base r%width fp8 r=0 ∧ p.base r+width fp8 r*envelope B c o<2^64)

theorem formed_signed32 (B : Nat) (c : Context) (o : OutputLayout B c)
    (h : launchChecks B c o=Bool.true) (n : Nat) (hn : n<envelope B c o) :
    (BitVec.ofInt 32 (n : Int)).toInt=(n : Int) := by
  simp only [launchChecks,decide_eq_true_eq] at h
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · norm_num at h ⊢;omega

theorem pointer_formed (B : Nat) (c : Context) (o : OutputLayout B c) (fp8 : Bool) (p : Pointers B c o)
    (h : pointerChecks B c o fp8 p=Bool.true) (r : Buffer) (n : Nat) (hn : n<envelope B c o) :
    p.base r%width fp8 r=0 ∧ p.base r+width fp8 r*n<2^64 := by
  simp only [pointerChecks,List.all_eq_true,decide_eq_true_eq] at h
  have member : r∈buffers := by cases r <;> simp [buffers]
  have bound:=h r member
  have le:=Nat.mul_le_mul_left (width fp8 r) (Nat.le_of_lt hn)
  exact ⟨bound.1,by omega⟩

/-- Exact finite bit check, with the signed-byte range supplied by ordinary
int8 observation. Numerical E5M2 decoding is separate from address safety. -/
def byteCheck (n : Int) : Bool := decide ((n%256).toNat &&& 124 ≠ 124)

private theorem mask_exponent : ∀ b : Fin 256,
    b.val &&& 124 ≠ 124 ↔ b.val/4%32 ≠ 31 := by decide

theorem finite_byte (n : Int) (range : -128≤n ∧ n<128) (h : byteCheck n=Bool.true) :
    FP8RealMemory.FiniteByte n := by
  refine ⟨by simpa [Float8E5Bits.isInt8] using range,?_⟩
  simp only [Float8E5Bits.decodeRat_isSome,decide_eq_true_eq,Float8E5Bits.exponent]
  apply (mask_exponent (Float8E5Bits.ofInt n)).mp
  simpa only [byteCheck,decide_eq_true_eq,Float8E5Bits.ofInt] using h

#axiomsClean active_bound
#axiomsClean admitted_metadata
#axiomsClean formed_signed32
#axiomsClean pointer_formed
#axiomsClean finite_byte

/-- Values read by the Python metadata branch, before any launch. -/
inductive DType where | half | int8 | other deriving DecidableEq
structure TensorMetadata where
  (ordinary strided cuda negative conjugate : Bool)
  device : Nat
  dtype : DType
  shape : List Nat
  strides : List Int
  deriving DecidableEq

def Strides.raw (s : Strides) : List Int := [s.batch,s.head,s.row,s.dim]
def tensorChecks (t : TensorMetadata) (device : Nat) (dtype : DType)
    (shape : List Nat) (s : Strides) : Bool := decide (
  t.ordinary=Bool.true ∧ t.strided=Bool.true ∧ t.cuda=Bool.true ∧ t.negative=Bool.false ∧ t.conjugate=Bool.false ∧
  t.device=device ∧ t.dtype=dtype ∧ t.shape=shape ∧ t.strides=s.raw)

/-- Scalar flags are observations of the ordinary Python type/finite tests.
The source scale is passed unchanged, including zero and negative values. -/
structure ScalarMetadata where
  (ordinaryScale finiteScale causalBool positionInt : Bool)
  position : Int

def inputChecks (B : Nat) (c : Context) (fp8 : Bool)
    (q k v : TensorMetadata) (a : ScalarMetadata) : Bool :=
  metadataChecks c &&
  tensorChecks q q.device .half [B,c.NH,c.H,c.D] (queryStrides c) &&
  tensorChecks k q.device (if fp8 then .int8 else .half) [B,c.NH,c.N,c.D] (keyStrides c) &&
  tensorChecks v q.device (if fp8 then .int8 else .half) [B,c.NH,c.N,c.D] (valueStrides c) &&
  decide (a.ordinaryScale=Bool.true ∧ a.finiteScale=Bool.true ∧ a.causalBool=Bool.true ∧
    a.positionInt=Bool.true ∧ a.position=(c.SP : Int))

theorem input_metadata (B : Nat) (c : Context) (fp8 : Bool)
    (q k v : TensorMetadata) (a : ScalarMetadata) (h : inputChecks B c fp8 q k v a=Bool.true) :
    metadataChecks c=Bool.true := by
  simp only [inputChecks,Bool.and_eq_true] at h
  tauto

/-- Independent per-axis and padded-lane bounds for the arithmetic formed by
this source. The shared launcher product is an ordinary runtime boundary. -/
theorem stride_bound (B : Nat) (c : Context) (s : Strides)
    (rows b h r d : Nat) (hb : b≤B) (hh : h≤c.NH) (hr : r≤rows) (hd : d≤c.D) :
    s.at b h r d < strideExtent B c s rows := by
  have h₁:=Nat.mul_le_mul_right s.batch hb
  have h₂:=Nat.mul_le_mul_right s.head hh
  have h₃:=Nat.mul_le_mul_right s.row hr
  have h₄:=Nat.mul_le_mul_right s.dim hd
  simp only [Strides.at,strideExtent]
  omega

def bufferStrides {B : Nat} {c : Context} (o : OutputLayout B c) : Buffer → Strides
  | .query=>queryStrides c | .key=>keyStrides c | .value=>valueStrides c | .output=>outputStrides o

def bufferRows (c : Context) : Buffer → Nat
  | .query | .output=>c.H+128 | .key | .value=>keyRows c

theorem source_offset_bound (B : Nat) (c : Context) (o : OutputLayout B c)
    (which : Buffer) (b h r d : Nat) (hb : b≤B) (hh : h≤c.NH)
    (hr : r≤bufferRows c which) (hd : d≤c.D) :
    (bufferStrides o which).at b h r d<envelope B c o := by
  have hbound:=stride_bound B c (bufferStrides o which) (bufferRows c which) b h r d hb hh hr hd
  cases which <;> simp only [bufferRows,bufferStrides,envelope] at * <;> omega

theorem padded_query (B : Nat) (c : Context) (i : Index B c.NH c.H c.D) (lane : Fin 64) :
    i.query.val/64*64+lane.val≤c.H+128 := by
  have hq:=i.query.isLt
  have hl:=lane.isLt
  omega

theorem loop_stop_bound (B : Nat) (c : Context) (i : Index B c.NH c.H c.D)
    (hSP : c.causal=Bool.true → c.SP=0) :
    FP8RealSetup.stop (contextFor c i).initial c.N c.SP 64 c.causal+128≤keyRows c := by
  unfold FP8RealSetup.stop keyRows
  cases h : c.causal <;> simp only [h,Bool.false_eq_true,↓reduceIte,contextFor]
  · omega
  · rw [hSP h]
    have hq:=i.query.isLt
    have hround : i.query.val/64*64≤((c.H+63)/64)*64 := by omega
    omega

theorem padded_key (B : Nat) (c : Context) (i : Index B c.NH c.H c.D)
    (start : Nat) (hstart : start<FP8RealSetup.stop (contextFor c i).initial c.N c.SP 64 c.causal)
    (lane : Fin 64) (hSP : c.causal=Bool.true → c.SP=0) : start+lane.val≤keyRows c ∧ start+64+lane.val≤keyRows c := by
  have hs:=loop_stop_bound B c i hSP
  have hl:=lane.isLt
  omega

/-- Decoding a linear program ID is proved from the real launch coordinates. -/
theorem base_context (B : Nat) (c : Context) (i : Index B c.NH c.H c.D) (sz sh : Nat) :
    alBase (contextFor c i).initial c.NH sz sh = i.batch.val*sz+i.head.val*sh := by
  have h:=MaskCoverage.batch_head_roundtrip c.NH i.batch.val i.head.val i.head.isLt
  simp only [alBase,contextFor,Nat.one_ne_zero,↓reduceIte,h.1,h.2]

theorem query_access (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds)
    (i : Index B c.NH c.H c.D) (idx : TileIndex [64,c.D])
    (hi : alActive (contextFor c i).initial c.H 64 idx) :
    alBase (contextFor c i).initial c.NH c.sqz c.sqh+
      ((contextFor c i).initial.pids 0*64+idx.1.val)*c.sqm+idx.2.1.val*c.sqk<bounds c.Q := by
  rw [base_context]
  exact lt_of_lt_of_le (active_bound (queryStrides c) B c.NH c.H c.D
    i.batch.val i.head.val _ idx.2.1.val i.batch.isLt i.head.isLt hi idx.2.1.isLt) storage.query

theorem key_access (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds)
    (i : Index B c.NH c.H c.D) (r : Nat) (hr : r<c.N) (d : Fin c.D) :
    alBase (contextFor c i).initial c.NH c.skz c.skh+r*c.skn+d.val*c.skk<bounds c.K := by
  rw [base_context]
  exact lt_of_lt_of_le (active_bound (keyStrides c) B c.NH c.N c.D
    i.batch.val i.head.val r d.val i.batch.isLt i.head.isLt hr d.isLt) storage.key

theorem value_access (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds)
    (i : Index B c.NH c.H c.D) (r : Nat) (hr : r<c.N) (d : Fin c.D) :
    alBase (contextFor c i).initial c.NH c.svz c.svh+r*c.svk+d.val*c.svn<bounds c.V := by
  rw [base_context]
  exact lt_of_lt_of_le (active_bound (valueStrides c) B c.NH c.N c.D
    i.batch.val i.head.val r d.val i.batch.isLt i.head.isLt hr d.isLt) storage.value

theorem output_access (B : Nat) (c : Context) (Out : RegionName) (o : OutputLayout B c)
    (bounds : RegionBounds) (storage : Storage B c Out o bounds)
    (i : Index B c.NH c.H c.D) (idx : TileIndex [64,c.D])
    (hi : alActive (contextFor c i).initial c.H 64 idx) :
    alOutOffset (contextFor c i).initial c.NH o.soz o.soh o.som o.son 64 idx<bounds Out := by
  unfold alOutOffset
  rw [base_context]
  exact lt_of_lt_of_le (active_bound (outputStrides o) B c.NH c.H c.D
    i.batch.val i.head.val _ idx.2.1.val i.batch.isLt i.head.isLt hi idx.2.1.isLt) storage.output

/-- Actual int8 observations, with only ordinary dtype range and coordinate
memory bindings. There is deliberately no finite-decoding field. -/
structure Bytes (B : Nat) (c : Context) where
  (key value : Fin B → Fin c.NH → Fin c.N → Fin c.D → Int)
structure ByteBinding (B : Nat) (c : Context) (x : Bytes B c) : Prop where
  keyRange : ∀ b h n d, -128≤x.key b h n d ∧ x.key b h n d<128
  valueRange : ∀ b h n d, -128≤x.value b h n d ∧ x.value b h n d<128
  key : ∀ b h n d, c.initial.readMemValue .int c.K
    ((keyStrides c).at b.val h.val n.val d.val)=x.key b h n d
  value : ∀ b h n d, c.initial.readMemValue .int c.V
    ((valueStrides c).at b.val h.val n.val d.val)=x.value b h n d

def byteChecks (B : Nat) (c : Context) (x : Bytes B c) : Bool := decide (
  (∀ b h n d, byteCheck (x.key b h n d)=Bool.true) ∧
  (∀ b h n d, byteCheck (x.value b h n d)=Bool.true))

theorem checked_bytes (B : Nat) (c : Context) (x : Bytes B c)
    (binding : ByteBinding B c x) (h : byteChecks B c x=Bool.true) :
    ByteHostResult.Domain (B:=B) (heads:=c.NH) c := by
  simp only [byteChecks,decide_eq_true_eq] at h
  constructor
  · intro i k _ j d hj
    rw [base_context]
    change FP8RealMemory.FiniteByte (c.initial.readMemValue .int c.K
      ((keyStrides c).at i.batch.val i.head.val (0+k*64+j.val) d.val))
    rw [binding.key i.batch i.head ⟨_,hj⟩ d]
    exact finite_byte _ (binding.keyRange _ _ _ _) (h.1 _ _ _ _)
  · intro i k _ j d hj
    rw [base_context]
    change FP8RealMemory.FiniteByte (c.initial.readMemValue .int c.V
      ((valueStrides c).at i.batch.val i.head.val (0+k*64+j.val) d.val))
    rw [binding.value i.batch i.head ⟨_,hj⟩ d]
    exact finite_byte _ (binding.valueRange _ _ _ _) (h.2 _ _ _ _)

#axiomsClean input_metadata
#axiomsClean source_offset_bound
#axiomsClean padded_query
#axiomsClean loop_stop_bound
#axiomsClean padded_key
#axiomsClean base_context
#axiomsClean query_access
#axiomsClean key_access
#axiomsClean value_access
#axiomsClean output_access
#axiomsClean checked_bytes


/-- The entire addressed element, not just its first byte, fits the checked
absolute address envelope. -/
theorem pointer_cell (B : Nat) (c : Context) (o : OutputLayout B c) (fp8 : Bool) (p : Pointers B c o)
    (h : pointerChecks B c o fp8 p=Bool.true) (r : Buffer) (n : Nat) (hn : n<envelope B c o) :
    p.base r+width fp8 r*n+width fp8 r<2^64 := by
  simp only [pointerChecks,List.all_eq_true,decide_eq_true_eq] at h
  have member : r∈buffers := by cases r <;> simp [buffers]
  have bound:=h r member
  have le:=Nat.mul_le_mul_left (width fp8 r) (show n+1≤envelope B c o by omega)
  simp only [Nat.mul_add,Nat.mul_one] at le
  omega

theorem initial_key (c : Context) (j : Fin 64) : j.val≤keyRows c := by
  have hj:=j.isLt
  unfold keyRows
  split <;> omega

theorem key_index_bound (B : Nat) (c : Context) (o : OutputLayout B c)
    (row : Nat) (hr : row≤keyRows c) : row+c.SP<envelope B c o := by
  have hround : (c.H+63)/64*64≤c.H+63 := by omega
  unfold keyRows at hr
  split at hr <;> unfold envelope <;> omega

#axiomsClean pointer_cell
#axiomsClean initial_key
#axiomsClean key_index_bound

end VeriTile.Bench.Repairs.AttentionLlama.Checked
