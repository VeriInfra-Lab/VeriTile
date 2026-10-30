import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Physical
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.NativeStaging
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

variable (runtime : Runtime)

def zeroExpr : Expr := fun _=>.zero

def logicalGraph {α : Type} (I : CompareISA α) (c : Config) (s : State Expr) (hbm : 0<c.BM)
    (hz n d : Nat) (hd : d<c.BD) : Graph :=
  if n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat then
    reference runtime I c (start (grid c) (n/c.BM,hz,0) s) (lane c hbm n d hd)
  else zero

def readNative {α : Type} (I : CompareISA α) (c : Config) (cell : Cell Expr) (d : Nat) : α :=
  (lower ((toFloat .real (cell.read .real)) (physicalOwner d))).eval I.toISA

theorem read_native_word {α : Type} (I : CompareISA α) (c : Config) (value : Expr) (d : Nat) :
    readNative I c (.mk .real (ofFloat .real value)) d=
      (lower (value (physicalOwner d))).eval I.toISA := by
  unfold readNative
  have h : toFloat .real
      ((Cell.mk .real (ofFloat .real value)).read .real)=value := by
    cases hs : c.storage <;> rfl
  rw [h]

theorem source_value_graph {α : Type} (I : CompareISA α) (c : Config) (s : State Expr) (hbm : 0<c.BM)
    (hz n d : Nat) (hd : d<c.BD) :
    lower ((if n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat then
      Whole.answer (algebra runtime) (dotModel c.storage I) c (start (grid c) (n/c.BM,hz,0) s) (lane c hbm n d hd)
     else zeroExpr) (physicalOwner d))=logicalGraph runtime I c s hbm hz n d hd := by
  by_cases hn : n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat
  · simp only [if_pos hn,logicalGraph,reference,Whole.answer,lane]
  · simp [hn,logicalGraph,zeroExpr,lower]

/-- Complete finite returned logical values of the staged native source.
This remains parameterized by scalar/fixed-instruction ISA semantics. -/
theorem returned_native {α : Type} (I : CompareISA α) (c : Config) (before : Physical.Store Expr)
    (fresh : before.capacity c.out=0) (sep : Separated c) (layout : Contiguous c)
    (hl : LengthsBound c before.extend) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (legal : (programs (grid c)).Perm work) :
    ∃ result,Physical.invoke (fun _=>algebra runtime) (fun _=>dotModel c.storage I) c zeroExpr before work=some result ∧
      result.result=returnedTensor c ∧ result.memory.capacity=Physical.allocatedCapacity c before ∧
      (∀ hz<c.Z*c.H,∀ n<c.T,∀ d (hd : d<c.BD),
        ∃ i : Fin (result.memory.capacity c.out),i.val=logicalAddress c hz n d ∧
          readNative I c (result.memory.cells c.out i) d=
            (logicalGraph runtime I c before.extend hbm hz n d hd).eval I.toISA) ∧
      (∀ r (i : Fin (before.capacity r)),∃ j : Fin (result.memory.capacity r),
        j.val=i.val ∧ result.memory.cells r j=before.cells r i) := by
  obtain ⟨result,hr,hdesc,hcap,hv,hf⟩:=Physical.finite_return (fun _=>algebra runtime) (fun _=>dotModel c.storage I)
    c zeroExpr before fresh sep layout hl hbm hbn work legal
  refine ⟨result,hr,hdesc,hcap,?_,hf⟩
  intro hz hh n hn d hd
  obtain ⟨i,hi,hval⟩:=hv hz hh n hn d hd
  refine ⟨i,hi,?_⟩
  rw [hval,read_native_word,source_value_graph]

theorem logical_transport {α β : Type} (f : α → β) (A : CompareISA α) (B : CompareISA β)
    (h : ComparisonTransport f A B) (c : Config) (s : State Expr) (hbm : 0<c.BM)
    (hz n d : Nat) (hd : d<c.BD) :
    f ((logicalGraph runtime A c s hbm hz n d hd).eval A.toISA)=
      (logicalGraph runtime B c s hbm hz n d hd).eval B.toISA := by
  unfold logicalGraph
  split
  · exact native_local_transport runtime f A B h c _ _
  · exact transport f A.toISA B.toISA h.primitive zero

/-- Every returned cell transports under local numerical laws. The theorem
assumes neither a whole reduction/dot equation nor an output agreement. -/
theorem returned_local_transport {α β : Type} (f : α → β) (A : CompareISA α) (B : CompareISA β)
    (h : ComparisonTransport f A B) (c : Config) (before : Physical.Store Expr)
    (fresh : before.capacity c.out=0) (sep : Separated c) (layout : Contiguous c)
    (hl : LengthsBound c before.extend) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (legal : (programs (grid c)).Perm work) :
    ∃ result,Physical.invoke (fun _=>algebra runtime) (fun _=>dotModel c.storage A) c zeroExpr before work=some result ∧
      result.result=returnedTensor c ∧
      (∀ hz<c.Z*c.H,∀ n<c.T,∀ d (hd : d<c.BD),
        ∃ i : Fin (result.memory.capacity c.out),i.val=logicalAddress c hz n d ∧
          f (readNative A c (result.memory.cells c.out i) d)=
            (logicalGraph runtime B c before.extend hbm hz n d hd).eval B.toISA) := by
  obtain ⟨result,hr,hdesc,_,hv,_⟩:=returned_native runtime A c before fresh sep layout hl hbm hbn work legal
  refine ⟨result,hr,hdesc,?_⟩
  intro hz hh n hn d hd
  obtain ⟨i,hi,hval⟩:=hv hz hh n hn d hd
  exact ⟨i,hi,by rw [hval,logical_transport runtime f A B h]⟩

#axiomsClean returned_native
#axiomsClean returned_local_transport
#specNonCircular logicalGraph avoiding [FP.Structural.exec,Whole.invoke]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Native
