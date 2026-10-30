import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Physical
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.Access
import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Proofs.RealAnswer
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
open VeriTile Triton FP.Structural Source Whole
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000

/-- Independent attention specification at an ordinary logical tensor index.
Metadata is read from the caller's contiguous arrays, and duplicate selections
retain their occurrence multiplicity. -/
def sparseExpected (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (hz n d : Nat) : ℝ :=
  let row:=hz*c.NR+n/c.BM
  VeriTile.Bench.Repairs.MixedSparseAttention.Whole001.MathTarget.expected c.BD c.BN ((s.mem (Region.cast c.seq) (hz/c.H)).read .nat) n
    ((s.mem (Region.cast c.blocks) row).read .nat) ((s.mem (Region.cast c.counts) row).read .nat)
    c.scale (fun k=>μ c.q (logicalAddress c hz n k))
    (fun j k=>μ c.k (logicalAddress c hz j k)) (fun j k=>μ c.v (logicalAddress c hz j k))
    (fun b=>(s.mem (Region.cast c.offsets) (row*c.NS+b)).read .nat)
    (fun j=>(s.mem (Region.cast c.columns) (row*c.NV+j)).read .nat) d

def logicalExpected (c : Config) (s : State R) (μ : RegionName → Nat → ℝ) (hz n d : Nat) : ℝ :=
  if n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat then sparseExpected c s μ hz n d else 0

theorem expected_dense (c : Config) (layout : Contiguous c) (s : State R)
    (μ : RegionName → Nat → ℝ) (hbm : 0<c.BM) (hz n d : Nat) (hd : d<c.BD) :
    expected c (start (grid c) (n/c.BM,hz,0) s) μ (lane c hbm n d hd).1
      (lane c hbm n d hd).2.1=sparseExpected c s μ hz n d := by
  have hrow : n/c.BM*c.BM+n%c.BM=n := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod n c.BM
  unfold expected sparseExpected
  delta qAt kAt vAt blockOffset columnIndex
  simp only [Access.query_address c layout,Access.key_address c layout,
    Access.value_address c layout]
  simp [sequenceLength,rowStart,blockCount,columnCount,blockOffset,columnIndex,metadataIndex,
    start,ids,lane,hrow]

theorem source_value_real (c : Config) (layout : Contiguous c) (s : State R)
    (μ : RegionName → Nat → ℝ) (reads : Reads c s μ) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (hz n d : Nat) (hd : d<c.BD) :
    (if n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat then
      Whole.answer algebra dotModel c (start (grid c) (n/c.BM,hz,0) s) (lane c hbm n d hd)
     else some 0)=some (logicalExpected c s μ hz n d) := by
  by_cases hn : n<(s.mem (Region.cast c.seq) (hz/c.H)).read .nat
  · have hrow : n/c.BM*c.BM+n%c.BM=n := by
      simpa only [Nat.mul_comm] using Nat.div_add_mod n c.BM
    have hreads : Reads c (start (grid c) (n/c.BM,hz,0) s) μ:=reads
    have hi : rowStart c (start (grid c) (n/c.BM,hz,0) s)+(lane c hbm n d hd).1.val<
        sequenceLength c (start (grid c) (n/c.BM,hz,0) s) := by
      simpa [rowStart,sequenceLength,start,ids,lane,hrow] using hn
    have h:=answer_real c (start (grid c) (n/c.BM,hz,0) s) μ hreads
      (lane c hbm n d hd).1 (lane c hbm n d hd).2.1 hi hbn
    rw [expected_dense c layout] at h
    simpa only [if_pos hn,logicalExpected,Whole.answer,outputValue,answer,algebra] using h
  · simp [hn,logicalExpected]

/-- Whole host call to independent real sparse attention, complete shape and
exterior frame. No result-shaped hypothesis is used. -/
theorem returned_real (c : Config) (s : State R) (μ : RegionName → Nat → ℝ)
    (reads : Reads c s μ) (sep : Separated c) (layout : Contiguous c)
    (hl : LengthsBound c s) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (legal : (programs (grid c)).Perm work) :
    ∃ result,Whole.invoke (fun _=>algebra) (fun _=>dotModel) c (some 0) s work=some result ∧
      result.result=returnedTensor c ∧
      (∀ hz<c.Z*c.H,∀ n<c.T,∀ d (hd : d<c.BD),
        result.memory.mem c.out (logicalAddress c hz n d)=
          .mk .real (ofFloat .real (some (logicalExpected c s μ hz n d)))) ∧
      (∀ r off,Exterior c r off → result.memory.mem r off=s.mem r off) := by
  obtain ⟨result,hr,hdesc,hv,hf⟩:=returned_source (fun _=>algebra) (fun _=>dotModel)
    c (some 0) s sep layout hl hbm hbn work legal
  refine ⟨result,hr,hdesc,?_,hf⟩
  intro hz hh n hn d hd
  rw [hv hz hh n hn d hd,source_value_real c layout s μ reads hbm hbn]

/-- The same independent specification for an actual finite returned tensor,
including fresh allocation and preservation of all prior finite cells. -/
theorem returned_finite (c : Config) (before : Physical.Store R)
    (fresh : before.capacity c.out=0) (μ : RegionName → Nat → ℝ)
    (reads : Reads c before.extend μ) (sep : Separated c) (layout : Contiguous c)
    (hl : LengthsBound c before.extend) (hbm : 0<c.BM) (hbn : 0<c.BN)
    (work : List Program) (legal : (programs (grid c)).Perm work) :
    ∃ result,Physical.invoke (fun _=>algebra) (fun _=>dotModel) c (some 0) before work=some result ∧
      result.result=returnedTensor c ∧ result.memory.capacity=Physical.allocatedCapacity c before ∧
      (∀ hz<c.Z*c.H,∀ n<c.T,∀ d (hd : d<c.BD),
        ∃ i : Fin (result.memory.capacity c.out),i.val=logicalAddress c hz n d ∧
          result.memory.cells c.out i=.mk .real
            (ofFloat .real (some (logicalExpected c before.extend μ hz n d)))) ∧
      (∀ r (i : Fin (before.capacity r)),∃ j : Fin (result.memory.capacity r),
        j.val=i.val ∧ result.memory.cells r j=before.cells r i) := by
  obtain ⟨result,hr,hdesc,hcap,hv,hf⟩:=Physical.finite_return (fun _=>algebra) (fun _=>dotModel)
    c (some 0) before fresh sep layout hl hbm hbn work legal
  refine ⟨result,hr,hdesc,hcap,?_,hf⟩
  intro hz hh n hn d hd
  obtain ⟨i,hi,hval⟩:=hv hz hh n hn d hd
  refine ⟨i,hi,?_⟩
  rw [hval,source_value_real c layout before.extend μ reads hbm hbn]

#axiomsClean returned_real
#axiomsClean returned_finite
#specNonCircular logicalExpected avoiding [FP.Structural.exec,Source.finalValues,Source.update,Whole.invoke]
end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Real
