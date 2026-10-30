import bench.tritonbench_g.context_attn_nopad.Proofs.RealData

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
open Classical
set_option maxHeartbeats 2500000

/-- Logical channel sampling only, independent of addresses and execution. The
zero outside the finite coordinate space is never used by an active output. -/
def sample (c : Call) (values : Coordinate c → ℝ) (b h t d : Nat) : ℝ :=
  if hb : b<c.batches then if hh : h<c.q.heads then
  if ht : t<IOData.timeExtent c then if hd : d<c.q.dims then
    values (⟨b,hb⟩,⟨h,hh⟩,⟨t,ht⟩,⟨d,hd⟩,PUnit.unit)
  else 0 else 0 else 0 else 0

def queryValues (c : Call) (xs : Values c) (j : Coordinate c) : ℝ := xs 0 (encode c j)
def keyValues (c : Call) (xs : Values c) (j : Coordinate c) : ℝ := xs 1 (encode c j)
def valueValues (c : Call) (xs : Values c) (j : Coordinate c) : ℝ := xs 2 (encode c j)

def score (c : Call) (q k : Coordinate c → ℝ) (b h t j : Nat) : ℝ :=
  (∑ d : Fin c.q.dims,sample c q b h t d.val * sample c k b h j d.val)*c.scale

def attention (c : Call) (q k v : Coordinate c → ℝ) (j : Coordinate c) : ℝ :=
  (∑ t ∈ Finset.range (j.2.2.1.val+1),
    Real.exp (score c q k j.1.val j.2.1.val j.2.2.1.val t) *
      sample c v j.1.val j.2.1.val t j.2.2.2.1.val) /
  ∑ t ∈ Finset.range (j.2.2.1.val+1),
    Real.exp (score c q k j.1.val j.2.1.val j.2.2.1.val t)

def expected (c : Call) (xs : Values c) : Fin 1 → Fin (laneCount c) → ℝ :=
  fun _ j=>attention c (queryValues c xs) (keyValues c xs) (valueValues c xs) (decode c j)

theorem input_matches (c : Call) (r : Regions) (xs : Values c)
    (s : BlockState) (hin : (io c r).InputBinding s xs)
    (b h t d : Nat) (hb : b<c.batches) (hh : h<c.q.heads)
    (ht : t<IOData.timeExtent c) (hd : d<c.q.dims) (on : t<c.length b) :
    (Program.inputs s r).q (c.q.offset (c.start b+t) h d)=sample c (queryValues c xs) b h t d ∧
    (Program.inputs s r).k (c.k.offset (c.start b+t) h d)=sample c (keyValues c xs) b h t d ∧
    (Program.inputs s r).v (c.v.offset (c.start b+t) h d)=sample c (valueValues c xs) b h t d := by
  let j : Coordinate c := (⟨b,hb⟩,⟨h,hh⟩,⟨t,ht⟩,⟨d,hd⟩,PUnit.unit)
  have hq := hin (⟨0,by decide⟩ : Fin 5) (encode c j) (by change IOData.active c (decode c (encode c j)); simpa only [decode_encode] using on)
  have hk := hin (⟨1,by decide⟩ : Fin 5) (encode c j) (by change IOData.active c (decode c (encode c j)); simpa only [decode_encode] using on)
  have hv := hin (⟨2,by decide⟩ : Fin 5) (encode c j) (by change IOData.active c (decode c (encode c j)); simpa only [decode_encode] using on)
  change s.readMem r.query (IOData.address c c.q (decode c (encode c j)))=_ at hq
  change s.readMem r.key (IOData.address c c.k (decode c (encode c j)))=_ at hk
  change s.readMem r.value (IOData.address c c.v (decode c (encode c j)))=_ at hv
  rw [decode_encode] at hq hk hv
  simpa [sample,hb,hh,ht,hd,queryValues,keyValues,valueValues,
    IOData.address,j,Program.inputs] using And.intro hq (And.intro hk hv)

theorem expected_matches (c : Call) (r : Regions) (xs : Values c)
    (s : BlockState) (hin : (io c r).InputBinding s xs) (j : Coordinate c)
    (on : IOData.active c j) :
    c.expected (Program.inputs s r) j.1.val j.2.1.val j.2.2.1.val j.2.2.2.1.val=
      attention c (queryValues c xs) (keyValues c xs) (valueValues c xs) j := by
  have hscore (t : Nat) (ht : t<j.2.2.1.val+1) :
      c.score (Program.inputs s r) j.1.val j.2.1.val j.2.2.1.val t=
        score c (queryValues c xs) (keyValues c xs) j.1.val j.2.1.val j.2.2.1.val t := by
    unfold Call.score score
    congr 1
    apply Finset.sum_congr rfl
    intro d _
    rw [(input_matches c r xs s hin _ _ _ _ j.1.isLt j.2.1.isLt
      j.2.2.1.isLt d.isLt on).1]
    rw [(input_matches c r xs s hin _ _ _ _ j.1.isLt j.2.1.isLt
      (by have := j.2.2.1.isLt; omega) d.isLt
      (by change j.2.2.1.val<c.length j.1.val at on; omega)).2.1]
  unfold Call.expected attention
  congr 1
  · apply Finset.sum_congr rfl
    intro t ht
    rw [hscore t (Finset.mem_range.mp ht)]
    rw [(input_matches c r xs s hin _ _ _ _ j.1.isLt j.2.1.isLt
      (by have := j.2.2.1.isLt; have := Finset.mem_range.mp ht; omega) j.2.2.2.1.isLt
      (by change j.2.2.1.val<c.length j.1.val at on; have := Finset.mem_range.mp ht; omega)).2.2]
  · apply Finset.sum_congr rfl
    intro t ht
    rw [hscore t (Finset.mem_range.mp ht)]

#specNonCircular expected avoiding [kernel,VeriTile.Triton.exec]
#axiomsClean expected_matches
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.RealData
