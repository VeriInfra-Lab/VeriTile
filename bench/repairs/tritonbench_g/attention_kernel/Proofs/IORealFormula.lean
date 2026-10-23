import bench.repairs.tritonbench_g.attention_kernel.Proofs.IOMemoryBinding

namespace VeriTile.Bench.Repairs.AttentionKernel.IORealFormula
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Whole002
open IOFormulaBinding IOMemoryBinding
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def realRead (p : Parameters) (s : State Real.R) (r : RegionName) (off : Nat) : ℝ :=
  (readState p s r off).unbotD 0

/-- A proof-only completion of unused cells. Actual input windows are kept;
the source result is insensitive to every other cell by `source_at`. -/
def project (p : Parameters) (s : State Real.R) : State Real.R :=
  {s with mem:=fun r off=>.mk p.storage.tile (ofFloat p.storage.float (some (realRead p s r off)))}

theorem finite_real_reads (p : Parameters) (vals : ChannelValues ℝ (IO.inputs p))
    (s : State Real.R) (b : Fin p.batches) (h : Fin p.heads)
    (loads : Reads p (readState p s) b.val h.val (encodedViews some p vals b h)) :
    Reads p (realRead p s) b.val h.val (Target.views p vals b h) := by
  constructor
  · intro r hr d hd
    simpa [realRead,encodedViews] using congrArg (fun x : Real.R=>x.unbotD 0) (loads.query r hr d hd)
  · intro r hr d hd
    simpa [realRead,encodedViews] using congrArg (fun x : Real.R=>x.unbotD 0) (loads.key r hr d hd)
  · intro r hr d hd
    simpa [realRead,encodedViews] using congrArg (fun x : Real.R=>x.unbotD 0) (loads.value r hr d hd)
  · intro r hr c hc
    simpa [realRead,encodedViews] using congrArg (fun x : Real.R=>x.unbotD 0) (loads.bias r hr c hc)

theorem projected_reads (p : Parameters) (vals : ChannelValues ℝ (IO.inputs p))
    (s : State Real.R) (b : Fin p.batches) (h : Fin p.heads)
    (loads : Reads p (realRead p s) b.val h.val (Target.views p vals b h)) :
    Reads p (readState p (project p s)) b.val h.val (encodedViews some p vals b h) := by
  constructor
  · intro r hr d hd
    simpa only [readState,project,Cell.read_same,Real.float_roundtrip,encodedViews] using
      congrArg (fun x : ℝ=>(some x : Real.R)) (loads.query r hr d hd)
  · intro r hr d hd
    simpa only [readState,project,Cell.read_same,Real.float_roundtrip,encodedViews] using
      congrArg (fun x : ℝ=>(some x : Real.R)) (loads.key r hr d hd)
  · intro r hr d hd
    simpa only [readState,project,Cell.read_same,Real.float_roundtrip,encodedViews] using
      congrArg (fun x : ℝ=>(some x : Real.R)) (loads.value r hr d hd)
  · intro r hr c hc
    simpa only [readState,project,Cell.read_same,Real.float_roundtrip,encodedViews] using
      congrArg (fun x : ℝ=>(some x : Real.R)) (loads.bias r hr c hc)

theorem mathematical_eq (p : Parameters) (guard : Guard p) (read : RegionName → Nat → ℝ)
    (batch head n d : Nat) (hn : n<p.length) (hd : d<p.dim) (inputs : Target.Views ℝ)
    (loads : Reads p read batch head inputs) :
    RealReturned.target p read batch head n d=
      RealMath.expected p.length p.dim p.bn p.scale inputs.q inputs.k inputs.v inputs.bias n d := by
  have scores (j : Nat) (hj : j<p.length) :
      RealMath.score p.dim p.bn p.scale
        (fun m e=>read p.Q (offset p.q batch head m e))
        (fun k e=>read p.K (offset p.k batch head k e))
        (fun m c=>read p.Bias (offset p.bias batch head m 0+c)) n j=
      RealMath.score p.dim p.bn p.scale inputs.q inputs.k inputs.bias n j := by
    have dot : (Finset.range p.dim).sum (fun e=>
        read p.Q (offset p.q batch head n e)*read p.K (offset p.k batch head j e))=
        (Finset.range p.dim).sum (fun e=>inputs.q n e*inputs.k j e) := by
      apply Finset.sum_congr rfl
      intro e he
      rw [loads.query n hn e (Finset.mem_range.mp he),loads.key j hj e (Finset.mem_range.mp he)]
    have dynamic:=loads.bias n hn (j/p.bn) (bias_index_bound p guard j hj)
    have static:=loads.bias n hn (64+(j%p.bn)%64) (by omega)
    dsimp only [RealMath.score]
    rw [dot]
    simp only [offset,Nat.zero_mul,Nat.add_zero,dynamic,static]
  unfold RealReturned.target RealMath.expected
  congr 1
  · apply Finset.sum_congr rfl
    intro j hj
    rw [scores j (Finset.mem_range.mp hj)]
    dsimp only
    rw [loads.value j (Finset.mem_range.mp hj) d hd]
  · apply Finset.sum_congr rfl
    intro j hj
    rw [scores j (Finset.mem_range.mp hj)]

theorem answer_eq (p : Parameters) (guard : Guard p) (hbm : 0<p.bm)
    (vals : ChannelValues ℝ (IO.inputs p)) (s : State Real.R)
    (b : Fin p.batches) (h : Fin p.heads) (n : Fin p.length) (d : Fin p.dim)
    (loads : Reads p (readState p s) b.val h.val (encodedViews some p vals b h)) :
    answer Real.algebra Real.dotModel p
      (start (grid p) (n.val/p.bm,b.val*p.heads+h.val,0) s) (lane p hbm n.val d.val d.isLt)=
      some (RealMath.expected p.length p.dim p.bn p.scale
        (Target.views p vals b h).q (Target.views p vals b h).k
        (Target.views p vals b h).v (Target.views p vals b h).bias n.val d.val) := by
  have finite:=finite_real_reads p vals s b h loads
  have projected:=projected_reads p vals s b h finite
  have same : answer Real.algebra Real.dotModel p
      (start (grid p) (n.val/p.bm,b.val*p.heads+h.val,0) s)=
      answer Real.algebra Real.dotModel p
        (start (grid p) (n.val/p.bm,b.val*p.heads+h.val,0) (project p s)) :=
    (source_at Real.algebra Real.dotModel p s _ _ _ h.isLt _ loads guard).trans
      (source_at Real.algebra Real.dotModel p (project p s) _ _ _ h.isLt _ projected guard).symm
  rw [same]
  have global : Real.Reads p
      (start (grid p) (n.val/p.bm,b.val*p.heads+h.val,0) (project p s)) (realRead p s) := by
    intro r hr off
    rfl
  have active : Source.active p
      (start (grid p) (n.val/p.bm,b.val*p.heads+h.val,0) (project p s))
      (lane p hbm n.val d.val d.isLt) := by
    change n.val/p.bm*p.bm+n.val%p.bm<p.length
    simpa only [Nat.mul_comm,Nat.div_add_mod] using n.isLt
  rw [Real.answer_real p _ _ global _ guard.bn active,
    RealReturned.target_at_program p (project p s) (realRead p s) hbm b.val h.val n.val d.val h.isLt d.isLt,
    mathematical_eq p guard (realRead p s) b.val h.val n.val d.val n.isLt d.isLt _ finite]

#axiomsClean answer_eq
end
end VeriTile.Bench.Repairs.AttentionKernel.IORealFormula
