import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeBody
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.SourceAccess
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeReference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open NativeGraph FPSuccess SourcePointers
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Independent numerical input arrays. No source state or executor occurs in
these operands or the configured K-block recurrence below. -/
structure Inputs where
  a : Nat → Expr
  b : Nat → Expr
  seq : Nat → Int

def zero : Expr := .literal (some .fp32) .real 0

def aTile (h : Host.Call) (c : Host.Config) (d : Coordinates) (x : Inputs)
    (q : Nat) (i : TileIndex [c.M,c.K]) : Expr :=
  .cast (some .fp32) .real .real
    (.fp32Load (if SourceAccess.aMask h c d q i.1 i.2.1 then x.a (aOffset h c d q i.1 i.2.1) else zero))
def bTile (h : Host.Call) (c : Host.Config) (d : Coordinates) (x : Inputs)
    (q : Nat) (i : TileIndex [c.K,c.N]) : Expr :=
  .cast (some .fp32) .real .real
    (.fp32Load (if SourceAccess.bMask h c d q i.1 i.2.1 then x.b (bOffset h c d q i.1 i.2.1) else zero))

/-- Literal configured K-block order. Each dotAcc is interpreted by the fixed
K8 instruction recurrence, including its actual carry; no FP reassociation. -/
def accumulator (h : Host.Call) (c : Host.Config) (d : Coordinates) (x : Inputs) :
    Nat → TileIndex [c.M,c.N] → Expr
  | 0,_ => zero
  | q+1,i => .dotAcc c (some .fp32) .fp32 .fp32 .fp32 options [] c.M c.K c.N
      (aTile h c d x q) (bTile h c d x q) (accumulator h c d x q i) i

def sequence (h : Host.Call) (d : Coordinates) (x : Inputs) (lane : Nat) (fallback : Int) :=
  if lane<limit h d then x.seq (seqOffset h d lane) else fallback

def output (h : Host.Call) (c : Host.Config) (d : Coordinates) (x : Inputs)
    (i : TileIndex [c.M,c.N]) : Expr :=
  .cast (some .fp32) .real (Host.outputDType h.aStorage h.requestedOutput)
    (if h.hasSeq then
       if sequence h d x (row c d i.1) (-1) = sequence h d x (col c d i.2.1) (-2)
       then accumulator h c d x (Host.cdiv h.shape.K c.K) i else zero
     else accumulator h c d x (Host.cdiv h.shape.K c.K) i)

def capture (h : Host.Call) (memory : RegionName → Nat → Cell Expr) : Inputs :=
  ⟨fun off => (memory h.A off).read .real,fun off => (memory h.B off).read .real,
   fun off => (memory (Region.cast h.Seq) off).read .int⟩

theorem setup_zero (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (s : State Expr) (ready : Base s) (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine (backend rest c).algebra) (parts h c).before s=some t ∧
      t.regs .real [c.M,c.N] "acc"=some (fun _ => zero) := by
  rcases ready with ⟨h0,h1,h2,h3,h4⟩
  obtain ⟨v0,h0⟩ := Option.isSome_iff_exists.mp h0
  obtain ⟨v1,h1⟩ := Option.isSome_iff_exists.mp h1
  obtain ⟨v2,h2⟩ := Option.isSome_iff_exists.mp h2
  obtain ⟨v3,h3⟩ := Option.isSome_iff_exists.mp h3
  obtain ⟨v4,h4⟩ := Option.isSome_iff_exists.mp h4
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 40000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,VeriTile.Bench.Repairs.BmmChunkFwd.originalKernel,
      ha,hb,seq,Source.conditional?,Source.signedRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,integralFloorDiv,integralMod,
      h0,h1,h2,h3,h4,backend,algebra,zero]

theorem body_reference (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (q : Nat) (s : State Expr) (hs : StateAt h c d q s)
    (x : Inputs) (hx : capture h s.mem=x)
    (acc : s.regs .real [c.M,c.N] "acc"=some (accumulator h c d x q))
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine (backend rest c).algebra) (parts h c).body
      (s.setReg "k" .int [] (fun _ => q))=some t ∧ StateAt h c d (q+1) t ∧
      t.mem=s.mem ∧ t.regs .real [c.M,c.N] "acc"=some (accumulator h c d x (q+1)) := by
  obtain ⟨t,ht,hstate,hmem⟩ := SourcePointers.body (backend rest c) h c d q s hs ha hb
  obtain ⟨t',a,b,carry,ht',hc,hav,hbv,hnext⟩ := body_fp32 rest h c s (ready h c d q s hs) q ha hb
  have same : t'=t := Option.some.inj (ht'.symm.trans ht)
  subst t'
  obtain ⟨t',ht',ha',hb'⟩ := SourceAccess.loads (backend rest c) h c d q s hs ha hb
  have same : t'=t := Option.some.inj (ht'.symm.trans ht)
  subst t'
  have ac : carry=accumulator h c d x q := Option.some.inj (hc.symm.trans acc)
  have aa : a=aTile h c d x q := by
    apply Option.some.inj
    rw [←hav,ha']
    apply congrArg some
    funext i
    rw [←hx]
    rfl
  have bb : b=bTile h c d x q := by
    apply Option.some.inj
    rw [←hbv,hb']
    apply congrArg some
    funext i
    rw [←hx]
    rfl
  exact ⟨t,ht,hstate,hmem,by simpa only [accumulator,ac,aa,bb] using hnext⟩

theorem iterations_reference (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (cur stop : Nat) (bound : cur≤stop) (s : State Expr)
    (hs : StateAt h c d cur s) (x : Inputs) (hx : capture h s.mem=x)
    (acc : s.regs .real [c.M,c.N] "acc"=some (accumulator h c d x cur))
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, rangeInt (engine (backend rest c).algebra) "k" cur stop 1 (parts h c).body s=some t ∧
      StateAt h c d stop t ∧ t.mem=s.mem ∧
      t.regs .real [c.M,c.N] "acc"=some (accumulator h c d x stop) := by
  by_cases less : cur<stop
  · obtain ⟨mid,hm,hr,hmem,hacc⟩ := body_reference rest h c d cur s hs x hx acc ha hb
    obtain ⟨t,ht,htr,htm,hta⟩ := iterations_reference rest h c d (cur+1) stop (by omega)
      mid hr x (by rw [hmem]; exact hx) hacc ha hb
    refine ⟨t,?_,htr,htm.trans hmem,hta⟩
    rw [rangeInt_nat_step,if_neg (by decide),if_pos less,hm]
    exact ht
  · have eq : cur=stop := by omega
    exact ⟨s,by rw [rangeInt_nat_step,if_neg (by decide),if_neg less],eq ▸ hs,rfl,eq ▸ acc⟩
termination_by stop-cur

def writeMemory (memory : RegionName → Nat → Cell Expr) (region : RegionName)
    (offset : Nat) (v : Cell Expr) : RegionName → Nat → Cell Expr :=
  fun r off => if r=region ∧ off=offset then v else memory r off

def scatter (h : Host.Call) (c : Host.Config) (d : Coordinates) (x : Inputs)
    (memory : RegionName → Nat → Cell Expr) :=
  (TileShape.allIndices [c.M,c.N]).foldl (fun m i =>
    if SourceAccess.outMask h c d i.1 i.2.1 then
      writeMemory m h.Out (outOffset h c d i.1 i.2.1)
        (.mk (Host.outputDType h.aStorage h.requestedOutput).toTileDType
          (ofFloat (Host.outputDType h.aStorage h.requestedOutput) (output h c d x i)))
    else m) memory

theorem scatter_memory {ι : Type} (r : ι → RegionName) (off : ι → Nat)
    (v : ι → Cell Expr) (mask : ι → Prop) [DecidablePred mask] (l : List ι) (s : State Expr) :
    (l.foldl (fun t i => if mask i then t.write (r i) (off i) (v i) else t) s).mem=
      l.foldl (fun m i => if mask i then writeMemory m (r i) (off i) (v i) else m) s.mem := by
  induction l generalizing s with
  | nil => rfl
  | cons i l ih =>
    simp only [List.foldl_cons]
    by_cases hi : mask i <;> simp only [hi,↓reduceIte,ih] <;> rfl

/-- Exact numerical values and writes of the executed source epilogue, with the
signed sequence loads tied to the independent input arrays. -/
theorem epilogue_reference (rest : Algebra Expr) (h : Host.Call) (c : Host.Config)
    (d : Coordinates) (s : State Expr) (hs : StateAt h c d (Host.cdiv h.shape.K c.K) s)
    (x : Inputs) (hx : capture h s.mem=x)
    (acc : s.regs .real [c.M,c.N] "acc"=some (accumulator h c d x (Host.cdiv h.shape.K c.K)))
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) :
    ∃ t, run (engine (backend rest c).algebra) (parts h c).after s=some t ∧
      t.mem=scatter h c d x s.mem := by
  subst x
  rcases hs with ⟨⟨h0,h1,h2,h3,h4⟩,h5,h6,h7,h8,h9,h10,h11,hseq⟩
  cases seq : h.hasSeq
  all_goals
    simp (config := {maxSteps := 40000}) [parts,branch,Kernels.source,Host.Call.kernel,TypedSource.specialize,VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource,VeriTile.Bench.Repairs.BmmChunkFwd.originalKernel,
      ha,hb,seq,Source.conditional?,Source.signedRange?,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,Ops.algebra,ComputeDType.eraseDType,
      numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,TileShape.axisDim,
      Region.cast,integralFloorDiv,integralMod,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,acc,hseq,FP.Structural.store,
      backend,algebra,scatter_memory,scatter,output,SourceAccess.outMask,
      SourcePointers.row,col,seqBase,seqOffset,outOffset,sequence,zero,minimum]
  all_goals
    apply congrArg (fun f => List.foldl f s.mem (TileShape.allIndices [c.M,c.N]))
    funext memory i
    have hm (n : Nat) : (n : Int)<
        (if (h.shape.chunk : Int)<(h.shape.seqlen : Int)-((d.chunk*h.shape.chunk : Nat) : Int)
         then (h.shape.chunk : Int) else (h.shape.seqlen : Int)-((d.chunk*h.shape.chunk : Nat) : Int))
        ↔ n<limit h d := by
      simp only [limit,←min_def_lt,←Int.natCast_mul,min_def]
      split_ifs <;> omega
    simp only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt,
      Int.toNat_natCast,hm]
    cases grouped : h.shape.grouped
    all_goals
      simp only [grouped,Bool.false_eq_true,↓reduceIte,
        ←Int.natCast_mul,←Int.natCast_add,Int.toNat_natCast]
      try rfl
    all_goals
      simp only [hm,add_zero,←Int.natCast_zero,←Int.natCast_add,
        Int.toNat_natCast,capture,Region.cast]
      try rfl
    all_goals norm_num only [zero]

#specNonCircular accumulator avoiding [Kernels.source,FP.Structural.exec,FP.Structural.run]
#specNonCircular output avoiding [Kernels.source,FP.Structural.exec,FP.Structural.run]
#axiomsClean setup_zero
#axiomsClean body_reference
#axiomsClean iterations_reference
#axiomsClean epilogue_reference
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeReference
