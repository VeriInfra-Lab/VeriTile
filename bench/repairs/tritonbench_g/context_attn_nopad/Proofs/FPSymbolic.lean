import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.FPBackend
import bench.tritonbench_g.context_attn_nopad.Proofs.RealIO

namespace VeriTile.Bench.Repairs.ContextAttnNopad.FPSymbolic
open VeriTile.Triton FP.Structural
open FP.ProfiledRewrite (engine)
open VeriTile.Bench.TritonBenchG.ContextAttnNopad
open Formula
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev program (p : Layout) (r : Program.Regions) (scale : ℝ) :=
  Program.kernel p r .fp32 Bool.false scale

abbrev parts (p : Layout) (r : Program.Regions) (scale : ℝ) :=
  Program.parts p r Bool.false scale

theorem source_split (p : Layout) (r : Program.Regions) (scale : ℝ) :
    (program p r scale).surfaceBody =
      (parts p r scale).before ++
      [.forRangeInt "start_n" (parts p r scale).start
        (parts p r scale).stop (parts p r scale).stride
        (parts p r scale).body] ++ (parts p r scale).after := rfl

def Has {α : Type} (s : State α) (d : TileDType) (sh : TileShape) (name : RegName) : Prop :=
  (s.regs d sh name).isSome

def integerInputs {α : Type} [Inhabited α] (r : Program.Regions) (s : State α) : Formula.Inputs :=
  ⟨fun _ => 0,fun _ => 0,fun _ => 0,
    fun b => (s.mem r.start b).read .int,fun b => (s.mem r.sequence b).read .int⟩

def launchOf {α : Type} (s : State α) : Launch := ⟨s.pids 2,s.pids 0,s.pids 1⟩

def Geometry {α : Type} [Inhabited α] (p : Layout) (r : Program.Regions)
    (initial s : State α) : Prop :=
  s.regs .nat [] "cur_head" = some (fun _ => initial.pids 1) ∧
  s.regs .int [] "cur_batch_in_all_start_index" =
    some (fun _ => (integerInputs r initial).start (initial.pids 0)) ∧
  s.regs .int [] "cur_batch_seq_len" =
    some (fun _ => (integerInputs r initial).sequence (initial.pids 0)) ∧
  s.regs .nat [p.BM] "offs_m" = some (fun i => queryRow p (launchOf initial) i.1.val) ∧
  s.regs .nat [p.D] "offs_d" = some (fun i => i.1.val)

def Controls {α : Type} (p : Layout) (s t : State α) : Prop :=
  t.regs .nat [] "cur_head" = s.regs .nat [] "cur_head" ∧
  t.regs .int [] "cur_batch_in_all_start_index" = s.regs .int [] "cur_batch_in_all_start_index" ∧
  t.regs .int [] "cur_batch_seq_len" = s.regs .int [] "cur_batch_seq_len" ∧
  t.regs .nat [p.BM] "offs_m" = s.regs .nat [p.BM] "offs_m" ∧
  t.regs .nat [p.D] "offs_d" = s.regs .nat [p.D] "offs_d"

theorem Controls.refl {α : Type} (p : Layout) (s : State α) : Controls p s s :=
  ⟨rfl,rfl,rfl,rfl,rfl⟩
theorem Controls.trans {α : Type} {p : Layout} {s t u : State α}
    (h : Controls p s t) (k : Controls p t u) : Controls p s u := by
  rcases h with ⟨h1,h2,h3,h4,h5⟩
  rcases k with ⟨k1,k2,k3,k4,k5⟩
  exact ⟨k1.trans h1,k2.trans h2,k3.trans h3,k4.trans h4,k5.trans h5⟩
theorem Geometry.preserve {α : Type} [Inhabited α] {p : Layout} {r : Program.Regions}
    {initial s t : State α} (h : Geometry p r initial s) (k : Controls p s t) :
    Geometry p r initial t := by
  rcases h with ⟨h1,h2,h3,h4,h5⟩
  rcases k with ⟨k1,k2,k3,k4,k5⟩
  exact ⟨k1.trans h1,k2.trans h2,k3.trans h3,k4.trans h4,k5.trans h5⟩

/-- Exact source active-write footprint, reusing the existing integer address
and mask formulas. No opaque floating result occurs in this predicate. -/
def Writes {α : Type} [Inhabited α] (p : Layout) (r : Program.Regions)
    (initial : State α) (rg : RegionName) (off : Nat) : Prop :=
  ∃ i : RealIO.Lane p, RealIO.active p (launchOf initial) (integerInputs r initial) i ∧
    r.out = rg ∧ RealIO.write p (launchOf initial) (integerInputs r initial) i = off

def Ready {α : Type} (p : Layout) (s : State α) : Prop :=
  Has s .nat [] "cur_head" ∧ Has s .nat [] "start_m" ∧
  Has s .int [] "cur_batch_in_all_start_index" ∧ Has s .int [] "cur_batch_seq_len" ∧
  Has s .nat [p.BM] "offs_m" ∧ Has s .nat [p.BN] "offs_n" ∧ Has s .nat [p.D] "offs_d" ∧
  Has s .real [p.BM,p.D] "q" ∧
  Has s .ptr [p.D,p.BN] "k_ptrs" ∧ Has s .ptr [p.BN,p.D] "v_ptrs" ∧
  Has s .real [p.BM] "m_i" ∧ Has s .real [p.BM] "l_i" ∧
  Has s .real [p.BM,p.D] "acc" ∧ Has s .nat [] "block_mask"

/-- Construct every prelude register without assuming any execution result. -/
theorem prelude {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ) (s : State α) :
    ∃ t, run (engine o.algebra) (parts p r scale).before s = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Geometry p r s t := by
  simp (config := {maxSteps := 300000}) [parts,program,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,Source.intRange?,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
    ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,Ready,Has,
    Geometry,integerInputs,launchOf,queryRow]
  rfl

/-- Execute one whole iteration on opaque FP values, preserving all registers
required by the next iteration and leaving every memory cell unchanged. -/
theorem body {α : Type} [Inhabited α] (o : FPBackend.Ops α)
    (p : Layout) (r : Program.Regions) (scale : ℝ)
    (s : State α) (hBN : 0 < p.BN) (ready : Ready p s) (index : Int) :
    ∃ t, run (engine o.algebra) (parts p r scale).body
        (s.setReg "start_n" .int [] (fun _ => index)) = some t ∧
      Ready p t ∧ t.mem = s.mem ∧ Controls p s t := by
  rcases ready with ⟨hh,hb,hs,hl,hr,hk,hd,hq,hkp,hvp,hm,hn,ha,hmask⟩
  obtain ⟨head,hh⟩ := Option.isSome_iff_exists.mp hh
  obtain ⟨block,hb⟩ := Option.isSome_iff_exists.mp hb
  obtain ⟨start,hs⟩ := Option.isSome_iff_exists.mp hs
  obtain ⟨length,hl⟩ := Option.isSome_iff_exists.mp hl
  obtain ⟨rows,hr⟩ := Option.isSome_iff_exists.mp hr
  obtain ⟨keys,hk⟩ := Option.isSome_iff_exists.mp hk
  obtain ⟨dims,hd⟩ := Option.isSome_iff_exists.mp hd
  obtain ⟨query,hq⟩ := Option.isSome_iff_exists.mp hq
  obtain ⟨kptr,hkp⟩ := Option.isSome_iff_exists.mp hkp
  obtain ⟨vptr,hvp⟩ := Option.isSome_iff_exists.mp hvp
  obtain ⟨center,hm⟩ := Option.isSome_iff_exists.mp hm
  obtain ⟨normalizer,hn⟩ := Option.isSome_iff_exists.mp hn
  obtain ⟨accumulator,ha⟩ := Option.isSome_iff_exists.mp ha
  obtain ⟨mask,hmask⟩ := Option.isSome_iff_exists.mp hmask
  simp (config := {maxSteps := 300000}) [parts,program,Program.parts,Program.mathematical,Program.kernel,
    VeriTile.Bench.TritonBenchG.ContextAttnNopad.original,Source.intRange?,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,FPBackend.Ops.algebra,
    ComputeDType.eraseDType,numeric,natLt,natLe,numericLt,numericLe,numericEq,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,integralFloorDiv,integralMod,
    Ready,Has,Controls,hh,hb,hs,hl,hr,hk,hd,hq,hkp,hvp,hm,hn,ha,hmask,hBN]

end
end VeriTile.Bench.Repairs.ContextAttnNopad.FPSymbolic
