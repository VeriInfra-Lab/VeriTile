import bench.repairs.tritonbench_g.bmm_chunk_bwd.Kernels
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealProgram
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.Control

namespace VeriTile.Bench.Repairs.BmmChunkBwd.FPDefault
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Default production FP32 branch; no dot/add reassociation or fusion. -/
def source (BM BN BCS : Nat) (p : RealProgram.Parameters) : ComputeKernel :=
  RealProgram.source .fp32 .fp32 Bool.false .fp32 .fp32 BM BN BCS p

def parts (BM BN BCS : Nat) (p : RealProgram.Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (source BM BN BCS p).surfaceBody "cs").getD
    ⟨[], .constNat 0, .constNat 0, .constNat 0, [], []⟩

variable {α : Type} [Inhabited α]

/-- Successful typed setup for arbitrary shapes, strides, pids, and FP algebra. -/
def setup (M : Algebra α) (BM BN BCS : Nat) (p : RealProgram.Parameters) (s : State α) :
    {t : State α // run (engine M) (parts BM BN BCS p).before s = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts, source, RealProgram.source, TypedSource.specialize,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    beq_self_eq_true, ↓reduceIte]
  simp (config := {maxSteps := 100000}) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop,
    State.setReg, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast]
  rfl

/-- The default dot primitive is an opaque total matrix operation. This is a
backend choice, not a law equating dot/add to dotAcc or to an exact Real sum. -/
structure DotPrimitive (α : Type) where
  plain : Option ComputeDType → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch++[m,k]) → Values α .real (batch++[k,n]) →
    Values α .real (batch++[m,n])
  typed : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch++[m,k]) → Values α right.toTileDType (batch++[k,n]) →
    Values α out.toTileDType (batch++[m,n])

def backend (M : Algebra α) (D : DotPrimitive α) : Algebra α :=
  { M with dot := fun p {_} {_} {_} {_} x y => some (D.plain p x y)
           dotTyped := fun p l r o opts {_} {_} {_} {_} x y => some (D.typed p l r o opts x y) }

def iteration (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (s : State α)
    (dp : Values α .ptr [BM,BCS]) (ap : Values α .ptr [BCS,BN])
    (rows : Values α .nat [BM]) (cols : Values α .nat [BN])
    (lane : Values α .nat [BCS]) (length index : Nat) (acc : Values α .real [BM,BN])
    (hdp : s.regs .ptr [BM,BCS] "dout_ptrs" = some dp)
    (hap : s.regs .ptr [BCS,BN] "a_ptrs" = some ap)
    (hr : s.regs .nat [BM] "offs_m" = some rows)
    (hn : s.regs .nat [BN] "offs_n" = some cols)
    (hk : s.regs .nat [BCS] "offs_cs" = some lane)
    (hl : s.regs .nat [] "chunk_size_limit" = some (fun _ => length))
    (hc : s.regs .nat [] "cs" = some (fun _ => index))
    (hz : s.regs .real [BM,BN] "acc" = some acc) :
    {t : State α // run (engine (backend M D)) (parts BM BN BCS p).body s = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts, source, RealProgram.source, TypedSource.specialize,
    ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
    beq_self_eq_true, ↓reduceIte]
  simp (config := {maxSteps := 100000}) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, backend, ComputeDType.eraseDType, numeric, integralFloorDiv,
    integralMod, numericLt, numericLe, numericEq, natLt, bop,
    hdp,hap,hr,hn,hk,hl,hc,hz,State.setReg,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast]
  rfl

def writes : List RegName := ["cs","dout","a","acc","dout_ptrs","a_ptrs"]

structure Live (M : Algebra α) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (initial s : State α) : Prop where
  fixed : ∀ dtype shape name, name ∉ writes →
    s.regs dtype shape name = (setup M BM BN BCS p initial).val.regs dtype shape name
  dout : ∃ dp, s.regs .ptr [BM,BCS] "dout_ptrs" = some dp
  a : ∃ ap, s.regs .ptr [BCS,BN] "a_ptrs" = some ap
  acc : ∃ acc, s.regs .real [BM,BN] "acc" = some acc
  mem : s.mem = initial.mem

theorem live_initial (M : Algebra α) (BM BN BCS : Nat) (p : RealProgram.Parameters) (s : State α) :
    Live M BM BN BCS p s (setup M BM BN BCS p s).val := by
  refine ⟨by intros; rfl, ?_, ?_, ?_, rfl⟩
  · exact ⟨_, by simp [setup]; rfl⟩
  · exact ⟨_, by simp [setup]; rfl⟩
  · exact ⟨_, by simp [setup]; rfl⟩

theorem live_iteration (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial s : State α) (index : Nat)
    (live : Live (backend M D) BM BN BCS p initial s) :
    ∃ t, run (engine (backend M D)) (parts BM BN BCS p).body
      (s.setReg "cs" .nat [] (fun _ => index)) = some t ∧
        Live (backend M D) BM BN BCS p initial t := by
  let entry := s.setReg "cs" .nat [] (fun _ => index)
  obtain ⟨dp,hdp⟩ := live.dout
  obtain ⟨ap,hap⟩ := live.a
  obtain ⟨acc,hz⟩ := live.acc
  obtain ⟨rows,hr⟩ : ∃ v, entry.regs .nat [BM] "offs_m" = some v := by
    have hh := live.fixed .nat [BM] "offs_m" (by decide)
    simp only [entry,State.setReg_ne_name (h := show "offs_m" ≠ "cs" from by decide),hh]
    exact ⟨_,by simp [setup]; rfl⟩
  obtain ⟨cols,hn⟩ : ∃ v, entry.regs .nat [BN] "offs_n" = some v := by
    have hh := live.fixed .nat [BN] "offs_n" (by decide)
    simp only [entry,State.setReg_ne_name (h := show "offs_n" ≠ "cs" from by decide),hh]
    exact ⟨_,by simp [setup]; rfl⟩
  obtain ⟨lane,hk⟩ : ∃ v, entry.regs .nat [BCS] "offs_cs" = some v := by
    have hh := live.fixed .nat [BCS] "offs_cs" (by decide)
    simp only [entry,State.setReg_ne_name (h := show "offs_cs" ≠ "cs" from by decide),hh]
    exact ⟨_,by simp [setup]; rfl⟩
  obtain ⟨length,hl⟩ : ∃ v, entry.regs .nat [] "chunk_size_limit" = some (fun _ => v) := by
    have hh := live.fixed .nat [] "chunk_size_limit" (by decide)
    simp only [entry,State.setReg_ne_name (h := show "chunk_size_limit" ≠ "cs" from by decide),hh]
    exact ⟨_,by simp [setup]; rfl⟩
  let stepResult := iteration M D BM BN BCS p entry dp ap rows cols lane length index acc
    (by simpa [entry] using hdp) (by simpa [entry] using hap) hr hn hk hl
    (by simp [entry]) (by simpa [entry] using hz)
  refine ⟨stepResult.val,stepResult.property,?_,?_,?_,?_,?_⟩
  · intro dtype shape name hn
    have names := hn
    simp only [writes,List.mem_cons,List.mem_singleton,not_or] at names
    rcases names with ⟨hcs,hd,ha,hz,hp,hq⟩
    simpa [stepResult,iteration,entry,State.setReg,hcs,hd,ha,hz,hp,hq] using
      live.fixed dtype shape name hn
  · exact ⟨_,by simp [stepResult,iteration]; rfl⟩
  · exact ⟨_,by simp [stepResult,iteration]; rfl⟩
  · exact ⟨_,by simp [stepResult,iteration]; rfl⟩
  · exact live.mem

/-- Complete dynamic loop succeeds for every symbolic source block count.
The only floating backend supplied here is the local dot operation. -/
theorem loop_success (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) (stop : Nat) :
    ∃ t, range (engine (backend M D)) "cs" 0 stop 1 (parts BM BN BCS p).body
      (setup (backend M D) BM BN BCS p initial).val = some t ∧
        Live (backend M D) BM BN BCS p initial t := by
  obtain ⟨last,t,he,_,hl⟩ := FP.Structural.range_invariant
    (engine (backend M D)) (idx := "cs") (stop := stop) (stride := 1)
    (body := (parts BM BN BCS p).body)
    (P := fun _ s => Live (backend M D) BM BN BCS p initial s) (by decide)
    (fun i s _ h => live_iteration M D BM BN BCS p initial s i h)
    0 (setup (backend M D) BM BN BCS p initial).val (live_initial _ _ _ _ _ _)
  exact ⟨t,he,hl⟩

theorem finish_success (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial s : State α)
    (live : Live (backend M D) BM BN BCS p initial s) :
    ∃ t : State α, run (engine (backend M D)) (parts BM BN BCS p).after s = some t := by
  obtain ⟨acc,hz⟩ := live.acc
  have hb := live.fixed .nat [] "pid_b" (by decide)
  have hc := live.fixed .nat [] "pid_c" (by decide)
  have hh := live.fixed .nat [] "pid_h" (by decide)
  have hm := live.fixed .nat [] "pid_m" (by decide)
  have hn := live.fixed .nat [] "pid_n" (by decide)
  have hl := live.fixed .nat [] "chunk_size_limit" (by decide)
  cases hr : p.hasResidual
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [parts, source, RealProgram.source, TypedSource.specialize,
      ComputeKernel.surfaceBody, Source.dynamicRange?, Option.map, Option.getD,
      beq_self_eq_true, ↓reduceIte]
    simp (config := {maxSteps := 100000}) [run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, backend, ComputeDType.eraseDType, numeric, integralFloorDiv,
      integralMod, numericLt, numericLe, numericEq, natLt, bop,
      hb,hc,hh,hm,hn,hl,hz,hr,setup,State.setReg,Cell.read,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,FP.Structural.store]
    rfl

theorem source_split (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    (source BM BN BCS p).surfaceBody = (parts BM BN BCS p).before ++
      [.forRangeDyn "cs" (parts BM BN BCS p).start (parts BM BN BCS p).stop
        (parts BM BN BCS p).stride (parts BM BN BCS p).body] ++ (parts BM BN BCS p).after := by
  rfl

/-- Successful complete source execution, including optional residual, output
cast and masked store. Symbolic shapes/strides/pids and arbitrary local floating
primitives are retained. FPOutput and FPGridOutput supply the independent output theorem. -/
theorem source_success (M : Algebra α) (D : DotPrimitive α) (BM BN BCS : Nat)
    (p : RealProgram.Parameters) (initial : State α) :
    ∃ t, FP.Structural.exec (engine (backend M D)) (source BM BN BCS p) initial = some t := by
  let remain := p.seqlen-initial.pids 2/p.groups*p.chunk
  let len := if p.chunk < remain then p.chunk else remain
  let stop := (len+BCS-1)/BCS
  obtain ⟨mid,hm,live⟩ := loop_success M D BM BN BCS p initial stop
  have loopRun : step (engine (backend M D))
      (.forRangeDyn "cs" (parts BM BN BCS p).start (parts BM BN BCS p).stop
        (parts BM BN BCS p).stride (parts BM BN BCS p).body)
      (setup (backend M D) BM BN BCS p initial).val = some mid := by
    have hb : evalOp (engine (backend M D)) none (parts BM BN BCS p).stop
        (setup (backend M D) BM BN BCS p initial).val = some (fun _ => stop) := by
      simp [parts,source,RealProgram.source,TypedSource.specialize,ComputeKernel.surfaceBody,
        Source.dynamicRange?,evalOp_unfold,setup,numeric,integralFloorDiv,bop,stop,len,remain]
      rfl
    simp only [step]
    rw [hb]
    change range (engine (backend M D)) "cs" 0 stop 1 (parts BM BN BCS p).body
      (setup (backend M D) BM BN BCS p initial).val = some mid
    exact hm
  obtain ⟨final,hfinal⟩ := finish_success M D BM BN BCS p initial mid live
  refine ⟨final,?_⟩
  change run (engine (backend M D)) (source BM BN BCS p).surfaceBody initial = _
  rw [source_split, List.append_assoc, run_append, (setup (backend M D) BM BN BCS p initial).property]
  simp only [Option.bind_some,List.singleton_append,run,loopRun,bind,Option.bind_some]
  exact hfinal

#axiomsClean setup
#axiomsClean source_success
#axiomsClean loop_success
#axiomsClean iteration
end
end VeriTile.Bench.Repairs.BmmChunkBwd.FPDefault
