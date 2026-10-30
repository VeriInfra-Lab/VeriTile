import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeAccess
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.NativeLoop

namespace VeriTile.Bench.Repairs.BmmChunkFwd.NativeProgram
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open SourcePointers NativePointers
noncomputable section

/-- The source program executes through the same setup and loop-terminal
pointer states used by the finite proof. Causal-inactive blocks do not enter
these phases. The finite loop executes the actual source K body. -/
def Block {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call) (c : Host.Config)
    (p : Geometry.PID h c) (s : State α) : Prop :=
  ∃ entry, FP.Structural.run (engine o.algebra) (FPSuccess.branch h c).before (FPGrid.atIndex s p)=some entry ∧
    Controls (coordinates h c p) entry ∧
    evalOp (engine o.algebra) none (FPSuccess.branch h c).condition entry=
      some (fun _ => NativeControl.active h c p) ∧
    (NativeControl.active h c p=Bool.true → ∃ prepared terminal final,
      FP.Structural.run (engine o.algebra) (FPSuccess.parts h c).before entry=some prepared ∧
      StateAt h c (coordinates h c p) 0 prepared ∧
      range (engine o.algebra) "k" 0 (Host.cdiv h.shape.K c.K) 1 (FPSuccess.parts h c).body prepared=some terminal ∧
      NativeLoop.run (engine o.algebra) (FPSuccess.parts h c).body (Host.cdiv h.shape.K c.K)
        0 (NativeControl.ceil h.shape.K c.K) prepared=some terminal ∧
      StateAt h c (coordinates h c p) (Host.cdiv h.shape.K c.K) terminal ∧
      FP.Structural.run (engine o.algebra) (FPSuccess.parts h c).after terminal=some final ∧
      FP.Structural.exec (engine o.algebra) (Kernels.source h c) (FPGrid.atIndex s p)=some final)

theorem block {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call) (c : Host.Config)
    (hc : c∈Host.configurations) (metadata : NativeControl.Metadata h c)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) (p : Geometry.PID h c) (s : State α) :
    Block o h c p s := by
  obtain ⟨entry,hpre,hcontrols,hguard⟩ := NativeControl.source_prologue o h c hc metadata p s
  have controls : Controls (coordinates h c p) entry := by
    obtain ⟨vb,vc,vh,vm,vn,_⟩ := NativeControl.control_values h c hc metadata p
    simpa only [Controls,coordinates,NativeControl.Controls,vb,vc,vh,vm,vn] using hcontrols
  refine ⟨entry,hpre,controls,hguard,?_⟩
  intro active
  obtain ⟨prepared,hsetup,hstate,_⟩ := SourcePointers.setup o h c _ entry controls ha hb
  obtain ⟨terminal,hrange,hnative,hterminal,_⟩ :=
    NativeLoop.source_iterations o h c hc metadata.sizes _ prepared hstate ha hb
  obtain ⟨final,hpost,_,_⟩ := SourceAccess.epilogue o h c _ _ terminal hterminal ha hb
  refine ⟨prepared,terminal,final,hsetup,hstate,hrange,hnative,hterminal,hpost,?_⟩
  have hs : evalOp (engine o.algebra) none (FPSuccess.parts h c).start prepared=some (fun _ => (0 : Nat)) := by
    simp only [FPSuccess.parts,FPSuccess.branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
  have he : evalOp (engine o.algebra) none (FPSuccess.parts h c).stop prepared=some (fun _ => Host.cdiv h.shape.K c.K) := by
    simp only [FPSuccess.parts,FPSuccess.branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
  have hd : evalOp (engine o.algebra) none (FPSuccess.parts h c).stride prepared=some (fun _ => (1 : Nat)) := by
    simp only [FPSuccess.parts,FPSuccess.branch,Kernels.source,Host.Call.kernel,ha,hb,TypedSource.specialize]; rfl
  have hloop : step (engine o.algebra)
      (.forRangeDyn "k" (FPSuccess.parts h c).start (FPSuccess.parts h c).stop (FPSuccess.parts h c).stride (FPSuccess.parts h c).body)
      prepared=some terminal := by
    simpa only [step,hs,he,hd,Option.bind_some] using hrange
  change FP.Structural.run _ (Kernels.source h c).surfaceBody _=some final
  rw [FPSuccess.source_split,run_append,hpre]
  simp only [Option.bind_some,FP.Structural.run,step,hguard,active,↓reduceIte]
  rw [FPSuccess.branch_split,run_append,run_append,hsetup]
  simp only [Option.bind_some,FP.Structural.run,hloop]
  simp [Option.bind,hpost]

/-- Complete native correspondence, quantified over all blocks and every
possible reached memory/register state. Consequently it applies unchanged to
all configurations in a tuning trace and the chosen invocation. -/
structure Certificate {α : Type} [Inhabited α] (o : FPSuccess.Ops α)
    (h : Host.Call) (c : Host.Config) (layout : Layout h) : Prop where
  indices : Finite.SourceIndices h c
  blocks : ∀ p s, Block o h c p s
  control : ∀ q, q≤Host.cdiv h.shape.K c.K →
    (BitVec.ofNat 64 q).slt (NativeControl.ceil h.shape.K c.K)=decide (q<Host.cdiv h.shape.K c.K) ∧
    (BitVec.ofNat 64 q).toInt=q ∧
    (q<Host.cdiv h.shape.K c.K → (BitVec.ofNat 64 q+1).toInt=q+1)
  aPointers : ∀ p q i e, aWord h c (coordinates h c p) q i e=BitVec.ofNat 64 (aOffset h c (coordinates h c p) q i e)
  bPointers : ∀ p q e j, bWord h c (coordinates h c p) q e j=BitVec.ofNat 64 (bOffset h c (coordinates h c p) q e j)
  loads : ∀ p q s, StateAt h c (coordinates h c p) q s →
    ∃ t, FP.Structural.run (engine o.algebra) (FPSuccess.parts h c).body (s.setReg "k" .nat [] (fun _ => q))=some t ∧
      t.regs .real [c.M,c.K] "a"=some (fun i => SourceAccess.aValue o h c (coordinates h c p) q s i.1 i.2.1) ∧
      t.regs .real [c.K,c.N] "b"=some (fun i => SourceAccess.bValue o h c (coordinates h c p) q s i.1 i.2.1)
  seqOut : ∀ p q s, StateAt h c (coordinates h c p) q s →
    ∃ t, FP.Structural.run (engine o.algebra) (FPSuccess.parts h c).after s=some t ∧
      t.regs .ptr [c.M,c.N] "out_ptrs"=some (fun i => (h.Out,outOffset h c (coordinates h c p) i.1 i.2.1)) ∧
      (h.hasSeq=Bool.true →
        t.regs .int [c.M] "seq_idx_m"=some (fun i => SourceAccess.seqValue h (coordinates h c p) s (row c (coordinates h c p) i.1) (-1)) ∧
        t.regs .int [c.N] "seq_idx_n"=some (fun i => SourceAccess.seqValue h (coordinates h c p) s (col c (coordinates h c p) i.1) (-2)))
  masks : ∀ p q, q<Host.cdiv h.shape.K c.K → ∀ (i : Fin c.M) (j : Fin c.N) (e : Fin c.K),
    NativeAccess.aMask h c p q i e=SourceAccess.aMask h c (coordinates h c p) q i e ∧
    NativeAccess.bMask h c p q e j=SourceAccess.bMask h c (coordinates h c p) q e j ∧
    NativeAccess.outMask h c p i j=SourceAccess.outMask h c (coordinates h c p) i j
  aBytes : ∀ p q i e, SourceAccess.aMask h c (coordinates h c p) q i e=Bool.true →
    Address layout.a (aWord h c (coordinates h c p) q i e) (aOffset h c (coordinates h c p) q i e)
  bBytes : ∀ p q e j, SourceAccess.bMask h c (coordinates h c p) q e j=Bool.true →
    Address layout.b (bWord h c (coordinates h c p) q e j) (bOffset h c (coordinates h c p) q e j)
  seqBytes : ∀ p lane, h.hasSeq=Bool.true → lane<limit h (coordinates h c p) →
    Address layout.seq (seqWord h (coordinates h c p) lane) (seqOffset h (coordinates h c p) lane)
  outBytes : ∀ p i j, SourceAccess.outMask h c (coordinates h c p) i j=Bool.true →
    Address layout.out (outWord h c (coordinates h c p) i j) (outOffset h c (coordinates h c p) i j)
  storeFrame : ∀ (p : Geometry.PID h c) (s : State α),
    ∃ final, FP.Structural.exec (engine o.algebra) (Kernels.source h c) (FPGrid.atIndex s p)=some final ∧
      (∀ rg off, ¬FPSuccess.ProgramWrites h c (FPGrid.atIndex s p) rg off → final.mem rg off=s.mem rg off) ∧
      (FPSuccess.active h c (FPGrid.atIndex s p)=Bool.false → final.mem=s.mem)

  nativeControls : ∀ p, NativeControl.b h c p=BitVec.ofNat 64 (coordinates h c p).batch ∧
    NativeControl.ch h c p=BitVec.ofNat 64 (coordinates h c p).chunk ∧
    NativeControl.head h c p=BitVec.ofNat 64 (coordinates h c p).head ∧
    NativeControl.m h c p=BitVec.ofNat 64 (coordinates h c p).m ∧
    NativeControl.n h c p=BitVec.ofNat 64 (coordinates h c p).n ∧
    NativeControl.ceil h.shape.K c.K=BitVec.ofNat 64 (Host.cdiv h.shape.K c.K)
  steps : ∀ p q s, StateAt h c (coordinates h c p) q s →
    ∃ t, FP.Structural.run (engine o.algebra) (FPSuccess.parts h c).body
      (s.setReg "k" .nat [] (fun _ => q))=some t ∧ StateAt h c (coordinates h c p) (q+1) t ∧ t.mem=s.mem
  outMaskExact : ∀ p (i : Fin c.M) (j : Fin c.N),
    NativeAccess.outMask h c p i j=SourceAccess.outMask h c (coordinates h c p) i j
  seqMaskExact : ∀ p (i : Fin c.M) (j : Fin c.N),
    Finite.mask (coordinates h c p).m c.M i (limit h (coordinates h c p))=
      decide (row c (coordinates h c p) i<limit h (coordinates h c p)) ∧
    Finite.mask (coordinates h c p).n c.N j (limit h (coordinates h c p))=
      decide (col c (coordinates h c p) j<limit h (coordinates h c p))
  seqPointer : ∀ p lane, seqWord h (coordinates h c p) lane=BitVec.ofNat 64 (seqOffset h (coordinates h c p) lane)
  outPointer : ∀ p i j, outWord h c (coordinates h c p) i j=BitVec.ofNat 64 (outOffset h c (coordinates h c p) i j)
  stores : ∀ p q s, StateAt h c (coordinates h c p) q s →
    ∃ t, FP.Structural.run (engine o.algebra) (FPSuccess.parts h c).after s=some t ∧
      (∀ rg off, ¬SourceAccess.Writes h c (coordinates h c p) rg off → t.mem rg off=s.mem rg off)

theorem certify {α : Type} [Inhabited α] (o : FPSuccess.Ops α) (h : Host.Call) (c : Host.Config)
    (hc : c∈Host.configurations) (metadata : NativeControl.Metadata h c) (layout : Layout h)
    (ha : h.aStorage=.fp32) (hb : h.bStorage=.fp32) : Certificate o h c layout where
  indices := Finite.source_indices h c hc metadata.sizes
  blocks := block o h c hc metadata ha hb
  control := NativeLoop.control h c hc metadata.sizes
  aPointers := fun p q i e => a_formula h c (coordinates h c p) q i e
  bPointers := fun p q e j => b_formula h c (coordinates h c p) q e j
  loads := fun p q s hs => SourceAccess.loads o h c (coordinates h c p) q s hs ha hb
  seqOut := fun p q s hs => SourceAccess.epilogue o h c (coordinates h c p) q s hs ha hb
  masks := NativeAccess.masks h c hc metadata.sizes
  aBytes := NativeAccess.a_active h c hc metadata.sizes layout
  bBytes := NativeAccess.b_active h c hc metadata.sizes layout
  seqBytes := NativePointers.seq_active h c hc metadata.sizes layout
  outBytes := NativeAccess.out_active h c layout
  storeFrame := fun p s => FPSuccess.complete o h c (FPGrid.atIndex s p)
  nativeControls := NativeControl.controls h c hc metadata
  steps := fun p q s hs => SourcePointers.body o h c (coordinates h c p) q s hs ha hb
  outMaskExact := NativeAccess.output_mask h c hc metadata.sizes
  seqMaskExact := NativeAccess.seq_masks h c hc metadata.sizes
  seqPointer := fun p lane => seq_formula h (coordinates h c p) lane
  outPointer := fun p i j => out_formula h c (coordinates h c p) i j
  stores := fun p q s hs => SourceAccess.epilogue_frame o h c (coordinates h c p) q s hs ha hb

#axiomsClean block
#axiomsClean certify
end
end VeriTile.Bench.Repairs.BmmChunkFwd.NativeProgram

#guard_msgs (drop info) in
#auditModuleAxioms
