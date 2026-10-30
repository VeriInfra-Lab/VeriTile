import bench.repairs.tritonbench_g.adam_update_triton.Proofs.Tile

namespace VeriTile.Bench.Repairs.AdamUpdateTriton.CommutedTile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SwigluBackward.Whole (writeCell scatter_mem fold_written fold_frame)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

open Tile (Semantics)

def programState {α : Type} [Inhabited α] (params : Scalars α) (pid : Nat)
    (s : State α) : State α :=
  bindArguments params {s with pids := fun ax => if ax=0 then pid else 0}

/-- Constructive execution of all 19 source statements with the actual four
runtime scalar registers. No numerical equality is needed for progress. -/
def sourceRun {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (total chunk block pid : Nat) (s : State α) :
    {t : State α // run sem.backend (commutedSource total chunk block).surfaceBody
      (programState params pid s)=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp (maxSteps := 1400000) [commutedSource,commutedWithArguments,commutedBody,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
    store,Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,numericEq,
    programState,bindArguments,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    Semantics.backend,engine,Algebra.withDefaultPrecision,resolvePrecision]
  rfl

def outputRegion : Output → RegionName
  | .parameter => "P" | .momentum => "M"

/-- Source expression order, including the comparison and bool-to-FP32
multiplication. Floating negation here is a source node, not a sign-XOR law. -/
def scalar (sem : Semantics α) (params : Scalars α) (output : Output) (p g m : α) : α :=
  let p := sem.algebra.fp32Load p
  let g := sem.algebra.fp32Load g
  let m := sem.algebra.fp32Load m
  let bin := sem.algebra.binary (some .fp32) .real
  let lit := sem.algebra.literal (some .fp32) .real
  let diff := bin .sub m g
  let update := bin .add (bin .mul diff params.beta1) g
  match output with
  | .parameter =>
    bin .add (bin .mul p (bin .sub (lit 1) (bin .mul params.lr params.wd)))
      (bin .mul (if sem.less (lit 0) update then
        sem.algebra.unary (some .fp32) (.neg .real) params.lr else params.lr)
        (if sem.equal update (lit 0)=Bool.false then lit 1 else lit 0))
  | .momentum => bin .add g (bin .mul diff params.beta2)

def raw {α : Type} [Inhabited α] (chunk block pid : Nat)
    (s : State α) (region : RegionName) (i : TileIndex [block]) : α :=
  if pid*block+i.1.val<chunk then (s.mem region (pid*block+i.1.val)).read .real
  else s.undef .real region (pid*block+i.1.val)

def scatter {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (chunk block pid : Nat) (s : State α) (output : Output) (mem : RegionName → Nat → Cell α) :=
  (TileShape.allIndices [block]).foldl (fun mem i =>
    if pid*block+i.1.val<chunk then
      writeCell mem (outputRegion output) (pid*block+i.1.val)
        (.mk .real (scalar sem params output (raw chunk block pid s "P" i)
          (raw chunk block pid s "G" i) (raw chunk block pid s "M" i)))
    else mem) mem

def memory {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (chunk block pid : Nat) (s : State α) : RegionName → Nat → Cell α :=
  scatter sem params chunk block pid s .momentum
    (scatter sem params chunk block pid s .parameter s.mem)

theorem sourceRun_mem {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (total chunk block pid : Nat) (s : State α) :
    (sourceRun sem params total chunk block pid s).val.mem=memory sem params chunk block pid s := by
  simp only [sourceRun,scatter_mem]
  rfl

theorem scatter_readback {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (chunk block pid : Nat) (s : State α) (output : Output) (mem : RegionName → Nat → Cell α)
    (target : TileIndex [block]) (active : pid*block+target.1.val<chunk) :
    scatter sem params chunk block pid s output mem (outputRegion output) (pid*block+target.1.val)=
      .mk .real (scalar sem params output
        ((s.mem "P" (pid*block+target.1.val)).read .real)
        ((s.mem "G" (pid*block+target.1.val)).read .real)
        ((s.mem "M" (pid*block+target.1.val)).read .real)) := by
  have unique : ∀ i : TileIndex [block], pid*block+i.1.val<chunk →
      pid*block+i.1.val=pid*block+target.1.val → i=target := by
    intro i _ h
    apply Prod.ext
    · apply Fin.ext; omega
    · exact Subsingleton.elim _ _
  obtain ⟨before,after,hall⟩ := List.append_of_mem (TileShape.mem_allIndices [block] target)
  unfold scatter
  rw [hall,List.foldl_append,List.foldl_cons]
  simp only [active,if_true]
  suffices h : _=Cell.mk .real (scalar sem params output (raw chunk block pid s "P" target)
      (raw chunk block pid s "G" target) (raw chunk block pid s "M" target)) by
    simpa only [raw,active,if_true] using h
  apply fold_written _ _ _ _ target unique
  simp [writeCell,raw,active]

theorem scatter_frame {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (chunk block pid : Nat) (s : State α) (output : Output) (mem : RegionName → Nat → Cell α)
    (region : RegionName) (address : Nat)
    (outside : region≠outputRegion output ∨ ∀ i : TileIndex [block],
      pid*block+i.1.val<chunk → address≠pid*block+i.1.val) :
    scatter sem params chunk block pid s output mem region address=mem region address :=
  fold_frame _ _ _ _ _ _ _ _ outside

def Outside (chunk block pid : Nat) (region : RegionName) (address : Nat) : Prop :=
  ∀ output,region≠outputRegion output ∨ ∀ i : TileIndex [block],
    pid*block+i.1.val<chunk → address≠pid*block+i.1.val

theorem frame {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (total chunk block pid : Nat) (s : State α) (region : RegionName) (address : Nat)
    (outside : Outside chunk block pid region address) :
    (sourceRun sem params total chunk block pid s).val.mem region address=s.mem region address := by
  rw [sourceRun_mem]
  unfold memory
  rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .momentum)]
  exact scatter_frame _ _ _ _ _ _ _ _ _ _ (outside .parameter)

theorem readback {α : Type} [Inhabited α] (sem : Semantics α) (params : Scalars α)
    (total chunk block pid : Nat) (s : State α) (output : Output)
    (target : TileIndex [block]) (active : pid*block+target.1.val<chunk) :
    (sourceRun sem params total chunk block pid s).val.mem (outputRegion output) (pid*block+target.1.val)=
      .mk .real (scalar sem params output
        ((s.mem "P" (pid*block+target.1.val)).read .real)
        ((s.mem "G" (pid*block+target.1.val)).read .real)
        ((s.mem "M" (pid*block+target.1.val)).read .real)) := by
  rw [sourceRun_mem]
  cases output <;> unfold memory
  · rw [scatter_frame _ _ _ _ _ _ _ _ _ _ (Or.inl (by decide))]
    exact scatter_readback _ _ _ _ _ _ _ _ _ active
  · exact scatter_readback _ _ _ _ _ _ _ _ _ active

#axiomsClean readback
#axiomsClean frame
#axiomsClean sourceRun
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AdamUpdateTriton.CommutedTile
