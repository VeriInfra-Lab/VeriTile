import bench.tritonbench_g.chunk_retention.Sources.ForwardOSource

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelModel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def featureLoad (M : Algebra α) (c : Shape) (s : State α) (r : RegionName)
    (f : Nat) (i : TileIndex [64,c.BK]) : α :=
  M.fp32Load (if row s i.1.val<c.T ∧ feature c f i.2.1.val<c.K then
    (s.mem r (s.pids 2*(c.T*c.K)+row s i.1.val*c.K+feature c f i.2.1.val)).read .real else zero M)
def keyLoad (M : Algebra α) (c : Shape) (s : State α) (f : Nat)
    (i : TileIndex [c.BK,64]) : α :=
  M.fp32Load (if feature c f i.1.val<c.K ∧ row s i.2.1.val<c.T then
    (s.mem "k" (s.pids 2*(c.T*c.K)+feature c f i.1.val+row s i.2.1.val*c.K)).read .real else zero M)
def queryGate (M : Algebra α) (c : Shape) (s : State α) (f : Nat)
    (i : TileIndex [64,c.BK]) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul (featureLoad M c s "q" f i) (rowDecay M c s ⟨i.1,()⟩))
def historyLoad (M : Algebra α) (c : Shape) (s : State α) (f : Nat)
    (i : TileIndex [c.BK,c.BV]) : α :=
  M.fp32Load (if feature c f i.1.val<c.K ∧ column c s i.2.1.val<c.V then
    (s.mem "h" (s.pids 2*(c.chunks*c.K*c.V)+s.pids 1*c.K*c.V+
      feature c f i.1.val*c.V+column c s i.2.1.val)).read .real else zero M)

abbrev Accum (c : Shape) (α : Type) := (Index c → α) × (ScoreIndex → α)
def update (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α)
    (f : Nat) (a : Accum c α) : Accum c α :=
  ⟨fun i => M.binary (some .fp32) .real .add (a.1 i)
      (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (queryGate M c s f) (historyLoad M c s f) i),
   fun i => M.binary (some .fp32) .real .add (a.2 i)
      (D.apply (some .fp32) .real .real .real {allowTF32:=some Bool.false} (batch:=[])
        (featureLoad M c s "q" f) (keyLoad M c s f) i)⟩
def accumulated (M : Algebra α) (D : DotPrimitive α) (c : Shape) (s : State α) : Accum c α :=
  foldRange (update M D c s) 0 c.keyBlocks 1 ⟨fun _ => zero M,fun _ => zero M⟩

theorem bodyContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Accum c α)
    (ctx : Context M c initial current) (ha : HasAcc c a.1 a.2 current) :
    let t := (bodyExecution M D c scale initial current f a.1 a.2 ctx ha).val
    Context M c initial t ∧
      HasAcc c (update M D c initial f a).1 (update M D c initial f a).2 t := by
  dsimp only
  constructor
  · constructor <;> simp [bodyExecution,ctx.memory,ctx.iv,ctx.it,ctx.bh,KernelModel.signed_head_mod,Int.toNat_mul,Int.toNat_add,Int.toNat_natCast,ctx.di,ctx.ds]
  · simp [HasAcc,bodyExecution,update,queryGate,featureLoad,keyLoad,historyLoad,feature,row,column,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    constructor <;> rfl

theorem loopExecution (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (ctx : Context M c initial current)
    (ha : HasAcc c (fun _ => zero M) (fun _ => zero M) current) :
    ∃ t, range (engine (withDot M D)) "i_k" 0 c.keyBlocks 1 (parts c scale).body current=some t ∧
      Context M c initial t ∧
        HasAcc c (accumulated M D c initial).1 (accumulated M D c initial).2 t := by
  let I := fun (_ : Nat) (a : Accum c α) (s : State α) => Context M c initial s ∧ HasAcc c a.1 a.2 s
  exact range_values (engine (withDot M D)) "i_k" (parts c scale).body (update M D c initial)
    I (fun a s => Context M c initial s ∧ HasAcc c a.1 a.2 s) 0 c.keyBlocks 1 _ current
    (by
      intro k a s h hk
      let t := bodyExecution M D c scale initial s k a.1 a.2 h.1 h.2
      exact ⟨t.val,t.property,bodyContext M D c scale initial s k a h.1 h.2⟩)
    (by intro n a s h; exact h) ⟨ctx,ha⟩

#specNonCircular accumulated avoiding [run,FP.Structural.exec]
#axiomsClean bodyContext
#axiomsClean loopExecution
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.ForwardOSource
