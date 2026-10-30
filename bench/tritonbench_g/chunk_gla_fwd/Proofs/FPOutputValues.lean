import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPOutput
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def featureLoad (M : Algebra α) (c : OutputGrid.Shape) (s : State α) (r : RegionName)
    (f : Nat) (i : TileIndex [c.BT,c.BK]) : α :=
  M.fp32Load (if row c s i.1.val<c.T ∧ feature c f i.2.1.val<c.K then
    (s.mem r (s.pids 2*(c.T*c.K)+row c s i.1.val*c.K+feature c f i.2.1.val)).read .real else zero M)

def queryGate (M : Algebra α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α)
    (f : Nat) (i : TileIndex [c.BT,c.BK]) : α :=
  M.cast (some .fp32) .real .real
    (M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) .real .real
        (M.binary (some .fp32) .real .mul (featureLoad M c s "q" f i) (M.literal (some .fp32) .real scale)))
      (M.unary (some .fp32) .exp (featureLoad M c s "g" f i)))

def historyLoad (M : Algebra α) (c : OutputGrid.Shape) (s : State α)
    (f : Nat) (i : TileIndex [c.BK,c.BV]) : α :=
  M.cast (some .fp32) .fp32 .real
    (if feature c f i.1.val<c.K ∧ column c s i.2.1.val<c.V then
      (s.mem "h" (s.pids 2*(Geometry.ceilDiv c.T c.BT*c.K*c.V)+s.pids 1*c.K*c.V+
        feature c f i.1.val*c.V+column c s i.2.1.val)).read .fp32
    else M.literal (some .fp32) .fp32 0)

def update (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α)
    (f : Nat) (a : Index c → α) (i : Index c) : α :=
  M.binary (some .fp32) .real .add (a i)
    (D.apply (some .fp32) .real .real .real {} (batch:=[]) (queryGate M c scale s f) (historyLoad M c s f) i)
def accumulated (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ) (s : State α) : Index c → α :=
  foldRange (update M D c scale s) 0 (Geometry.ceilDiv c.K c.BK) 1 (fun _ => zero M)

theorem bodyContext (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Index c → α)
    (ctx : Context c initial current) (ha : HasAcc c a current) :
    let t := (bodyExecution M D c scale initial current f a ctx ha).val
    Context c initial t ∧ HasAcc c (update M D c scale initial f a) t := by
  dsimp only
  constructor
  · constructor <;> simp [bodyExecution,ctx.memory,ctx.iv,ctx.it,ctx.bh,ctx.causal]
  · simp [HasAcc,bodyExecution,update,queryGate,featureLoad,historyLoad,feature,row,column,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    rfl

#specNonCircular accumulated avoiding [run,FP.Structural.exec]
#axiomsClean bodyContext
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput
