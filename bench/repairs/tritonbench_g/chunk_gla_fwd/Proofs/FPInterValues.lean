import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.FPInter
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.FPInter
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange range_values)
open ProducerGeometry (Shape)
open FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def reference (M : Algebra α) (c : Shape) (s : State α) (f e : Nat) : α :=
  M.fp32Load (if feature c f e<c.K then
    (s.mem "g" (s.pids 2*(c.T*c.K)+rowStart c s*c.K+feature c f e)).read .real else zero M)
def queryLoad (M : Algebra α) (c : Shape) (s : State α) (r : RegionName)
    (f : Nat) (i : TileIndex [16,c.BK]) : α :=
  M.fp32Load (if row c s i.1.val<c.T ∧ feature c f i.2.1.val<c.K then
    (s.mem r (s.pids 2*(c.T*c.K)+row c s i.1.val*c.K+feature c f i.2.1.val)).read .real else zero M)
def keyLoad (M : Algebra α) (c : Shape) (s : State α) (r : RegionName)
    (f : Nat) (i : TileIndex [c.BK,16]) : α :=
  M.fp32Load (if feature c f i.1.val<c.K ∧ columnStart c s+i.2.1.val<c.T then
    (s.mem r (s.pids 2*(c.T*c.K)+feature c f i.1.val+(columnStart c s+i.2.1.val)*c.K)).read .real else zero M)
def queryGate (M : Algebra α) (c : Shape) (scale : ℝ) (s : State α)
    (f : Nat) (i : TileIndex [16,c.BK]) : α :=
  M.binary (some .fp32) .real .mul
    (M.binary (some .fp32) .real .mul (queryLoad M c s "q" f i)
      (M.unary (some .fp32) .exp
        (M.binary (some .fp32) .real .sub (queryLoad M c s "g" f i) (reference M c s f i.2.1.val))))
    (M.literal (some .fp32) .real scale)
def keyGate (M : Algebra α) (c : Shape) (s : State α) (f : Nat) (i : TileIndex [c.BK,16]) : α :=
  M.binary (some .fp32) .real .mul (keyLoad M c s "k" f i)
    (M.unary (some .fp32) .exp
      (M.binary (some .fp32) .real .sub (reference M c s f i.1.val) (keyLoad M c s "g" f i)))
def update (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α)
    (f : Nat) (a : Index → α) (i : Index) : α :=
  M.binary (some .fp32) .real .add (a i)
    (D.apply (some .fp32) .real .real .real {} (batch:=[]) (queryGate M c scale s f) (keyGate M c s f) i)
def accumulated (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) : Index → α :=
  foldRange (update M D c scale s) 0 (Geometry.ceilDiv c.K c.BK) 1 (fun _ => zero M)
def value (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ) (s : State α) (i : Index) : α :=
  M.cast (some .fp32) .real .fp32 (accumulated M D c scale s i)

theorem bodyContext (M : Algebra α) (D : DotPrimitive α) (c : Shape) (scale : ℝ)
    (initial current : State α) (f : Nat) (a : Index → α)
    (ctx : Context c initial current) (ha : HasAcc a current) :
    let t := (bodyExecution M D c scale initial current f a ctx ha).val
    Context c initial t ∧ HasAcc (update M D c scale initial f a) t := by
  dsimp only
  constructor
  · constructor <;> simp [bodyExecution,ctx.memory,ctx.it,ctx.ic,ctx.bh,ctx.ii,ctx.ij]
  · simp [HasAcc,bodyExecution,update,queryGate,keyGate,reference,queryLoad,keyLoad,feature,row,rowStart,columnStart,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    rfl

#specNonCircular value avoiding [run,FP.Structural.exec]
#axiomsClean bodyContext
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.FPInter
