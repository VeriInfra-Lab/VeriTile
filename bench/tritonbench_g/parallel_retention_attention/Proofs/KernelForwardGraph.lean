/- Reuses the historical complete retention execution proof, adapted only
to this repair's explicit zero-padded source and task-local namespace. -/
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelSources
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPForwardGraph
namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot ForwardConfig queryChunk keyChunk valueChunk foldRange)
noncomputable section

/-- The shared launch record's Z region is unused by retention. -/
def forwardKernel (c : ForwardConfig) : ComputeKernel :=
  forwardSource c.q c.k c.v c.o c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d
    c.scale c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV

/-- All decay arithmetic is evaluated by precision-selected finite-value
primitives, including both transcendental operations and integer conversion.
No real exponential or logarithm interpretation occurs here. -/
def logDecay {α : Type} (M : Algebra α) (head : Nat) : α :=
  M.unary (some .fp32) .log2
    (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real 1.0)
      (M.unary (some .fp32) .exp2
        (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real (-5.0))
          (M.binary (some .fp32) .real .mul (M.fromNat (some .fp32) head)
            (M.literal (some .fp32) .real 1.0)))))

def rowDecay {α : Type} (M : Algebra α) (b : α) (n : Nat) : α :=
  M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul (M.fromNat (some .fp32) n) b)

def blockDecay {α : Type} (M : Algebra α) (b : α) (BTS : Nat) : α :=
  M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul b (M.fromNat (some .fp32) BTS))

/-- Independent O recurrence, with the different off-diagonal and diagonal
decays preserved. Signed subtraction is retained even for masked-out lanes. -/
def forwardUpdate {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) (diagonal : Bool) (start : Nat)
    (previous : TileIndex [c.BTL,c.BV] → α) : TileIndex [c.BTL,c.BV] → α :=
  let b := logDecay M (s.pids 2 % c.H)
  let key := matrixRead M s c.k (s.pids 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
    (keyChunk c s) start c.BK c.BTS
  let value := matrixRead M s c.v (s.pids 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
    start (valueChunk c s) c.BTS c.BV
  let raw := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := []) (forwardQuery M c s) key
  let decay := fun (i : TileIndex [c.BTL,c.BTS]) =>
    if diagonal then
      if start-queryChunk c s+i.2.1.val ≤ i.1.val then
        M.unary (some .fp32) .exp2 (M.binary (some .fp32) .real .mul
          (M.fromInt (some .fp32) ((i.1.val : Int) - (start-queryChunk c s+i.2.1.val : Nat))) b)
      else M.literal (some .fp32) .real 0.0
    else rowDecay M b (c.BTS-i.2.1.val)
  let score := fun i => M.binary (some .fp32) .real .mul (raw i) (decay i)
  let contribution := D.apply (some .fp32) {allowTF32 := some Bool.false} (batch := [])
    (fun i => M.cast (some .fp32) .real .real (score i)) value
  fun i => M.binary (some .fp32) .real .add
    (if diagonal then previous i else M.binary (some .fp32) .real .mul (previous i) (blockDecay M b c.BTS))
    (contribution i)

def betweenValues {α : Type} (M : Algebra α) (c : ForwardConfig) (s : State α)
    (previous : TileIndex [c.BTL,c.BV] → α) : TileIndex [c.BTL,c.BV] → α :=
  fun i => M.binary (some .fp32) .real .mul (previous i)
    (rowDecay M (logDecay M (s.pids 2%c.H)) i.1.val)

def forwardValues {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) : TileIndex [c.BTL,c.BV] → α :=
  let initial := fun _ => M.literal (some .fp32) .real 0
  let before := foldRange (forwardUpdate M D c s Bool.false) 0 (queryChunk c s) c.BTS initial
  foldRange (forwardUpdate M D c s Bool.true) (queryChunk c s) ((s.pids 1+1)*c.BTL) c.BTS
    (betweenValues M c s before)

def outputMemory {α : Type} [Inhabited α] (M : Algebra α) (c : ForwardConfig)
    (s : State α) (values : TileIndex [c.BTL,c.BV] → α) : State α :=
  (TileShape.allIndices [c.BTL,c.BV]).foldl (fun t i =>
    let row := queryChunk c s+i.1.val
    let col := valueChunk c s+i.2.1.val
    if row<c.T ∧ col<c.V then t.write c.o
      ((s.pids 2+c.B*c.H*(s.pids 0/((c.V+c.BV-1)/c.BV)))*c.s_vo_h + row*c.s_vo_t + col*c.s_vo_d)
      (.mk .real (M.cast (some .fp32) .real .real (values i))) else t) s

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
