import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.Whole002.Kernels
import bench.tritonbench_g.parallel_attention.Proofs.FPRange
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
structure Config where
  storage : TypedSource.Storage
  q : RegionName
  k : RegionName
  v : RegionName
  out : RegionName
  seq : Region .nat
  blocks : Region .nat
  offsets : Region .nat
  counts : Region .nat
  columns : Region .nat
  sqz : Nat
  sqh : Nat
  sqm : Nat
  sqk : Nat
  skz : Nat
  skh : Nat
  skn : Nat
  skk : Nat
  svz : Nat
  svh : Nat
  svn : Nat
  svk : Nat
  soz : Nat
  soh : Nat
  som : Nat
  sok : Nat
  Z : Nat
  H : Nat
  T : Nat
  NR : Nat
  NS : Nat
  NV : Nat
  BM : Nat
  BN : Nat
  BD : Nat
  scale : ℝ

def parsedKernel (c : Config) : ComputeKernel :=
  match c.storage with
  | .fp16=>TypedSource.FP16.source c.q c.k c.v c.seq c.scale c.blocks c.offsets c.counts c.columns c.out
    c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svn c.svk
    c.soz c.soh c.som c.sok c.Z c.H c.T c.NR c.NS c.NV c.BM c.BN c.BD
  | .bf16=>TypedSource.BF16.source c.q c.k c.v c.seq c.scale c.blocks c.offsets c.counts c.columns c.out
    c.sqz c.sqh c.sqm c.sqk c.skz c.skh c.skn c.skk c.svz c.svh c.svn c.svk
    c.soz c.soh c.som c.sok c.Z c.H c.T c.NR c.NS c.NV c.BM c.BN c.BD

/-- Typed dot availability only; its numerical result remains the exact
primitive call on independently specified fp16 operands. -/
structure Dot (α : Type) where
  equal : α → α → Bool
  apply : Option ComputeDType → DotOptions → {batch : TileShape} → {m k n : Nat} →
    (TileIndex (batch ++ [m,k]) → α) → (TileIndex (batch ++ [k,n]) → α) →
    (TileIndex (batch ++ [m,n]) → α)

def backend {α : Type} (M : Algebra α) (D : Dot α) : Algebra α :=
  { M with
    dotTyped := fun p l r o opts {_} {_} {_} {_} a b =>
      match l,r,o with
      | .fp16,.fp16,.fp32 => some (D.apply p opts a b)
      | .bf16,.bf16,.fp32 => some (D.apply p opts a b)
      | l,r,o => M.dotTyped p l r o opts a b
    compareEq := fun _ _ => some D.equal }

def parsedActiveBody (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := (parsedKernel c).surfaceBody
  match code[3]? with
  | some (ComputeStmt.ifThen _ body) => body
  | _ => []

/-- The actual parsed floating maximum, retaining dtype and precision.
No comparison/select replacement or finiteness premise is needed. -/
def maximumAssignment (BM BN : Nat) : ComputeStmt :=
  .assign .real [BM] "m_i_new" (.compute (.alg .fp32
    (.maximumFloat .real (.consSame .nil) (.ref .real [BM] "m_i")
      (.reduceMax (shape := [BM,BN]) ⟨1,by simp⟩ Bool.false (.ref .real [BM,BN] "qk")))))

def activeBody (c : Config) : List ComputeStmt := parsedActiveBody c

def kernel (c : Config) : ComputeKernel := parsedKernel c

/-- The executed kernel is the complete source parser result, without the
previous local maximum-node replacement. -/
theorem kernel_exact (c : Config) : kernel c = parsedKernel c := rfl

def blockBody (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := activeBody c
  match code[21]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

def columnBody (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := activeBody c
  match code[23]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

/-- Both original loops contain the actual floating maximum primitive. -/
theorem maximum_source_nodes (c : Config) :
    (blockBody c)[10]? = some (maximumAssignment c.BM c.BN) ∧
    (columnBody c)[8]? = some (maximumAssignment c.BM c.BN) := by
  cases hs : c.storage <;> constructor <;>
    simp [hs,blockBody,columnBody,activeBody,parsedActiveBody,parsedKernel,TypedSource.FP16.source,TypedSource.BF16.source,TypedSource.Storage.tile,TypedSource.Storage.float,
      ComputeKernel.surfaceBody,maximumAssignment]

#axiomsClean kernel_exact
#axiomsClean maximum_source_nodes

end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.Source
