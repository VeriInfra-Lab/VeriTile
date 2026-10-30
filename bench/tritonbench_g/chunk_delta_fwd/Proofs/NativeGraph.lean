import bench.tritonbench_g.chunk_delta_fwd.Proofs.HostCall
import bench.tritonbench_g.chunk_gla_fwd.Proofs.NativeStaging
import bench.tritonbench_g.chunk_gla_fwd.Proofs.NativeComposition
namespace VeriTile.Bench.Repairs.ChunkDeltaFwd.Native
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr lowerDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

/-- Task-specific default compiler fusion. A one-step cumsum has the exact
shape H + (0 + dot(K,Vnew)); actual TTGIR seeds that dot with H. Multiple
inner steps retain their final outer add. Loads and casts materialize values. -/
def lower : Expr → Graph
  | .bits b => .bits b
  | .literal p d x => .literal p d x
  | .binary p d .add acc (.binary _ _ .add (.literal zp zd z)
      (.pendingDot _ _ _ _ o _ k _ row col a b)) =>
      if z=0 then lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j)) (lower acc)
      else .binary p d .add (lower acc)
        (lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j)) (.literal zp zd z))
  | .binary _ _ .add acc (.pendingDot _ _ _ _ o _ k _ row col a b) =>
      lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j)) (lower acc)
  | .binary p d op a b => .binary p d op (lower a) (lower b)
  | .unary p op a => .unary p op (lower a)
  | .cast p s t a => .cast p s t (lower a)
  | .load a => .move .register (lower a)
  | .move route a => .move route (lower a)
  | .pendingDot _ _ _ _ o _ k _ row col a b =>
      lowerDot o k row col (fun j => lower (a j)) (fun j => lower (b j))
        VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.zero

abbrev algebra := VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.algebra ⟨0,by decide⟩
abbrev dot := VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.dot
def inputs (bits : RegionName → Nat → BitVec 32) : Memory Expr :=
  fun r o => .mk .real (.bits (bits r o))
def historyGraph (p : Parameters) (h : Geometry.ShapeValid p) (bits : RegionName → Nat → BitVec 32)
    (head c key value : Nat) (hk : key < p.K) : Graph :=
  lower (historyValue algebra dot p h (inputs bits) head c key value hk)
def correctedGraph (p : Parameters) (h : Geometry.ShapeValid p) (bits : RegionName → Nat → BitVec 32)
    (head t value : Nat) : Graph := lower (correctedValue algebra dot p h (inputs bits) head t value)

theorem single_step_fusion (p q : Option ComputeDType) (d e : FloatDType)
    (m k n row col : Nat) (a b : Fin k → Expr) (seed : Expr) :
    lower (.binary p d .add seed (.binary q e .add (.literal (some .fp32) .real 0)
      (.pendingDot (some .fp32) .real .real .real {allowTF32:=some Bool.false} m k n row col a b)))=
      ieeeDot k (fun j => lower (a j)) (fun j => lower (b j)) (lower seed) := by
  simp [lower,lowerDot,VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging.defaultTF32]

theorem correction_dot (m k n row col : Nat) (a b : Fin k → Expr) :
    lower (.pendingDot (some .fp32) .real .real .real {allowTF32:=some Bool.false} m k n row col a b)=
      ieeeDot k (fun j => lower (a j)) (fun j => lower (b j))
        VeriTile.Bench.Repairs.ChunkGlaFwd.NativeInstructions.zero := rfl

/-- The only numerical boundary is scalar/fixed-ISA interpretation. Every
variable-width dot has already expanded to one FMA per ordered feature. -/
theorem history_transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (ports : LocalTransport f A B) (p : Parameters) (h : Geometry.ShapeValid p)
    (bits : RegionName → Nat → BitVec 32) (head c key value : Nat) (hk : key < p.K) :
    f ((historyGraph p h bits head c key value hk).eval A)=
      (historyGraph p h bits head c key value hk).eval B := transport f A B ports _
theorem corrected_transport {α β : Type} (f : α → β) (A : ISA α) (B : ISA β)
    (ports : LocalTransport f A B) (p : Parameters) (h : Geometry.ShapeValid p)
    (bits : RegionName → Nat → BitVec 32) (head t value : Nat) :
    f ((correctedGraph p h bits head t value).eval A)=(correctedGraph p h bits head t value).eval B :=
  transport f A B ports _

#axiomsClean single_step_fusion
#axiomsClean history_transport
#axiomsClean corrected_transport
#specNonCircular historyGraph avoiding [run,FP.Structural.exec,runGrid]
end
end VeriTile.Bench.Repairs.ChunkDeltaFwd.Native
