import bench.repairs.tritonbench_g.chunk_gate_recurrence.Kernels
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Base
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealProgram
import VeriTile.Triton.Float.Control

/-! The two complete canonical source programs at their kernel boundary.
The optional seed uses its actual four strides. The unchanged backward source
keeps its first zero-carry update, full 64x16 reduction and separate DL postlude.
No host zeroing, host reduction, allocator action or return object occurs here. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelProgram
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 4000000

def sourceSHA256 := "dfbecac806662178eb8615196fd78193102675b8add6692c0af831d5e5e5dbc1"
inductive Kind where
  | forward | backward
  deriving DecidableEq
structure SeedStrides where
  batch : Nat
  head : Nat
  key : Nat
  value : Nat
structure Program (c : Shape) where
  head : Nat
  key : Nat
  value : Nat
  headInside : head<c.heads
  keyInside : key<c.KT
  valueInside : value<c.VT

def layout (c : Shape) : TritonBenchG.ChunkGateRecurrence.RealProgram.Layout :=
  ⟨c.N,c.K,c.V,64,16⟩
def launch (p : Program c) : TritonBenchG.ChunkGateRecurrence.RealProgram.Launch :=
  ⟨p.head,p.key,p.value⟩
def address (c : Shape) (p : Program c) (t e j : Nat) : Nat :=
  TritonBenchG.ChunkGateRecurrence.RealProgram.address (layout c) (launch p) t e j
def gateAddress (c : Shape) (p : Program c) (t : Nat) : Nat := p.head*c.N+t
def seedAddress (c : Shape) (p : Program c) (s : SeedStrides) (e j : Nat) : Nat :=
  (p.head/c.H)*s.batch+(p.head%c.H)*s.head+(p.key*64+e)*s.key+(p.value*16+j)*s.value
def dseedAddress (c : Shape) (p : Program c) (e j : Nat) : Nat :=
  p.head*c.K*c.V+p.key*c.V*64+e*c.V+p.value*16+j
def gradientAddress (c : Shape) (p : Program c) (t : Nat) : Nat :=
  p.head*c.N*c.KT*c.VT+p.key*c.VT+p.value+t*c.KT*c.VT

theorem address_next (c : Shape) (p : Program c) (t e j : Nat) :
    address c p (t+1) e j=address c p t e j+c.K*c.V :=
  TritonBenchG.ChunkGateRecurrence.RealProgram.address_succ _ _ _ _ _
theorem address_unique (c : Shape) (p : Program c) (t u : Nat)
    (e k : Fin 64) (j v : Fin 16)
    (same : address c p t e.val j.val=address c p u k.val v.val) :
    t=u ∧ e=k ∧ j=v := by
  apply TritonBenchG.ChunkGateRecurrence.RealProgram.noCollision (layout c) (launch p)
    (Nat.le_mul_of_pos_left _ c.kpos) (Nat.le_mul_of_pos_left _ c.vpos) t u e k j v same

def source (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) : ComputeKernel :=
  match kind with
  | .forward => Kernels.forward "S" "D" "O" "Seed" c.H c.N c.K c.V s.batch s.head s.key s.value seed
  | .backward => Kernels.backward "S" "D" "DI" "DG" "DL" "DS" c.H c.N c.K c.V 64 16
def parts (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) : VeriTile.Triton.Source.DynamicRange :=
  (VeriTile.Triton.Source.dynamicRange? (source kind c s seed).surfaceBody "_i").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 1,[],[]⟩
def before (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) := (parts kind c s seed).before
def body (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) := (parts kind c s seed).body
def after (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) := (parts kind c s seed).after
def loop (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) : ComputeStmt :=
  .forRangeDyn "_i" (.constNat 0) (.sub .nat .nil (.constNat c.N) (.constNat 1))
    (.constNat 1) (body kind c s seed)
theorem source_body (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) :
    (source kind c s seed).surfaceBody=before kind c s seed++[loop kind c s seed]++after kind c s seed := by
  cases kind <;> rfl

variable {α : Type} [Inhabited α]
abbrev Lane := TileIndex [64,16]
def ptr (r : RegionName) (off : Fin 64 → Fin 16 → Nat) : Values α .ptr [64,16] :=
  fun i => (r,off i.1 i.2.1)
def raw (r : RegionName) (dtype : FloatDType) (off : Fin 64 → Fin 16 → Nat) (u : State α) : Lane → α :=
  fun i => toFloat dtype ((u.mem r (off i.1 i.2.1)).read dtype.toTileDType)
def scatter (r : RegionName) (dtype : FloatDType) (off : Fin 64 → Fin 16 → Nat)
    (value : Lane → α) (u : State α) : State α :=
  (TileShape.allIndices [64,16]).foldl
    (fun v i => v.write r (off i.1 i.2.1) (.mk dtype.toTileDType (ofFloat dtype (value i)))) u
@[simp] theorem scatter_regs (r : RegionName) (dtype : FloatDType) (off : Fin 64 → Fin 16 → Nat)
    (value : Lane → α) (u : State α) : (scatter r dtype off value u).regs=u.regs := by
  apply State.scatter_regs
@[simp] theorem scatter_other (r : RegionName) (dtype : FloatDType) (off : Fin 64 → Fin 16 → Nat)
    (value : Lane → α) (u : State α) (q : RegionName) (o : Nat) (different : q≠r) :
    (scatter r dtype off value u).mem q o=u.mem q o :=
  State.scatter_frame _ _ _ _ _ _ (Or.inl different) _

#axiomsClean source_body
#axiomsClean address_unique
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelProgram
