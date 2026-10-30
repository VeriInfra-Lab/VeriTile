import bench.tritonbench_g.matmul_persistent_triton.IO
import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulPersistentTriton
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 16000
variable {α : Type}

@[simp] theorem payload_value (profile : Storage) (x : α) : payload profile (value profile x)=x := by
  cases profile <;> rfl
@[simp] theorem pack_value (profile : Storage) (x : α) :
    Cell.mk (Core.dtype profile) (value profile x)=Core.pack profile x := by
  cases profile <;> rfl

@[simp] theorem payload_read [Inhabited α] (profile : Storage) (c : Cell α) :
    payload profile (c.read (Core.dtype profile))=Core.readCell profile c := by
  cases profile <;> rfl

@[simp] theorem payload_map (profile : Storage) (x : Value α (Core.dtype profile)) :
    payload profile (mapValue id (Core.dtype profile) x)=payload profile x := by
  cases profile <;> rfl

/-- Independent matrix operands; zero is used only for masked K lanes or
empty axes, never for a live logical matrix coordinate. -/
def left (zero : α) (p : Core.Parameters) (xs : ChannelValues α (inputs p))
    (rowBlock iteration : Nat) (row : Fin 128) (lane : Fin p.problem.profile.blockK) : α :=
  let r := clamp p.problem.M (rowBlock*128+row.val)
  let k := iteration*p.problem.profile.blockK+lane.val
  if hr : r<p.problem.M then if hk : k<p.problem.K then
    payload p.problem.profile (xs 0 (⟨r,hr⟩,⟨k,hk⟩,PUnit.unit)) else zero else zero

def right (zero : α) (p : Core.Parameters) (xs : ChannelValues α (inputs p))
    (columnBlock iteration : Nat) (lane : Fin p.problem.profile.blockK) (column : Fin 256) : α :=
  let c := clamp p.problem.N (columnBlock*256+column.val)
  let k := iteration*p.problem.profile.blockK+lane.val
  if hk : k<p.problem.K then if hc : c<p.problem.N then
    payload p.problem.profile (xs 1 (⟨k,hk⟩,⟨c,hc⟩,PUnit.unit)) else zero else zero

def accumulator (P : Source.Arithmetic α α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) (rowBlock columnBlock : Nat) : Nat → Source.Acc α
  | 0 => fun _ _ => P.accumulatorZero
  | q+1 => P.dot p.problem.profile.blockK (left P.inputZero p xs rowBlock q)
      (right P.inputZero p xs columnBlock q) (accumulator P p xs rowBlock columnBlock q)

def result (P : Source.Arithmetic α α) (p : Core.Parameters)
    (xs : ChannelValues α (inputs p)) : ChannelValues α (outputs p) :=
  fun | ⟨0,_⟩ => fun j => value p.problem.profile (P.cast p.problem.profile
    (accumulator P p xs (j.1.val/128) (j.2.1.val/256) p.problem.kTiles
      ⟨j.1.val%128,Nat.mod_lt _ (by decide)⟩ ⟨j.2.1.val%256,Nat.mod_lt _ (by decide)⟩))

/-- The mathematical specification is the ordinary K-indexed matrix product. -/
def mathematicalValue (p : Core.Parameters) (xs : ChannelValues ℝ (inputs p))
    (j : TileIndex [p.problem.M,p.problem.N]) : Value ℝ (Core.dtype p.problem.profile) :=
  value p.problem.profile
    (∑ k : Fin p.problem.K,
      payload p.problem.profile (xs 0 (j.1,k,PUnit.unit)) *
      payload p.problem.profile (xs 1 (k,j.2.1,PUnit.unit)))

def mathematical (p : Core.Parameters) (xs : ChannelValues ℝ (inputs p)) :
    ChannelValues ℝ (outputs p) := fun | ⟨0,_⟩ => mathematicalValue p xs

#specNonCircular mathematical avoiding [Core.executeGrid,Core.program,Source.run]
#specNonCircular result avoiding [Core.executeGrid,Core.program,Source.run,Source.accumulator]
end
end VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.PublicData
