import bench.tritonbench_g.streamk_matmul.Proofs.Decode

/-! Parametricity of the existing decoded arithmetic, for interpreting closed
symbolic instruction data. These hypotheses are ordinary interpreter mapping
equations, discharged by evaluation definitions; no numerical assumption is
registered or introduced into a public theorem. -/
namespace VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
open Whole001 Source
noncomputable section
set_option maxHeartbeats 4000000

structure InstructionMap (source : Instructions α) (target : Instructions β) (f : α → β) : Prop where
  zero : f source.zero=target.zero
  mma : ∀ a b seed, f (source.mma8 a b seed)=
    target.mma8 (fun j=>f (a j)) (fun j=>f (b j)) (f seed)
  half : ∀ x, f (source.halfRoundTrip x)=target.halfRoundTrip (f x)
  atomic : ∀ x y, f (source.atomicAdd x y)=target.atomicAdd (f x) (f y)

theorem fold_map (f : α → β) (source : α → ι → α) (target : β → ι → β)
    (law : ∀ a i,f (source a i)=target (f a) i) (xs : List ι) (seed : α) :
    f (xs.foldl source seed)=xs.foldl target (f seed) := by
  induction xs generalizing seed with
  | nil=>rfl
  | cons i rest ih=>rw [List.foldl_cons,ih,law,List.foldl_cons]

theorem inputA_map (p : Parameters) (tile : Tile) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (a : A p → α)
    (env : Env) (lane : Lanes) (iter : Nat) :
    f (inputA p tile source a env lane iter)=
      inputA p tile target (fun i=>f (a i)) env lane iter := by
  unfold inputA
  dsimp only
  split
  · split
    · rfl
    · exact law.zero
  · exact law.zero

theorem inputB_map (p : Parameters) (tile : Tile) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (b : B p → α)
    (env : Env) (lane : Lanes) (iter : Nat) :
    f (inputB p tile source b env lane iter)=
      inputB p tile target (fun i=>f (b i)) env lane iter := by
  unfold inputB
  dsimp only
  split
  · split
    · rfl
    · exact law.zero
  · exact law.zero

theorem dotBlock_map (p : Parameters) (tile : Tile) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (a : A p → α) (b : B p → α)
    (env : Env) (lane : Lane p) (iter : Nat) (seed : α) :
    f (dotBlock p tile source a b env lane iter seed)=
      dotBlock p tile target (fun i=>f (a i)) (fun i=>f (b i)) env lane iter (f seed) := by
  unfold dotBlock
  apply fold_map
  intro acc group
  rw [law.mma]
  congr 1
  · funext j
    exact inputA_map p tile source target f law a env _ iter
  · funext j
    exact inputB_map p tile source target f law b env _ iter

theorem dotValue_map (p : Parameters) (tile : Tile) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (a : A p → α) (b : B p → α)
    (env : Env) (lane : Lane p) :
    f (dotValue p tile source a b env lane)=
      dotValue p tile target (fun i=>f (a i)) (fun i=>f (b i)) env lane := by
  unfold dotValue
  rw [fold_map f _
    (fun acc iter=>dotBlock p tile target (fun i=>f (a i)) (fun i=>f (b i)) env lane iter acc)
    (fun seed iter=>dotBlock_map p tile source target f law a b env lane iter seed),law.zero]

theorem loadA_map (p : Parameters) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (a : A p → α) (row k : Nat) :
    f (loadA p source.zero a row k)=loadA p target.zero (fun i=>f (a i)) row k := by
  unfold loadA
  split
  · rfl
  · exact law.zero

theorem loadB_map (p : Parameters) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (b : B p → α) (k col : Nat) :
    f (loadB p source.zero b k col)=loadB p target.zero (fun i=>f (b i)) k col := by
  unfold loadB
  split
  · rfl
  · exact law.zero

theorem block_map (p : Parameters) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (a : A p → α) (b : B p → α)
    (tile iter : Nat) (lane : Lane p) (seed : α) :
    f (block p source a b tile iter lane seed)=
      block p target (fun i=>f (a i)) (fun i=>f (b i)) tile iter lane (f seed) := by
  unfold block
  apply fold_map
  intro acc group
  rw [law.mma]
  congr 1
  · funext j
    exact loadA_map p source target f law a _ _
  · funext j
    exact loadB_map p source target f law b _ _

theorem chunk_map (p : Parameters) (source : Instructions α) (target : Instructions β)
    (f : α → β) (law : InstructionMap source target f) (a : A p → α) (b : B p → α)
    (tile lo hi : Nat) (lane : Lane p) :
    f (chunkValue p source a b tile lo hi lane)=
      chunkValue p target (fun i=>f (a i)) (fun i=>f (b i)) tile lo hi lane := by
  unfold chunkValue
  rw [fold_map f _
    (fun acc iter=>block p target (fun i=>f (a i)) (fun i=>f (b i)) tile iter lane acc)
    (fun seed iter=>block_map p source target f law a b tile iter lane seed),law.zero]

#axiomsClean inputA_map
#axiomsClean inputB_map
#axiomsClean dotBlock_map
#axiomsClean dotValue_map
#axiomsClean chunk_map
end
end VeriTile.Bench.Repairs.StreamK.TileChecked.Reader
