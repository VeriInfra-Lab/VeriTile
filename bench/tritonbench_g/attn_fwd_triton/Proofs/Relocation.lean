import bench.tritonbench_g.attn_fwd_triton.Proofs.Memory
import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.KernelLemmas.EvalHelpers

/-! Pointer-base relocation permits independent views in one physical memory.
The region map is not assumed disjoint: read-only roles may overlap. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Relocation
open VeriTile.Triton Memory
noncomputable section

def parameters (p : Parameters) (flat : RegionName) : Parameters :=
  {p with Q := flat,K := flat,V := flat,QS := flat,KS := flat,Out := flat}

def offsets (A : FlatAlloc) (p : Parameters) : BaseOffsets where
  query := A.base p.Q
  key := A.base p.K
  value := A.base p.V
  output := A.base p.Out
  queryScale := A.base p.QS
  keyScale := A.base p.KS

def config (A : FlatAlloc) (c : Config) : Config :=
  {c with p := parameters c.p A.flat, offsets := {
    query := A.base c.p.Q+c.offsets.query
    key := A.base c.p.K+c.offsets.key
    value := A.base c.p.V+c.offsets.value
    output := A.base c.p.Out+c.offsets.output
    queryScale := A.base c.p.QS+c.offsets.queryScale
    keyScale := A.base c.p.KS+c.offsets.keyScale}}

def shifted {shape : TileShape} (flat : RegionName) (base : Nat) (ptr : Tile .ptr shape) : Tile .ptr shape :=
  ⟨fun i => (flat.cast,base+(ptr.data i).2)⟩

theorem shift_base {shape : TileShape} (bc : Broadcast [] shape shape) (s : BlockState) (region flat : RegionName) (base : Nat)
    (e : Op .nat shape) (ptr : Tile .ptr shape)
    (h : evalOp (.ptrAdd bc (.ptrBase region) e) s = some ptr) :
    evalOp (.ptrAdd bc (.ptrAdd .nil (.ptrBase flat) (.constNat base)) e) s =
      some (shifted flat base ptr) := by
  cases he : evalOp e s with
  | none =>
    simp only [evalOp_ptrAdd,evalOp_ptrBase,he,bind,Option.bind] at h
    cases h
  | some off =>
    simp only [evalOp_ptrAdd,evalOp_ptrBase,evalOp_constNat,he,bind,Option.bind] at h ⊢
    cases h
    congr 1
    apply Tile.ext
    intro i
    simp [shifted,Tile.ptrAdd_data,Tile.scalar,-Tile.scalar_eta]

theorem flatten_ptr {shape : TileShape} (A : FlatAlloc) (bc : Broadcast [] shape shape)
    (s : BlockState) (region : RegionName) (e : Op .nat shape) (ptr : Tile .ptr shape)
    (he : A.flattenOp e = e) (h : evalOp (.ptrAdd bc (.ptrBase region) e) s = some ptr) :
    evalOp (A.flattenOp (.ptrAdd bc (.ptrBase region) e)) s = some (shifted A.flat (A.base region) ptr) := by
  simpa only [FlatAlloc.flattenOp,he,Region.cast] using shift_base bc s region A.flat (A.base region) e ptr h

theorem shift_query (A : FlatAlloc) (c : Config) (BM BD : Nat) :
    shifted A.flat (A.base c.p.Q) (queryPointer c BM BD) = queryPointer (config A c) BM BD := by
  apply Tile.ext
  intro i
  simp [shifted,queryPointer,config,parameters,qbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem shift_key (A : FlatAlloc) (c : Config) (BD BN start : Nat) :
    shifted A.flat (A.base c.p.K) (keyPointer c BD BN start) = keyPointer (config A c) BD BN start := by
  apply Tile.ext
  intro i
  simp [shifted,keyPointer,config,parameters,kbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem shift_value (A : FlatAlloc) (c : Config) (BN BD start : Nat) :
    shifted A.flat (A.base c.p.V) (valuePointer c BN BD start) = valuePointer (config A c) BN BD start := by
  apply Tile.ext
  intro i
  simp [shifted,valuePointer,config,parameters,vbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem shift_output (A : FlatAlloc) (c : Config) (BM BD : Nat) :
    shifted A.flat (A.base c.p.Out) (outputPointer c BM BD) = outputPointer (config A c) BM BD := by
  apply Tile.ext
  intro i
  simp [shifted,outputPointer,config,parameters,obase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem shift_queryScale (A : FlatAlloc) (c : Config) (BM : Nat) :
    shifted A.flat (A.base c.p.QS) (queryScalePointer c BM) = queryScalePointer (config A c) BM := by
  apply Tile.ext
  intro i
  simp [shifted,queryScalePointer,Tile.scalar,config,parameters,qsbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,-Tile.scalar_eta]

theorem shift_keyScale (A : FlatAlloc) (c : Config) (BN start : Nat) :
    shifted A.flat (A.base c.p.KS) (keyScalePointer c BN start) = keyScalePointer (config A c) BN start := by
  apply Tile.ext
  intro i
  simp [shifted,keyScalePointer,Tile.scalar,config,parameters,ksbase,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,-Tile.scalar_eta]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.Relocation
