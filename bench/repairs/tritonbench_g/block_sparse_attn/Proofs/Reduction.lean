import bench.repairs.tritonbench_g.block_sparse_attn.Kernels
namespace VeriTile.Bench.Repairs.BlockSparseAttn.SM90B16

def localTree (o : Fin 4) : Tree 16 :=
  .merge (.merge (.merge (.leaf ⟨2*o.val,by omega⟩)
    (.leaf ⟨2*o.val+1,by omega⟩)) (.leaf ⟨8+2*o.val,by omega⟩))
      (.leaf ⟨9+2*o.val,by omega⟩)
def xor2 (o : Fin 4) : Fin 4 := Fin.ofNat 4 (Nat.xor o.val 2)
def xor1 (o : Fin 4) : Fin 4 := Fin.ofNat 4 (Nat.xor o.val 1)
def stage2 (o : Fin 4) : Tree 16 := .merge (localTree o) (localTree (xor2 o))
def tree (o : Fin 4) : Tree 16 := .merge (stage2 o) (stage2 (xor1 o))

theorem coverage (o : Fin 4) : (tree o).leaves.Perm (List.finRange 16) := by
  fin_cases o <;> decide

/-- Four row replicas, local [2o,2o+1,8+2o,9+2o], XOR2 then XOR1.
This concrete data is checked against the B16 TTGIR/LLVM/PTX profile. -/
def plan : Reduction 16 where
  replicas:=4
  tree:=tree
  keyOwner:=fun j=>owner j.val
  outputOwner:=owner
  coverage:=coverage

/-- Concrete structural compiler profile. The retained AST/TTGIR/PTX audit
binds these finite data fields to Triton 3.7.1 SM90, four warps, three stages.
This is not a numeric operation or a whole-result/compiler callback. -/
def Binding (x : Layout) : Prop :=
  x.shape.blockM=16 ∧ x.shape.blockN=16 ∧ x.shape.blockD=16 ∧
  x.scoreInstruction=.mma16n8k8 ∧ x.valueInstruction=.mma16n8k8 ∧ HEq x.reduction plan

#axiomsClean coverage
end VeriTile.Bench.Repairs.BlockSparseAttn.SM90B16
