import bench.tritonbench_g.triton_attention.Proofs.NativeKernels
namespace VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
set_option maxRecDepth 20000
set_option maxHeartbeats 4000000

def preLocalLeaves (owner : Nat) : List Nat :=
  [owner * 4 + 1,owner * 4,owner * 4 + 2,owner * 4 + 3]

def shuffleLeaves : List Nat → (Nat → List Nat) → Nat → List Nat
  | [],f,o => f o
  | mask :: rest,f,o => shuffleLeaves rest (fun j => f j ++ f (j ^^^ mask)) o

def preprocessLeaves (dim owner : Nat) : List Nat :=
  shuffleLeaves (preMasks dim) preLocalLeaves (owner % (dim / 4))

/-- Every supported native preprocess tree consumes each logical channel
exactly once, including every possible structural store-owner replica. The
order is retained as a list; this is not a floating sum reassociation law. -/
theorem preprocess_coverage (c : Shape) (owner : Nat) :
    (preprocessLeaves c.dim owner).Perm (List.range c.dim) := by
  have hd := c.dim_supported
  rcases hd with hd | hd | hd | hd
  · rw [hd]
    have h : owner % 4 < 4 := Nat.mod_lt _ (by decide)
    interval_cases hm : owner % 4 <;> simp [preprocessLeaves,preMasks,hm,shuffleLeaves,preLocalLeaves] <;> decide
  · rw [hd]
    have h : owner % 8 < 8 := Nat.mod_lt _ (by decide)
    interval_cases hm : owner % 8 <;> simp [preprocessLeaves,preMasks,hm,shuffleLeaves,preLocalLeaves] <;> decide
  · rw [hd]
    have h : owner % 16 < 16 := Nat.mod_lt _ (by decide)
    interval_cases hm : owner % 16 <;> simp [preprocessLeaves,preMasks,hm,shuffleLeaves,preLocalLeaves] <;> decide
  · rw [hd]
    have h : owner % 32 < 32 := Nat.mod_lt _ (by decide)
    interval_cases hm : owner % 32 <;> simp [preprocessLeaves,preMasks,hm,shuffleLeaves,preLocalLeaves] <;> decide

#axiomsClean preprocess_coverage
end VeriTile.Bench.Repairs.TritonAttention.Whole001.Native
