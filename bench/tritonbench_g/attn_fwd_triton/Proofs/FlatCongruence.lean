import VeriTile.Triton.Memory.Flatten
import bench.tritonbench_g.attn_fwd_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatCongruence
open VeriTile.Triton

@[simp] theorem expand (A : FlatAlloc) {d : TileDType} {shape : TileShape}
    (ax : Fin (shape.length+1)) (e : Op d shape) :
    A.flattenOp (.expandDim ax e) = .expandDim ax (A.flattenOp e) := by
  rw [FlatAlloc.flattenOp]

@[simp] theorem sum (A : FlatAlloc) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (e : Op .real shape) :
    A.flattenOp (.reduceSum ax keep e) = .reduceSum ax keep (A.flattenOp e) := by
  rw [FlatAlloc.flattenOp]

@[simp] theorem max (A : FlatAlloc) {shape : TileShape}
    (ax : Fin shape.length) (keep : Bool) (e : Op .real shape) :
    A.flattenOp (.reduceMax ax keep e) = .reduceMax ax keep (A.flattenOp e) := by
  rw [FlatAlloc.flattenOp]

@[simp] theorem dot (A : FlatAlloc) {batch : TileShape} {M K N : Nat}
    (a : Op .real (batch++[M,K])) (b : Op .real (batch++[K,N])) (opts : DotOptions) :
    A.flattenOp (.dot a b opts) = .dot (A.flattenOp a) (A.flattenOp b) opts := by
  rw [FlatAlloc.flattenOp]

@[simp] theorem vec0 (A : FlatAlloc) {d : TileDType} {N : Nat} (e : Op d [N]) :
    @FlatAlloc.flattenOp A d [1,N] (.expandDim ⟨0,by simp⟩ e) = .expandDim ⟨0,by simp⟩ (A.flattenOp e) :=
  expand A 0 e

@[simp] theorem vec1 (A : FlatAlloc) {d : TileDType} {N : Nat} (e : Op d [N]) :
    @FlatAlloc.flattenOp A d [N,1] (.expandDim ⟨1,by simp⟩ e) = .expandDim ⟨1,by simp⟩ (A.flattenOp e) :=
  expand A 1 e

@[simp] theorem sum1 (A : FlatAlloc) {M N : Nat} (e : Op .real [M,N]) :
    @FlatAlloc.flattenOp A .real [M] (.reduceSum ⟨1,by simp⟩ Bool.false e) = .reduceSum ⟨1,by simp⟩ Bool.false (A.flattenOp e) :=
  sum A 1 Bool.false e

@[simp] theorem max1 (A : FlatAlloc) {M N : Nat} (e : Op .real [M,N]) :
    @FlatAlloc.flattenOp A .real [M] (.reduceMax ⟨1,by simp⟩ Bool.false e) = .reduceMax ⟨1,by simp⟩ Bool.false (A.flattenOp e) :=
  max A 1 Bool.false e

@[simp] theorem vec0n (A : FlatAlloc) {d : TileDType} {N : Nat} (e : Op d [N]) :
    @FlatAlloc.flattenOp A d [1,N] (.expandDim 0 e) = .expandDim 0 (A.flattenOp e) :=
  expand A 0 e

@[simp] theorem vec1n (A : FlatAlloc) {d : TileDType} {N : Nat} (e : Op d [N]) :
    @FlatAlloc.flattenOp A d [N,1] (.expandDim 1 e) = .expandDim 1 (A.flattenOp e) :=
  expand A 1 e

@[simp] theorem sum1n (A : FlatAlloc) {M N : Nat} (e : Op .real [M,N]) :
    @FlatAlloc.flattenOp A .real [M] (.reduceSum 1 Bool.false e) = .reduceSum 1 Bool.false (A.flattenOp e) :=
  sum A 1 Bool.false e

@[simp] theorem max1n (A : FlatAlloc) {M N : Nat} (e : Op .real [M,N]) :
    @FlatAlloc.flattenOp A .real [M] (.reduceMax 1 Bool.false e) = .reduceMax 1 Bool.false (A.flattenOp e) :=
  max A 1 Bool.false e

@[simp] theorem dot2 (A : FlatAlloc) {M K N : Nat}
    (a : Op .real [M,K]) (b : Op .real [K,N]) (opts : DotOptions) :
    @FlatAlloc.flattenOp A .real [M,N] (.dot (batch := []) a b opts) = .dot (batch := []) (A.flattenOp a) (A.flattenOp b) opts :=
  dot A (batch := []) (M := M) (K := K) (N := N) a b opts

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.FlatCongruence
