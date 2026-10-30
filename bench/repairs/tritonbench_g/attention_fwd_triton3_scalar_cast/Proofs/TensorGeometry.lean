import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.Bytes
import bench.repairs.tritonbench_g.attention_fwd_triton3.Proofs.Return

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBinding
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3
open Host Geometry Ownership
noncomputable section

def index4 (H R D z h row dim : Nat) : Nat :=
  z*(H*(R*D))+(h*(R*D)+(row*D+dim))

theorem join_lt (outer inner width height : Nat) (ho : outer<height) (hi : inner<width) :
    outer*width+inner<height*width := by nlinarith

theorem index4_bound (Z H R D z h row dim : Nat)
    (hz : z<Z) (hh : h<H) (hr : row<R) (hd : dim<D) :
    index4 H R D z h row dim<Z*(H*(R*D)) :=
  join_lt _ _ _ _ hz (join_lt _ _ _ _ hh (join_lt _ _ _ _ hr hd))

theorem offset_cons (dim head tail : Nat) (shape : List Nat) (stride : Nat) (strides : List Nat)
    (bound : tail<shape.prod) :
    offset (dim::shape) (stride::strides) (head*shape.prod+tail)=
      head*stride+offset shape strides tail := by
  simp only [offset,Nat.mul_comm head shape.prod,
    Nat.mul_add_div (by omega : 0<shape.prod),Nat.div_eq_of_lt bound,Nat.add_zero,
    Nat.mul_add_mod,Nat.mod_eq_of_lt bound]

theorem offset_four (Z H R D : Nat) (s : Strides) (z h row dim : Nat)
    (hh : h<H) (hr : row<R) (hd : dim<D) :
    offset [Z,H,R,D] [s.batch,s.head,s.row,s.dim] (index4 H R D z h row dim)=
      s.at z h row dim := by
  have tail2 := join_lt _ _ _ _ hr hd
  have tail3 := join_lt _ _ _ _ hh tail2
  have first := offset_cons Z z (h*(R*D)+(row*D+dim)) [H,R,D] s.batch [s.head,s.row,s.dim]
    (by simpa using tail3)
  have second := offset_cons H h (row*D+dim) [R,D] s.head [s.row,s.dim] (by simpa using tail2)
  have third := offset_cons R row dim [D] s.row [s.dim] (by simpa using hd)
  simp only [List.prod_cons,List.prod_nil,Nat.mul_one] at first second third
  rw [index4,first,second,third]
  simp [offset,Strides.at,Nat.add_assoc]

theorem output_metadata (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r) :
    (r .output).tensor.metadata.shape=[c.Z,c.H,c.rounded,c.D] ∧
    (r .output).strides=[c.o.batch,c.o.head,c.o.row,c.o.dim] ∧
    (r .center).tensor.metadata.shape=[c.Z,c.H,c.rounded] ∧
    (r .denominator).tensor.metadata.shape=[c.Z,c.H,c.rounded] ∧
    (r .center).tensor.metadata.isContiguous=Bool.true ∧
    (r .denominator).tensor.metadata.isContiguous=Bool.true := by
  have checks := a.checked.metadata
  simp only [Checked.metadataChecks,decide_eq_true_eq] at checks
  obtain ⟨_,_,_,_,os,ms,ls,mc,lc,_,_,_,ost,_⟩ := checks
  exact ⟨os,ost,ms,ls.trans ms,mc,lc⟩

theorem output_index (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r) (z h row dim : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : row<c.rounded) (hd : dim<c.D) :
    index4 c.H c.rounded c.D z h row dim<(r .output).tensor.metadata.numel ∧
    offset (r .output).tensor.metadata.shape (r .output).strides
      (index4 c.H c.rounded c.D z h row dim)=c.o.at z h row dim := by
  obtain ⟨shape,strides,_⟩ := output_metadata c A p r a
  constructor
  · simpa [Torch.TensorMetadata.numel,shape] using index4_bound _ _ _ _ _ _ _ _ hz hh hr hd
  · rw [shape,strides]
    exact offset_four _ _ _ _ _ _ _ _ _ hh hr hd

theorem contiguous_offset (x : Payload) (laws : Laws x)
    (contiguous : x.tensor.metadata.isContiguous=Bool.true)
    (i : Nat) (bound : i<x.tensor.metadata.numel) :
    offset x.tensor.metadata.shape x.strides i=i := by
  have first := laws.physical i bound
  have second := laws.tensor.contiguous contiguous i bound
  omega

theorem carry_index (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r) (role : Role)
    (carry : role=.center ∨ role=.denominator) (z h row : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : row<c.rounded) :
    (z*c.H+h)*c.rounded+row<(r role).tensor.metadata.numel ∧
    offset (r role).tensor.metadata.shape (r role).strides ((z*c.H+h)*c.rounded+row)=
      (z*c.H+h)*c.rounded+row := by
  have metadata := output_metadata c A p r a
  have shape : (r role).tensor.metadata.shape=[c.Z,c.H,c.rounded] := by
    rcases carry with h|h <;> subst role
    · exact metadata.2.2.1
    · exact metadata.2.2.2.1
  have contiguous : (r role).tensor.metadata.isContiguous=Bool.true := by
    rcases carry with h|h <;> subst role
    · exact metadata.2.2.2.2.1
    · exact metadata.2.2.2.2.2
  have bound := join_lt _ _ _ _ (join_lt _ _ _ _ hz hh) hr
  have bound' : (z*c.H+h)*c.rounded+row<(r role).tensor.metadata.numel := by
    simpa [Torch.TensorMetadata.numel,shape,Nat.mul_assoc] using bound
  exact ⟨bound',contiguous_offset _ (a.laws role) contiguous _ bound'⟩

theorem query_within_workspace (c : Call) (row : Nat) (hr : row<c.NQ) : row<c.rounded := by
  have h := Geometry.row_coverage c.NQ 64 row (by decide) hr
  have m := Nat.mul_le_mul_right 64 (show row/64+1≤Geometry.tiles c.NQ 64 by omega)
  have rest := Nat.mod_lt row (by decide : 0<64)
  have eqn := Nat.div_add_mod row 64
  unfold Call.rounded
  omega

theorem output_coordinate (c : Call) (A : FlatAlloc) (p : Checked.Pointers A)
    (r : Request) (a : Admission c A p r) (z h row dim : Nat)
    (hz : z<c.Z) (hh : h<c.H) (hr : row<c.NQ) (hd : dim<c.D) :
    ∃ idx : GridIndex c.grid,∃ lane : Lane c.BM c.D,
      active c lane ∧ Ownership.address c idx lane=(region .output,
        offset (r .output).tensor.metadata.shape (r .output).strides
          (index4 c.H c.rounded c.D z h row dim)) := by
  rw [(output_index c A p r a z h row dim hz hh (query_within_workspace c row hr) hd).2]
  exact Return.output_covered c (Contract.safety a.checked).valid z h row dim hz hh hr hd

#axiomsClean output_index
#axiomsClean carry_index
#axiomsClean output_coordinate
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.TensorBinding
