import bench.repairs.tritonbench_g.fast_layernorm.Proofs.Runtime
import bench.repairs.tritonbench_g.fast_layernorm.Proofs.BackwardRuntime

/-! Exact row/column recipes and native integer bounds. Active byte addresses
come from the actual allocated tensors. Masked inactive address expressions
need not denote valid storage, but their signed element arithmetic is bounded. -/
namespace VeriTile.Bench.Repairs.FastLayernorm.Addresses
open VeriTile.Bench.Repairs.LayernormFwdTriton
open Storage Host Preparation Execution Layout Compiler
noncomputable section

variable {api : Memory.API α Store Byte} {neg : α → α} {inputs : List Storage.Tensor}
  {lib : APIHost api neg inputs Metadata} {x w : Storage.Tensor} {bias : Option Storage.Tensor}
  {rowCount N : Nat} {initial : State Store Metadata}

theorem forward_strides (r : Preparation.Route lib x w bias rowCount N initial) (hn : 0<N)
    (xc : r.data.x.tensor.view.strides=contiguousStrides r.data.x.tensor.view.shape)
    (yc : r.data.y.tensor.view.strides=contiguousStrides r.data.y.tensor.view.shape) :
    r.data.x.tensor.view.strides.headD 0=N ∧ r.data.y.tensor.view.strides.headD 0=N := by
  have hx := (materialize_metadata r.xr).1
  have hy := (lib.emptyLikeResult r.xr.output r.br.final r.y r.yOK).2.1
  exact ⟨matrix_row_stride _ _ _ hx hn xc,matrix_row_stride _ _ _ (hy.trans hx) hn yc⟩

theorem row_recipe (d : Dense) (rowCount N row col : Nat) (shape : d.count=rowCount*N)
    (hr : row<rowCount) (hc : col<N) :
    addressOffset (.rowColumn 0) [N] row col N=row*N+col ∧
    row*N+col<2^63 ∧ d.address (row*N+col)+4≤2^64 := by
  have h := row_offset_safe d rowCount N row col shape hr hc
  exact ⟨rfl,h.2,dense_address_bound d _ h.1⟩

theorem masked_row_recipe (d : Dense) (rowCount N row col : Nat) (shape : d.count=rowCount*N)
    (hn : 0<N) (hr : row<rowCount) :
    addressOffset .maskedRowColumn [] row col N=row*N+(if col<N then col else 0) ∧
    addressOffset .maskedRowColumn [] row col N<2^63 ∧
    d.address (addressOffset .maskedRowColumn [] row col N)+4≤2^64 := by
  have h := masked_affine_offset_safe d rowCount N row col shape hn hr
  exact ⟨rfl,h.2,dense_address_bound d _ h.1⟩

theorem inactive_row_offset (d : Dense) (rowCount N row col B : Nat)
    (shape : d.count=rowCount*N) (hr : row<rowCount) (hc : col<B) (hb : B≤65536) :
    row*N+col<2^63 := by
  have hd := dense_count_bound d
  have hm := Nat.mul_le_mul_right N (show row≤rowCount by omega)
  norm_num at *
  omega

theorem unsigned64_exact (n : Nat) (h : n<2^63) : n%(2^64)=n := by
  apply Nat.mod_eq_of_lt
  norm_num at *
  omega

theorem affine_column_bound (weight : Dense) (bias : Option Dense) (tile lane : Nat)
    (ht : tile<(max weight.count (bias.map Dense.count |>.getD 0)+127)/128) (hl : lane<128) :
    tile*128+lane<2^63 := by
  have hw := dense_count_bound weight
  have hb : (bias.map Dense.count |>.getD 0)≤2^62 := by
    cases bias with
    | none => simp
    | some b => exact dense_count_bound b
  have hc : max weight.count (bias.map Dense.count |>.getD 0)≤2^62 := max_le hw hb
  have ht' := Nat.mul_le_mul_right 128 (show tile≤(max weight.count (bias.map Dense.count |>.getD 0)+127)/128 by omega)
  have hd := Nat.div_mul_le_self (max weight.count (bias.map Dense.count |>.getD 0)+127) 128
  norm_num at *
  omega

theorem loop_step (x : Dense) (rowCount N row : Nat) (shape : x.count=rowCount*N)
    (hn : 0<N) (hr : row<rowCount) :
    row+1≤rowCount ∧ row+1<2^63 ∧ rowCount-(row+1)<rowCount-row := by
  have hle : row+1≤rowCount := by omega
  exact ⟨hle,loop_counter_safe x rowCount N shape hn (row+1) hle,by omega⟩

theorem decoded_addresses (code : Selected decoder N B) :
    (decoder.inspect B code.object).addresses=addressRecipes code.object.entry := code.addresses

#axiomsClean forward_strides
#axiomsClean row_recipe
#axiomsClean masked_row_recipe
#axiomsClean inactive_row_offset
#axiomsClean unsigned64_exact
#axiomsClean affine_column_bound
#axiomsClean loop_step
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm.Addresses
