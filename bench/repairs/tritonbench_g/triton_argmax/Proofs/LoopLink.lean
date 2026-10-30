import bench.repairs.tritonbench_g.triton_argmax.Proofs.Tree
namespace VeriTile.Bench.Repairs.TritonArgmax.LoopLink
 def scan {α : Type} (step : Nat → α → α) : Nat → Nat → α → α
  | 0,_,a => a
  | n+1,start,a => scan step n (start+1) (step start a)

theorem scan_fold {α : Type} (step : Nat → α → α) (count start : Nat) (a : α) :
    scan step count start a=(List.range' start count).foldl (fun x i => step i x) a := by
  induction count generalizing start a with
  | zero => rfl
  | succ n ih => simp only [scan,List.range'_succ,List.foldl_cons,ih]

theorem ceil_tiles (n tile : Nat) : tile < (n+127)/128 ↔ tile*128 < n := by
  have h := Nat.div_add_mod (n+127) 128
  have r := Nat.mod_lt (n+127) (by omega : 0 < 128)
  omega

end VeriTile.Bench.Repairs.TritonArgmax.LoopLink
