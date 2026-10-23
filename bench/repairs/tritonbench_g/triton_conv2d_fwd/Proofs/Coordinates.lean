import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Formed
namespace VeriTile.Bench.Repairs.Conv2d

def coordinates (s : Shape) (g r f h w c : Nat) : List Nat :=
  [g,r,f,h,w,c,r/s.ow,batch s r,outH s r,outW s r,g*s.cg,g*s.og,g*s.og+f,
    s.outputs*s.oh*s.ow,s.oh*s.ow,s.ow,1]

theorem coordinates_bound (s : Shape) (v : Valid s) (g r f h w c : Nat)
    (hg : g<s.groups) (hr : r<pSpatial s) (hf : f<pFeatures s) (hh : h<s.kh)
    (hw : w<s.kw) (hc : c<s.padded) (z : Nat) (hz : z∈coordinates s g r f h w c) :
    z ≤ envelope s := by
  have scalar := parameters_bound s
  have ge := scalar s.groups (by simp [scalarParameters])
  have re := scalar (pSpatial s) (by simp [scalarParameters])
  have fe := scalar (pFeatures s) (by simp [scalarParameters])
  have he := scalar s.kh (by simp [scalarParameters])
  have we := scalar s.kw (by simp [scalarParameters])
  have ce := scalar s.padded (by simp [scalarParameters])
  have ohe := scalar s.oh (by simp [scalarParameters])
  have owe := scalar s.ow (by simp [scalarParameters])
  have rd := Nat.div_le_self r s.ow
  have bd : batch s r ≤ r := le_trans (Nat.div_le_self (r/s.ow) s.oh) rd
  have oh := Nat.mod_lt (r/s.ow) v.oh
  have ow := Nat.mod_lt r v.ow
  have gi : g*s.cg ≤ s.channels := Nat.mul_le_mul_right s.cg (Nat.le_of_lt hg)
  have go : g*s.og ≤ s.outputs := Nat.mul_le_mul_right s.og (Nat.le_of_lt hg)
  have cie := scalar s.channels (by simp [scalarParameters])
  have coe : s.outputs+pFeatures s ≤ envelope s := by unfold envelope;omega
  have one : 1 ≤ pBatch s := Nat.succ_le_succ (Nat.zero_le _)
  have stride : s.outputs*s.oh*s.ow ≤ pBatch s*s.outputs*s.oh*s.ow := by
    simpa only [Nat.mul_assoc,one_mul] using Nat.mul_le_mul_right (s.outputs*s.oh*s.ow) one
  have eo : outputEnvelope s ≤ envelope s := by unfold envelope;omega
  unfold outputEnvelope at eo
  simp only [coordinates,List.mem_cons,List.not_mem_nil,or_false] at hz
  rcases hz with hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz|hz <;>
    subst z <;> (try simp only [outH,outW])
  all_goals omega

theorem group_division (s : Shape) (v : Valid s) :
    s.channels/s.groups=s.cg ∧ s.outputs/s.groups=s.og := by
  simp [Shape.channels,Shape.outputs,Nat.mul_div_cancel_left _ v.groups]

theorem channel_lane (s : Shape) (cb k : Nat) (hb : cb<blocks s.cg 32) (hk : k<32) :
    cb*32+k<s.padded ∧ (cb+1)*32 ≤ s.padded := by
  unfold Shape.padded
  omega

theorem source_grid_arithmetic (s : Shape) (v : Valid s) (p : Program) (l : Lane)
    (hp : p∈programs s) (hl : l∈lanes) :
    p.1 ≤ envelope s ∧ p.2.1 ≤ envelope s ∧ p.2.2 ≤ envelope s ∧
      p.1*128 ≤ envelope s ∧ p.2.1*32 ≤ envelope s := by
  have bounds := source_offsets s v p l hp hl
  have re := parameters_bound s (pSpatial s) (by simp [scalarParameters])
  have fe := parameters_bound s (pFeatures s) (by simp [scalarParameters])
  have ge := parameters_bound s s.groups (by simp [scalarParameters])
  simp only [cellOf] at bounds
  omega

#axiomsClean coordinates_bound
#axiomsClean group_division
#axiomsClean channel_lane
#axiomsClean source_grid_arithmetic
end VeriTile.Bench.Repairs.Conv2d
