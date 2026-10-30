import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.QuantProfile

/-! State reached by the checked raw `_quant_row` expansion. The execution
proofs check this explicit state against the actual source-derived blocks. -/
namespace VeriTile.Bench.Repairs.KvCacheFilling.QuantExecution
open VeriTile.Triton FP.Structural QuantProfile
noncomputable section

def helperNames0 : List String :=
  ["__veritile_helper_0_arg_x",
   "__veritile_helper_1_arg_lo",
   "__veritile_helper_2_arg_hi",
   "__veritile_helper_3_local_raw_scale",
   "__veritile_helper_4_local_scale",
   "__veritile_helper_5_local_zero",
   "__veritile_helper_6_local_code",
   "__veritile_helper_7_result",
   "__veritile_helper_8_result",
   "__veritile_helper_9_result"]

def helperNames10 : List String :=
  ["__veritile_helper_10_arg_x",
   "__veritile_helper_11_arg_lo",
   "__veritile_helper_12_arg_hi",
   "__veritile_helper_13_local_raw_scale",
   "__veritile_helper_14_local_scale",
   "__veritile_helper_15_local_zero",
   "__veritile_helper_16_local_code",
   "__veritile_helper_17_result",
   "__veritile_helper_18_result",
   "__veritile_helper_19_result"]

def helperNames20 : List String :=
  ["__veritile_helper_20_arg_x",
   "__veritile_helper_21_arg_lo",
   "__veritile_helper_22_arg_hi",
   "__veritile_helper_23_local_raw_scale",
   "__veritile_helper_24_local_scale",
   "__veritile_helper_25_local_zero",
   "__veritile_helper_26_local_code",
   "__veritile_helper_27_result",
   "__veritile_helper_28_result",
   "__veritile_helper_29_result"]

def helperNames30 : List String :=
  ["__veritile_helper_30_arg_x",
   "__veritile_helper_31_arg_lo",
   "__veritile_helper_32_arg_hi",
   "__veritile_helper_33_local_raw_scale",
   "__veritile_helper_34_local_scale",
   "__veritile_helper_35_local_zero",
   "__veritile_helper_36_local_code",
   "__veritile_helper_37_result",
   "__veritile_helper_38_result",
   "__veritile_helper_39_result"]

def helperNames40 : List String :=
  ["__veritile_helper_40_arg_x",
   "__veritile_helper_41_arg_lo",
   "__veritile_helper_42_arg_hi",
   "__veritile_helper_43_local_raw_scale",
   "__veritile_helper_44_local_scale",
   "__veritile_helper_45_local_zero",
   "__veritile_helper_46_local_code",
   "__veritile_helper_47_result",
   "__veritile_helper_48_result",
   "__veritile_helper_49_result"]

def helperNames50 : List String :=
  ["__veritile_helper_50_arg_x",
   "__veritile_helper_51_arg_lo",
   "__veritile_helper_52_arg_hi",
   "__veritile_helper_53_local_raw_scale",
   "__veritile_helper_54_local_scale",
   "__veritile_helper_55_local_zero",
   "__veritile_helper_56_local_code",
   "__veritile_helper_57_result",
   "__veritile_helper_58_result",
   "__veritile_helper_59_result"]

def helperState {α : Type} (a : Ops α) (names : List String) (levels n : Nat)
    (xs : Fin n → α) (lo hi : α) (s : State α) : State α :=
  let z := s.setReg names[0]! .real [n] (fun i => xs i.1)
  let z := z.setReg names[1]! .real [] (fun _ => lo)
  let z := z.setReg names[2]! .real [] (fun _ => hi)
  let z := z.setReg names[3]! .real []
    (fun _ => a.divide (a.sub hi lo) (a.literal levels))
  let z := z.setReg names[4]! .real [] (fun _ => scale a levels lo hi)
  let z := z.setReg names[5]! .real [] (fun _ => zero a levels lo hi)
  let z := z.setReg names[6]! .nat [n]
    (fun i => (code a levels lo hi (xs i.1)).toNat)
  let z := z.setReg names[7]! .nat [n]
    (fun i => (code a levels lo hi (xs i.1)).toNat)
  let z := z.setReg names[8]! .real [] (fun _ => scale a levels lo hi)
  z.setReg names[9]! .real [] (fun _ => zero a levels lo hi)

end
end VeriTile.Bench.Repairs.KvCacheFilling.QuantExecution
