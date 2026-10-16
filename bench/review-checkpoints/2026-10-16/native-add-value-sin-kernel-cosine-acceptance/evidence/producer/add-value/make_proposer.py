from pathlib import Path
W=Path(__file__).resolve().parent;OLD=W.parent/'add-value-finite-acceptance-wip'
s=(OLD/'propose.py').read_text().replace("PUBLIC=BASE+'FiniteCorrect.lean';NAME='add_value_finite_source_outcomes'","PUBLIC=BASE+'NativeCorrect.lean';NAME='add_value_native_source_outcomes'")
s=s.replace("modules=list(load(W/'evidence/staging-map.json').values())", "modules=[str(p.relative_to(W/'after')) for p in sorted((W/'after').rglob('*.lean'))]")
s=s.replace("native='native-attempts/20261015T211358396065Z/';receipt=load(W/native/'receipt.json');", "native=str(Path(load(W/'evidence/native-latest.json')['receipt']).parent)+'/';receipt=load(W/native/'receipt.json');")
s=s.replace("'Proofs/FiniteScalar.lean'","'Proofs/NativeScalar.lean'").replace('AddValueFinite.','AddValueNative.')
start=s.index('configs=[]');end=s.index('def callback(port):',start)
s=s[:start]+'''configs=[]
observations=['input_access','missing_bytes','undefined_conversion','output_access','ownership_conflict']
for d,r in records.items():
 p=profiles.get(d);terminal='returned_or_undefined_conversion' if p else 'architecture_rejected' if 'fnuz' in d else 'binder_rejected'
 outcomes=[('allocation_error','none'),('cuda_assertion','none'),('context_error','none'),(terminal,'none')]
 if p:outcomes += [(v,'none') for v in ['compilation_error','grid_limit','pointer_packing_error','launch_setup_error','two31_no_access_return']]+[('native_observation',v) for v in observations]
 for outcome,observation in outcomes:
  compute={'index':'int32','pointer':'uint64','value_argument':'fp32'}
  if p:compute.update(value=p['compute'],store='int8' if d=='bool' else dtype(d))
  c={'dtype_profile':{'inputs':{'x':dtype(d)},'outputs':{'output':dtype(d)},'compute':compute},'configuration':{
   'entrypoint':'puzzle1','source_variant':'original','BLOCK_SIZE':1024,'value':{'python':10,'runtime_type':'fp32','bits':0x41200000,'constexpr':False},
   'numel':({'query':'not_reached_allocation_or_cuda_failure'} if outcome in ['allocation_error','cuda_assertion'] else {'lower_inclusive':2**31 if outcome=='two31_no_access_return' else 0,'upper_exclusive':2**31+1 if outcome=='two31_no_access_return' else 2**63}),
   'outcome':outcome,'native_observation':observation,'grid':'ceil_div_original_output_numel_1024','rank':'arbitrary_including_rank0_numel1',
   'target':{'backend':'cuda','arch':90,'warp_size':32},'triton_version':receipt['triton_version'],'torch_version':receipt['torch_version'],'frontend_options_sha256':sha(canonical(records['float32']['options'])),
   'debug':False,'instrumentation_mode':'none','runtime_N':'annotated_tl.int32_for_all_host_extents','bound_conversion':'PyLong_AsLong_success_then_int32_assignment_mod2pow32',
   'lane_arithmetic':'i32_pid_times1024_plus_lane_wrap_before_signed_slt','pointer_arithmetic':'sign_extend_i32_offset_then_scale_storage_width_then_wrap_uint64',
   'setup_order':['context','signature_and_backend_compilation','signed_grid_tuple_parse','pointer_and_runtime_scalar_packing','launch'],'grid_limit':2**31-1,
   'layouts':'all_actual_dense_strided_Torch_metadata; original_source_flat_data_ptr_access','view_metadata':'unconditional_Torch_storage_data_and_extracted_pointer_binding; actual_empty_like_preserve_format_strides_even_with_no_active_masks',
   'allocation':'fresh_output_empty_like_input_metadata_and_exterior_frame','permissions':'observed_at_each_actual_request_in_current_launch_address_space;owner_device_metadata_separate',
   'device':'allocation_then_two_CUDA_assertion;no_same_device_assertion;peer_or_mapped_memory_allowed_by_permissions',
   'scalar_definedness':'partial_float_to_integer_conversion_observed_before_value_materialization;no_undefined_fallback','source_pointers':'exactly_one_input_pointer_and_one_output_pointer;value_is_runtime_fp32_scalar',
   'observations':'mathematical_requests_only;no_native_trap_race_winner_rollback_or_postfailure_memory',
   'numeric_claim':'actual_loaded_word_and_source_result_with_defined_conversion_partition' if outcome=='returned_or_undefined_conversion' else 'value_insensitive_constructor_or_first_setup_error_only',
   'output_bytes':'independent_initial_memory_write_plan_and_pure_packing_final_byte_formula;untouched_bytes_equal_allocated_contents'}}
  validate_case(c);configs.append((d,c))
cases=[c for _,c in configs];assert len(cases)==238 and len({canonical(c) for c in cases})==238
''' +s[end:]
s=s.replace("c['configuration']['outcome']=='returned' and c['configuration']['size_class']!='empty'", "c['configuration']['outcome']=='returned_or_undefined_conversion'")
s=s.replace("ident='ADD-VALUE-FINITE-'", "ident='ADD-VALUE-NATIVE-'").replace("bid=TASK+':finite:'", "bid=TASK+':native:'")
s=s.replace("BASE+'Proofs/FiniteSource.lean'", "BASE+'Proofs/NativeProgram.lean'").replace("BASE+'Proofs/FiniteHost.lean'", "BASE+'NativeCorrect.lean'")
s=s.replace("'AddValueNative.sourceGrid'", "'AddValueNative.executeProgram'").replace("'AddValueNative.puzzle1'", "'AddValueNative.executeHost'")
s=s.replace('all active partial float-to-integer store results must be defined. Undefined cases have no native result or exception claim.','actual partial float-to-integer conversion observed before materializing a store; no arbitrary fallback, native exception or postfailure-memory claim.')
s=s.replace("['finite B1024 indices and 64bit pointer extents','fresh output, input preservation and byte frames','unconditional actual allocation metadata and view bindings','exact seven-statement AST lowering and unary source execution','active partial conversion definedness','ordinary trusted allocation/context/load/store primitives']", "['actual annotated i32_ABI signed masks and wrapped64bit source addresses','fresh output input preservation and independently derived byte frames','unconditional actual allocation metadata and extracted-pointer binding','exact original unary source execution with runtime fp32 argument10','partial conversion observation before store values','ordinary trusted allocation/context/typed-load/pure-packing/store primitives']")
s=s.replace('All22 canonical Torch storage dtypes:15 admitted,2 fnuz architecture rejects,5 first binder/signature rejects; empty, single and multiple B1024 blocks; arbitrary rank including scalar tensors.','All22 dense Torch dtype profiles:15 full backend routes,2 fnuz architecture rejects,5 binder rejects. Full nonnegative signed64 host numel with actual annotated i32 arithmetic and wrapped64bit pointers; arbitrary rank including scalar tensors.')
a=s.index("'review_basis':'PRODUCER PROPOSAL");b=s.index("'original_scope':scope}",a)
s=s[:a]+"'review_basis':'Producer proposal requiring independent root review. Complete original unary tile load, runtime fp32 value10, typed add/store and full numel grid. Actual signed32 masks and sign-extended wrapped64bit addresses; allocation, two-CUDA assertion, context/compilation/grid-parse/pointer-packing/launch outcomes. No active value/access/no-wrap premise on master; structural observations and independent final bytes. Historical bounded five-module proposal remains unregistered.',"+s[b:]
s=s.replace('Complete309 finite source outcome cases for all22 canonical dtype profiles, runtime fp32 value10, allocation before CUDA assertion, full numel/B1024 grid, typed unary source execution, unconditional metadata/view bindings, positive finite storage/permissions, independent typed results and arbitrary output tail/frame. See domain-review.json.','Complete238 full-host-domain cases for all22 dense dtype profiles. Unary source loads and runtime fp32 value10; actual i32/u64 arithmetic; independent final byte result and frame. No request safety, defined-value or no-wrap premise is imposed on the master. Observations assert no native trap, race winner, rollback or postfailure memory.')
s=s.replace('Actual Torch metadata, canonical22 storage dtypes, CUDA sm90, debug=false,Triton3.7.1; positive success requires SafeAPI and ValuesDefined; all error and empty-grid paths remain explicit.','Actual Torch metadata, all22 dense dtype profiles, full nonnegative signed64 host extent, CUDA sm90 debug=false Triton3.7.1. Access and conversion safety are observations; original CUDA assertion remains. All allocation/setup failures and returned/observed paths explicit.')
s=s.replace("'kind':'add_value_finite_exact_numeric_authority_delta'", "'kind':'add_value_native_exact_numeric_authority_delta'")
s=s.replace('309 exact source outcome cases;82 scalar ports on positive successful cases; errors and zero-grid launches require no numerical operations.','238 exact source outcome cases;41 scalar ports for loaded-word returned/undefined partitions; value-insensitive constructor-only and setup-failure cases claim no numerical intermediates.')
s=s.replace('assert len(provenance)==82 and len(rules)==15','assert len(provenance)==41 and len(rules)==15')
s=s.replace("'id':'add_value_finite_original_host_grid'", "'id':'add_value_native_i32_full_host_extent'")
s=s.replace("BASE+'Kernels.lean':['puzzle1_kernel'],BASE+'Proofs/NativeProgram.lean':['lowerBody','sourceGrid'],BASE+'Proofs/NativeScalar.lean':['source','result']", "BASE+'Proofs/NativeProgram.lean':['executeProgram','programPlan'],BASE+'Proofs/NativeScalar.lean':['source','result']")
s=s.replace("BASE+'NativeCorrect.lean':['puzzle1','launch']", "BASE+'NativeCorrect.lean':['executeHost']")
s=s.replace('Checked seven-statement lowering from unchanged source DSL AST, independently compiled original annotated Python at all22 signatures; source execution has one input pointer and runtime fp32 value10. Ordinary Torch metadata/allocation, launch context/setup and typed memory primitives trusted; no generic compiler/MMU proof or native scalar validation claim.','Exact original Python AST and all22 native profiles;14 ABI/binder/backend extent profiles; full signed32 masks and sign-extended wrapped addresses; one actual input pointer plus runtime fp32 scalar10. Ordinary Torch metadata/fresh allocation, successful mapped pointer extraction, typed loads and pure packing/store primitives trusted. No result/launch-success oracle or arbitrary-carrier readback axiom; no native scalar validation claim.')
s=s.replace('309 validate_case calls','238 validate_case calls').replace('82 exact case-operation binding bijection','41 exact case-operation binding bijection')
(W/'propose.py').write_text(s)
