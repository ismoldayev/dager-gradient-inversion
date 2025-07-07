import argparse
from html import parser
import time
import sys

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='DAGER attack')

    parser.add_argument('--neptune', type=str, help='neptune project name, leave empty to not use neptune', default=None)
    parser.add_argument('--neptune_offline', action='store_true', help='Run Neptune in offline mode')
    parser.add_argument('--label', type=str, default='name of the run', required=False)
    
    # Method and setting
    parser.add_argument('--rng_seed', type=int, default=101) 
    parser.add_argument('--dataset', choices=['cola', 'sst2', 'rte', 'rotten_tomatoes', 'stanfordnlp/imdb', 'glnmario/ECHR'], required=True)
    parser.add_argument('--task', choices=['seq_class', 'next_token_pred'], required=True)
    parser.add_argument('--pad', choices=['right', 'left'], default='right')
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('-b','--batch_size', type=int, default=1)
    parser.add_argument('--n_inputs', type=int, required=True) # val:10/20, test:100
    parser.add_argument('--start_input', type=int, default=0)
    parser.add_argument('--end_input', type=int, default=100000)

    # Model path (defaults to huggingface download, use local path if offline)
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    parser.add_argument('--finetuned_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--device_grad', type=str, default='cpu')
    parser.add_argument('--attn_implementation', type=str, default='sdpa', choices=['sdpa', 'eager'])

    parser.add_argument('--precision', type=str, default='full', choices=['8bit', 'half', 'full', 'double'])
    parser.add_argument('--parallel', type=int, default=1000)
    parser.add_argument('--grad_b', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--rank_tol', type=float, default=None) 
    parser.add_argument('--rank_cutoff', type=int, default=20)
    parser.add_argument('--l1_span_thresh', type=float, default=1e-5) 
    parser.add_argument('--l2_span_thresh', type=float, default=1e-3) 
    parser.add_argument('--l1_filter', choices=['maxB', 'all'], required=True)
    parser.add_argument('--l2_filter', choices=['overlap', 'non-overlap'], required=True)
    parser.add_argument('--distinct_thresh', type=float, default=0.7)
    parser.add_argument('--max_ids', type=int, default=-1)
    parser.add_argument('--maxC', type=int, default=10000000) 
    parser.add_argument('--reduce_incorrect', type=int, default=0)
    parser.add_argument('--n_incorrect', type=int, default=None)
    
    # FedAVG
    parser.add_argument('--algo', type=str, default='sgd', choices=['sgd', 'fedavg'])
    parser.add_argument('--avg_epochs', type=int, default=None)
    parser.add_argument('--b_mini', type=int, default=None)
    parser.add_argument('--avg_lr', type=float, default=None)
    parser.add_argument('--dist_norm', type=str, default='l2', choices=['l1', 'l2'])
    
    #DP
    parser.add_argument('--defense_noise', type=float, default=None) # add noise to true grads
    parser.add_argument('--max_len', type=int, default=1e10) 
    parser.add_argument('--p1_std_thrs', type=float, default=5)
    parser.add_argument('--l2_std_thrs', type=float, default=5)
    parser.add_argument('--dp_l2_filter', type=str, default='maxB', choices=['maxB', 'outliers'])
    parser.add_argument('--defense_pct_mask', type=float, default=None) # mask some percentage of gradients
    
    #Dropout
    parser.add_argument('--grad_mode', type=str, default='eval', choices=['eval', 'train'])
    
    #Rebuttal experiments
    parser.add_argument('--hidden_act', type=str, default=None)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'mse'])
    
    #LoRA
    parser.add_argument('--train_method', type=str, default='full', choices=['full', 'lora'])
    parser.add_argument('--lora_r', type=int, default=None)

    # Q Optimization (legacy - kept for compatibility)
    q_opt_group = parser.add_argument_group('Q Optimization Attack Arguments (Legacy)')
    q_opt_group.add_argument('--num_steps_q_opt', type=int, default=1000, 
                             help='Number of optimization steps for q parameters')
    q_opt_group.add_argument('--lr_q_opt', type=float, default=0.01, 
                             help='Learning rate for q parameter optimization')

    q_opt_group.add_argument('--initialize_q_from_gt', action='store_true',
                             help='DEBUG: Initialize q for the first token from GT embedding.')
    q_opt_group.add_argument('--gt_init_mse_threshold', type=float, default=0.2,
                             help='MSE threshold for using GT-derived q for init.')
    q_opt_group.add_argument('--max_seq_len_to_recover', type=int, default=1,
                             help='Maximum sequence length to attempt to recover. -1 for full original length.')
    q_opt_group.add_argument('--num_steps_q_opt_per_token', type=int, default=None,
                             help='Optimization steps per token. Defaults to num_steps_q_opt if None.')
    q_opt_group.add_argument('--teacher_force_recovered_tokens', type=lambda x: (str(x).lower() == 'true'), default=True,
                             help='Use canonical embedding of recovered token for next step context (True/False).')
    q_opt_group.add_argument('--apply_norm_heuristic', action='store_true',
                             help='Apply norm heuristic during optimization.')
    
    # Raw Embedding Optimization 
    raw_opt_group = parser.add_argument_group('Raw Embedding Optimization Attack Arguments')
    raw_opt_group.add_argument('--num_steps_x_opt', type=int, default=None,
                              help='Number of optimization steps for raw embedding parameters. Defaults to num_steps_q_opt if None.')
    raw_opt_group.add_argument('--lr_x_opt', type=float, default=None,
                              help='Learning rate for raw embedding optimization. Defaults to lr_q_opt if None.')
    raw_opt_group.add_argument('--initialize_x_from_gt', action='store_true',
                              help='Initialize raw embeddings from ground truth embeddings.')
    raw_opt_group.add_argument('--use_batch_reinit', action='store_true',
                             help='Enable batch reinitialization of raw embedding optimization.')
    raw_opt_group.add_argument('--num_reinit', type=int, default=5,
                             help='Number of random initializations to try in batch reinit.')
    raw_opt_group.add_argument('--enable_sanity_checks', action='store_true',
                              help='Enable running and printing of full sanity checks')
    # Orthogonal context option: exclude previous token direction when building Ck basis
    raw_opt_group.add_argument('--use_orthogonal_context', action='store_true',
                              help='Use orthogonal basis for context by excluding previous token direction')
    # Diversity regularization: penalize similarity to previous tokens
    raw_opt_group.add_argument('--add_diversity_loss', action='store_true',
                              help='Add diversity regularization during optimization')
    # Debug comparison: compare GT vs closest token losses
    raw_opt_group.add_argument('--compare_gt_vs_closest_token_losses', action='store_true',
                              help='Compare ground-truth vs closest token losses for debugging')

    parser.add_argument('--debug_fix_eff_rank_Ck', type=int, default=-1,
                         help='DEBUG: Fix effective rank for Ck_basis. -1 for auto. Values > 0 override auto rank for Ck.')

    if argv is None:
       argv = sys.argv[1:]
    args=parser.parse_args(argv)

    if args.num_steps_q_opt_per_token is None:
        args.num_steps_q_opt_per_token = args.num_steps_q_opt
        
    if args.n_incorrect is None:
        args.n_incorrect = args.batch_size

    if args.neptune is not None:
        import neptune.new as neptune
        assert('label' in args)
        nep_par = { 'project':f"{args.neptune}", 'source_files':["*.py"] } 
        if args.neptune_offline:
            nep_par['mode'] = 'offline'
            args.neptune_id = 'DAG-0'

        run = neptune.init( **nep_par )
        args_dict = vars(args)
        run[f"parameters"] = args_dict
        args.neptune = run
        if not args.neptune_offline:
            print('waiting...')
            start_wait=time.time()
            args.neptune.wait()
            print('waited: ',time.time()-start_wait)
            args.neptune_id = args.neptune['sys/id'].fetch()
        print( '\n\n\nArgs:', *argv, '\n\n\n' ) 
    return args
