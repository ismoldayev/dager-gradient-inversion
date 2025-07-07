# optimize_raw_embed.py
print("--- SCRIPT optimize_raw_embed.py: STARTED ---")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from args_factory import get_args
from utils.data import TextDataset
from utils.models import ModelWrapper
from sanity_checks import (
	project_onto_subspace_lstsq,
	compute_span_distance_sq,
	calculate_loss_directly_from_true_embeddings_FULL_CONTEXT,
	compare_gt_vs_closest_token_losses,
)

print("--- All initial imports seem OK ---")


def get_input_basis(
	grad: torch.Tensor,
	eff_rank: int,
	emb_size: int,
	device,
) -> torch.Tensor:
	"""
	Always return a basis that spans R^{emb_size}.
	"""
	U, S, Vh = torch.linalg.svd(grad.cpu().float(), full_matrices=False)
	# print(f"  SVD shapes: U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
	r = min(eff_rank, (S > 0).sum().item())

	if grad.shape[1] == emb_size:  # (out,  emb_size)
		# columns already live in R^{emb_size}
		basis = Vh.T[:, :r]  # (emb_size, r) - just V
	elif grad.shape[0] == emb_size:  # (emb_size,  out)
		# rows live in R^{emb_size}
		basis = U[:, :r]  # (emb_size, r) - just U
	else:
		raise ValueError(f"Weight dims {grad.shape} do not contain emb_size={emb_size}")

	return basis.T.to(device).float()  # (eff_rank x emb_size)


def calculate_loss_for_current_x_iterative(
	x_raw_row_params,  # (1, emb_dim) OPTIM VARIABLE
	fixed_prev_raw_embeds_rows,  # list[(1, emb_dim)]
	current_token_pos_idx,
	Ck_basis_list,
	model_wrapper: ModelWrapper,
	alpha_k_weights,
	args,
	debug_step=None,  # Add debug step parameter
):
	"""
	Returns scalar loss for current raw-embedding variable.
	"""
	#    shapes: each row (1, emb_dim) → cat → (seq_len, emb_dim) → (1, seq_len, emb_dim)
	H_raw_0 = torch.cat(
		fixed_prev_raw_embeds_rows + [x_raw_row_params], dim=0
	).unsqueeze(
		0
	)  # (1, seq_len, emb_dim)

	total_loss = torch.tensor(0.0, device=args.device, dtype=torch.float32)
	layer_losses = []

	H_raw_k = H_raw_0
	for k in range(len(Ck_basis_list)):
		# --- span check for token j at layer k ---
		x_LN_k_j_col = model_wrapper.get_input_to_block_after_ln(H_raw_k, k)[
			0, current_token_pos_idx, :
		].unsqueeze(
			1
		)  # (n,1)
		dist_sq = compute_span_distance_sq(x_LN_k_j_col, Ck_basis_list[k])
		alpha = alpha_k_weights[k] if k < len(alpha_k_weights) else alpha_k_weights[0]
		layer_loss = alpha * dist_sq
		total_loss += layer_loss
		layer_losses.append(layer_loss.item())


		if k < len(Ck_basis_list) - 1:
			H_raw_k = model_wrapper.forward_sequence_through_block(
				H_raw_k,
				block_idx=k,
				current_seq_len=H_raw_k.shape[1],
				device=args.device,
				already_ln=False,  # keep LN!
			)

	diversity_loss = torch.tensor(0.0, device=args.device, dtype=torch.float32)
	if args.add_diversity_loss and len(fixed_prev_raw_embeds_rows) > 0 and current_token_pos_idx > 0:
		current_emb = x_raw_row_params.squeeze()
		for prev_emb_row in fixed_prev_raw_embeds_rows:
			prev_emb = prev_emb_row.squeeze()
			# Compute cosine similarity and penalize high similarity
			cos_sim = torch.nn.functional.cosine_similarity(
				current_emb.unsqueeze(0), prev_emb.unsqueeze(0), dim=1
			)
			cos_sim_scalar = cos_sim.squeeze().abs() ** 2
			diversity_loss = diversity_loss + 0.1 * cos_sim_scalar 
		total_loss += diversity_loss

	# Debug output for first few steps
	if debug_step is not None and debug_step < 5:
		print(
			f"    Debug step {debug_step}, pos {current_token_pos_idx}: Layer losses = {layer_losses}, Diversity = {diversity_loss.item():.6e}, Total = {total_loss.item():.6e}"
		)

	return total_loss


def optimize_raw_embedding_for_token_at_pos(
	fixed_prev_raw_embeds_rows,  # list[(1,n)]
	current_token_pos_idx,
	Ck_basis_list,
	model_wrapper,
	args,
	num_steps,
	lr,
	alpha_k_weights,
	true_raw_embed_row=None,
	apply_norm_heuristic=False,
):
	print(
		f"\n--- Optimizing raw embedding for token at position {current_token_pos_idx} ---"
	)

	emb_dim = model_wrapper.emb_size
	x_raw_row_params = torch.randn(
		1, emb_dim, requires_grad=True, device=args.device, dtype=torch.float32
	)
	optimizer = torch.optim.AdamW([x_raw_row_params], lr=lr, weight_decay=1e-4)

	# Initialize with true raw embedding if available
	if true_raw_embed_row is not None:
		x_raw_row_params.data = true_raw_embed_row.clone().float()
		print(
			f"  Initialized with true raw embedding for token {current_token_pos_idx}"
		)

	print(
		f"  Starting optimization for raw_embed_{current_token_pos_idx} (1 x {emb_dim}) with {num_steps} steps, lr={lr}"
	)
	print(f"  Norm heuristic during optimization: {apply_norm_heuristic}")
	losses_over_steps = []
	initial_model_training_state_opt = model_wrapper.model.training
	model_wrapper.model.eval()

	for step in range(num_steps):
		optimizer.zero_grad()
		loss = calculate_loss_for_current_x_iterative(
			x_raw_row_params,
			fixed_prev_raw_embeds_rows,
			current_token_pos_idx,
			Ck_basis_list,
			model_wrapper,
			alpha_k_weights,
			args,
			debug_step=(
				step if step < 5 else None
			),  # Pass debug step for first few iterations
		)
		if loss.requires_grad:
			loss.backward(retain_graph=True)
			optimizer.step()
			# Norm heuristic: scale raw embedding to match true embedding norm to prevent collapse
			if apply_norm_heuristic and true_raw_embed_row is not None:
				with torch.no_grad():
					rec_norm = torch.norm(x_raw_row_params)
					true_norm = torch.norm(true_raw_embed_row)
					if rec_norm > 0:
						scale = true_norm / rec_norm
						x_raw_row_params.data.mul_(scale)
		else:
			print(
				f"  Token {current_token_pos_idx}, Step {step:4d}: No grad. Skipping."
			)
			break

		losses_over_steps.append(loss.item())
		if (
			step % (num_steps // 10 if num_steps >= 100 else 20) == 0
			or step == num_steps - 1
		):
			grad_norm_str = (
				f", GradNorm: {torch.norm(x_raw_row_params.grad).item():.2e}"
				if x_raw_row_params.grad is not None
				else ""
			)

			# Debug: Show current closest token every few steps for first few positions
			if current_token_pos_idx < 3 and step % (num_steps // 5) == 0:
				with torch.no_grad():
					closest_id, closest_str = recover_token_from_embedding(
						x_raw_row_params.detach(),
						current_token_pos_idx,
						model_wrapper,
						args,
						top_k=1,
					)
					print(
						f"  Token {current_token_pos_idx}, Step {step:4d}, Loss: {loss.item():.3e}{grad_norm_str}, Closest: '{closest_str}'"
					)
			else:
				print(
					f"  Token {current_token_pos_idx}, Step {step:4d}, Loss: {loss.item():.3e}{grad_norm_str}"
				)

		# early stopping if loss gets very small
		if loss.item() < 1e-6:
			print(f"  Early stopping at step {step} due to very small loss")
			break

	if initial_model_training_state_opt:
		model_wrapper.model.train()
	else:
		model_wrapper.model.eval()

	# Plot loss curve for first token
	# todo: fix this because it doesn't work rn
	if args.save_loss_plot and current_token_pos_idx == 0:
		plt.figure(figsize=(10, 6))
		plt.plot(losses_over_steps)
		plt.xlabel("Optimization Step")
		plt.ylabel("Total Loss")
		plt.title(
			f"Loss Curve (Token {current_token_pos_idx}, Raw Embedding Opt, NormHeur:{apply_norm_heuristic}, LR:{lr})"
		)
		plt.yscale(
			"log"
			if len(losses_over_steps) > 1
			and max(losses_over_steps, default=0)
			/ max(min(losses_over_steps, default=1e-9), 1e-9)
			> 100
			else "linear"
		)
		plt.grid(True)
		plot_filename = f"loss_curve_x_raw_tok{current_token_pos_idx}_{args.dataset}_{args.model_path.replace('/','_')}.png"
		try:
			plt.savefig(plot_filename)
			print(f"  Loss curve saved to {plot_filename}")
			plt.close()
		except Exception as e:
			print(f"  Error saving loss plot: {e}")

	# Return the learned raw embedding row
	return x_raw_row_params.detach()


def recover_token_from_embedding(
	reconstructed_embedding_row_vec, position_idx, model_wrapper, args, top_k=5
):
	reconstructed_embedding_row_vec = reconstructed_embedding_row_vec.float().to(
		args.device
	)
	target_embeddings_for_vocab = None
	with torch.no_grad():
		if model_wrapper.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			word_embeddings_table = (
				model_wrapper.model.transformer.wte.weight.data.float()
			)
			pos_embedding = (
				model_wrapper.model.transformer.wpe.weight.data[position_idx, :]
				.float()
				.unsqueeze(0)
			)
			# Use raw embeddings for comparison (word + position)
			target_embeddings_for_vocab = word_embeddings_table + pos_embedding
		elif model_wrapper.has_rope():
			target_embeddings_for_vocab = (
				model_wrapper.model.model.embed_tokens.weight.data.float()
			)
		elif model_wrapper.is_bert():
			word_emb = (
				model_wrapper.model.bert.embeddings.word_embeddings.weight.data.float()
			)
			pos_emb = (
				model_wrapper.model.bert.embeddings.position_embeddings.weight.data[
					position_idx, :
				]
				.float()
				.unsqueeze(0)
			)
			type_emb_0 = (
				model_wrapper.model.bert.embeddings.token_type_embeddings.weight.data[
					0, :
				]
				.float()
				.unsqueeze(0)
			)
			target_embeddings_for_vocab = word_emb + pos_emb + type_emb_0
		else:
			print("Token recovery method not defined for model.")
			return None, "Error"
	if target_embeddings_for_vocab is None:
		return None, "Error"

	# reconstructed embedding must have the right shape for cdist: (1, embed_dim)
	if reconstructed_embedding_row_vec.ndim == 1:
		reconstructed_embedding_row_vec = reconstructed_embedding_row_vec.unsqueeze(0)
	elif reconstructed_embedding_row_vec.shape[0] > 1:
		if reconstructed_embedding_row_vec.shape[1] == 1:
			reconstructed_embedding_row_vec = reconstructed_embedding_row_vec.T
		else:
			reconstructed_embedding_row_vec = reconstructed_embedding_row_vec[:1, :]

	# target_embeddings_for_vocab should be (vocab_size, embed_dim)
	distances = torch.cdist(
		reconstructed_embedding_row_vec, target_embeddings_for_vocab, p=2
	).squeeze(
		0
	)  # remove batch dimension: (vocab_size,)

	sorted_distances, top_indices = torch.sort(distances)

	print(f"  Pos {position_idx} Top {top_k} L2 (raw): ", end="")
	for i in range(min(top_k, len(top_indices))):
		token_id = top_indices[i].item()
		dist = sorted_distances[i].item()
		print(
			f"'{model_wrapper.tokenizer.decode([token_id])}'({dist:.2f}) ",
			end="",
		)
	print()

	# compute L2 distances after layer normalization
	reconstructed_ln = model_wrapper.get_input_to_block_after_ln(
		reconstructed_embedding_row_vec.unsqueeze(0), 0
	).squeeze(
		0
	)  # (1, embed_dim)

	# compute LN embeddings in batches
	batch_size = 1000
	vocab_size = target_embeddings_for_vocab.shape[0]
	target_embeddings_ln_list = []

	for start_idx in range(0, vocab_size, batch_size):
		end_idx = min(start_idx + batch_size, vocab_size)
		target_batch = target_embeddings_for_vocab[
			start_idx:end_idx
		]  # (batch_size, embed_dim)

		# Apply LN to the batch
		target_batch_ln = model_wrapper.get_input_to_block_after_ln(
			target_batch.unsqueeze(0), 0
		).squeeze(
			0
		)  # (batch_size, embed_dim)
		target_embeddings_ln_list.append(target_batch_ln)

	target_embeddings_ln = torch.cat(
		target_embeddings_ln_list, dim=0
	)  # (vocab_size, embed_dim)

	distances_ln = torch.cdist(
		reconstructed_ln, target_embeddings_ln, p=2
	).squeeze()  # Remove any extra dimensions: (vocab_size,)

	# distances_ln must be 1D
	if distances_ln.ndim != 1:
		distances_ln = distances_ln.flatten()

	sorted_distances_ln, top_indices_ln = torch.sort(distances_ln)

	print(f"  Pos {position_idx} Top {top_k} L2 (LN): ", end="")
	for i in range(min(top_k, len(top_indices_ln))):
		token_id = top_indices_ln[i].item()
		dist = sorted_distances_ln[i].item()
		print(
			f"'{model_wrapper.tokenizer.decode([token_id])}'({dist:.4f}) ",
			end="",
		)
	print()

	cosine_similarities = torch.nn.functional.cosine_similarity(
		reconstructed_embedding_row_vec, target_embeddings_for_vocab, dim=1
	)
	sorted_cos_sim, top_cos_indices = torch.sort(cosine_similarities, descending=True)
	print(f"  Pos {position_idx} Top {top_k} CosSim (raw): ", end="")
	for i in range(min(top_k, len(top_cos_indices))):
		token_id = top_cos_indices[i].item()
		cos_sim = sorted_cos_sim[i].item()
		print(
			f"'{model_wrapper.tokenizer.decode([token_id])}'({cos_sim:.4f}) ",
			end="",
		)
	print()

	best_l2_token_id = top_indices[0].item() if len(top_indices) > 0 else None
	best_l2_token_str = (
		model_wrapper.tokenizer.decode([best_l2_token_id])
		if best_l2_token_id is not None
		else "N/A"
	)
	return best_l2_token_id, best_l2_token_str


def main_raw_embedding_attack():
	args = get_args()
	for arg_name, default_val in [
		("save_loss_plot", True),
		("max_seq_len_to_recover", 1),
		("teacher_force_recovered_tokens", True),
		("debug_fix_eff_rank_Ck", -1),
		("num_steps_x_opt", None),
		("lr_x_opt", None),
		("use_orthogonal_context", False),
		("add_diversity_loss", False),
		(
			"compare_gt_vs_closest_token_losses",
			False,
		),  # for debugging
		("use_batch_reinit", False),
		("num_reinit", 5),
	]:
		if not hasattr(args, arg_name):
			setattr(args, arg_name, default_val)

	# Set default values for optimization parameters
	if args.num_steps_x_opt is None:
		args.num_steps_x_opt = getattr(args, "num_steps_q_opt", 600)
	if args.lr_x_opt is None:
		args.lr_x_opt = getattr(args, "lr_q_opt", 0.01)

	print(f"Run args: {args}")
	print(f"\n--- Feature Configuration ---")
	print(f"  Use orthogonal context: {args.use_orthogonal_context}")
	print(f"  Add diversity loss: {args.add_diversity_loss}")
	print(
		f"  Compare GT vs closest token losses: {args.compare_gt_vs_closest_token_losses}"
	)
	print(f"  Teacher force recovered tokens: {args.teacher_force_recovered_tokens}")
	print(f"  Use batch reinitialization: {args.use_batch_reinit}")
	if args.use_batch_reinit:
		print(f"  Number of reinitializations: {args.num_reinit}")
	print(f"--- End Feature Configuration ---\n")
	device = torch.device(args.device)
	dataset = TextDataset(
		args.device,
		args.dataset,
		args.split,
		n_inputs=1,
		batch_size=args.batch_size,
		cache_dir=args.cache_dir,
	)
	model_wrapper = ModelWrapper(args)

	print(f"\n--- Running Iterative Raw Embedding Optimization Attack ---")
	sample_sequences, sample_labels = dataset[0]
	print(f"\nInput text (sample_sequences[0]): '{sample_sequences[0]}'")
	orig_batch_tokenized = model_wrapper.tokenizer(
		sample_sequences,
		padding=True,
		truncation=True,
		max_length=min(model_wrapper.tokenizer.model_max_length, 512),
		return_tensors="pt",
	).to(args.device)

	actual_token_ids_full = orig_batch_tokenized["input_ids"][0].tolist()
	actual_attention_mask_full = orig_batch_tokenized["attention_mask"][0].tolist()
	try:
		original_seq_len = (
			actual_attention_mask_full.index(0)
			if 0 in actual_attention_mask_full
			else len(actual_attention_mask_full)
		)
	except ValueError:
		original_seq_len = len(actual_attention_mask_full)
	actual_token_ids = actual_token_ids_full[:original_seq_len]
	if original_seq_len == 0:
		print("Error: Original sample is empty. Exiting.")
		return

	max_len_to_recover = (
		original_seq_len
		if args.max_seq_len_to_recover == -1
		else min(args.max_seq_len_to_recover, original_seq_len)
	)
	print(f"Actual token IDs (len {original_seq_len}): {actual_token_ids}")
	print(f"Attempting to recover up to {max_len_to_recover} tokens.")

	true_raw_embeddings_full_sequence_cols = []
	for pos_idx in range(original_seq_len):
		emb_col = model_wrapper.get_canonical_embedding_for_id_pos(
			actual_token_ids[pos_idx], pos_idx, args.device
		).T
		true_raw_embeddings_full_sequence_cols.append(emb_col.float())

	H_raw_true = torch.stack(
		[emb.squeeze(1) for emb in true_raw_embeddings_full_sequence_cols], dim=0
	).unsqueeze(
		0
	)  # (1, seq_len, 768)
	true_X_LN0_full = model_wrapper.get_input_to_block_after_ln(H_raw_true, 0)

	all_true_grads_list_raw = model_wrapper.compute_grads(
		orig_batch_tokenized, sample_labels
	)
	all_param_names = [name for name, _ in model_wrapper.model.named_parameters()]
	all_true_grads_dict = {
		name: grad
		for name, grad in zip(all_param_names, all_true_grads_list_raw)
		if grad is not None
	}

	grads_for_projections_to_use = []
	param_names_of_Wk_used_for_Ck = []
	captured_X_LN_orig_for_these_Wk = []

	projection_param_names_template = {
		"gpt2": "transformer.h.{}.attn.c_attn.weight",
		"openai-community/gpt2-large": "transformer.h.{}.attn.c_attn.weight",
	}
	param_template = projection_param_names_template.get(
		args.model_path, "transformer.h.{}.attn.c_attn.weight"
	)
	num_constraints_to_setup = args.n_layers + 1
	print(
		f"DEBUG: Setting up {num_constraints_to_setup} constraints (layers 0 to {args.n_layers})."
	)
	for k_idx in range(num_constraints_to_setup):
		Wk_name_candidate = param_template.format(k_idx)
		grad_dL_dWk = all_true_grads_dict.get(Wk_name_candidate)
		if (
			grad_dL_dWk is None
			and "gpt2" in args.model_path
			and Wk_name_candidate.endswith("c_attn.weight")
		):
			Wk_name_alt = f"transformer.h.{k_idx}.attn.c_proj.weight"
			print(
				f"DEBUG: Grad for {Wk_name_candidate} not found, trying {Wk_name_alt}"
			)
			grad_dL_dWk = all_true_grads_dict.get(Wk_name_alt)
			if grad_dL_dWk is not None:
				print(
					f"ERROR_SETUP: Using alt grad {Wk_name_alt}. Captured activation is for {Wk_name_candidate}'s input. This will cause mismatch for Sanity 2.A."
				)
				pass
		if grad_dL_dWk is not None:
			activation_key_for_hook = param_template.format(k_idx)
			X_LN_orig_k = model_wrapper.captured_original_ln_activations.get(
				activation_key_for_hook
			)
			if X_LN_orig_k is not None:
				grads_for_projections_to_use.append(grad_dL_dWk)
				param_names_of_Wk_used_for_Ck.append(Wk_name_candidate)
				captured_X_LN_orig_for_these_Wk.append(
					X_LN_orig_k.to(args.device).float()
				)
				# print(
				#     f"  Collected grad (from {Wk_name_candidate}) and X_LN_orig (for input to {activation_key_for_hook}) for Ck_basis[{len(grads_for_projections_to_use)-1}]"
				# )
			else:
				print(
					f"Warning: Grad found for {Wk_name_candidate}, but no captured X_LN_orig for key {activation_key_for_hook}. Skipping Ck_basis for layer {k_idx}."
				)
		else:
			print(
				f"Warning: No grad found for Wk candidate {Wk_name_candidate} (or its fallbacks). Skipping Ck_basis for layer {k_idx}."
			)
	if not grads_for_projections_to_use:
		print(
			"Error: No projection grads (for which X_LN_orig was also captured) collected. Exiting."
		)
		if args.neptune:
			args.neptune.stop()
		return

	# Build Ck basis list for constraints
	Ck_basis_list_for_opt = []
	for idx_ck, grad_Wk_tensor_loop in enumerate(grads_for_projections_to_use):
		_Uk_loop, _Sk_diag_loop, _Vhk_loop = torch.linalg.svd(
			grad_Wk_tensor_loop.cpu().float()
		)
		rank_tol_svd_k = args.rank_tol if args.rank_tol is not None else 1e-7
		eff_rank_ck_k_auto = max(
			1,
			min(
				torch.sum(_Sk_diag_loop > rank_tol_svd_k).item(),
				grad_Wk_tensor_loop.shape[0],
				grad_Wk_tensor_loop.shape[1],
			),
		)
		current_eff_rank_Ck = min(
			eff_rank_ck_k_auto, model_wrapper.emb_size - (args.rank_cutoff or 0)
		)
		if current_eff_rank_Ck == 0 and eff_rank_ck_k_auto > 0:
			current_eff_rank_Ck = 1
		if hasattr(args, "debug_fix_eff_rank_Ck") and args.debug_fix_eff_rank_Ck > 0:
			user_fixed_rank = args.debug_fix_eff_rank_Ck
			possible_rank_from_svd = (_Sk_diag_loop > 0).sum().item()
			max_possible_rank = min(
				model_wrapper.emb_size,
				grad_Wk_tensor_loop.shape[0],
				grad_Wk_tensor_loop.shape[1],
				possible_rank_from_svd,
			)
			current_eff_rank_Ck = min(user_fixed_rank, max_possible_rank)

		Ck_basis_k = get_input_basis(
			grad_Wk_tensor_loop,
			current_eff_rank_Ck,
			model_wrapper.emb_size,
			args.device,
		)
		Ck_basis_list_for_opt.append(Ck_basis_k)

		# Analyze the quality of the basis
		basis_rank = torch.linalg.matrix_rank(Ck_basis_k)
		basis_condition_number = torch.linalg.cond(Ck_basis_k @ Ck_basis_k.T)

		# print(
		#     f"  Ck_basis[{idx_ck}] (from grad '{param_names_of_Wk_used_for_Ck[idx_ck]}'): eff_rank_Ck = {current_eff_rank_Ck}, shape {Ck_basis_k.shape}, actual_rank = {basis_rank}, cond = {basis_condition_number:.2e}"
		# )

	# Debug: Check similarity between basis vectors across layers
	# removed because it's wrong
	
	if not Ck_basis_list_for_opt:
		print("Error: Ck_basis_list_for_opt is empty. Exiting.")
		if args.neptune:
			args.neptune.stop()
		return
	if args.enable_sanity_checks and true_raw_embeddings_full_sequence_cols:
		calculate_loss_directly_from_true_embeddings_FULL_CONTEXT(
			true_raw_embeddings_full_sequence_cols,
			original_seq_len,
			len(Ck_basis_list_for_opt),
			Ck_basis_list_for_opt,
			model_wrapper,
			[1.0] * len(Ck_basis_list_for_opt),
			args,
			true_X_LN0_full,
			captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis=captured_X_LN_orig_for_these_Wk,
		)
		from sanity_checks import check_uniform_shift_invariance
		check_uniform_shift_invariance(
			true_raw_embeddings_full_sequence_cols,
			Ck_basis_list_for_opt,
			model_wrapper,
			alpha_k_weights=[1.0] * len(Ck_basis_list_for_opt),
			args=args,
			shift_value=123.456,
		)
	recovered_token_ids = []
	recovered_token_strs = []
	fixed_prev_raw_embeds_rows = []

	for j_token_pos in range(max_len_to_recover):
		true_raw_embed_for_this_token_row = None
		if j_token_pos < len(true_raw_embeddings_full_sequence_cols):
			true_raw_embed_for_this_token_row = true_raw_embeddings_full_sequence_cols[
				j_token_pos
			].T

		# create orthogonal basis if requested and we have previous context
		Ck_basis_list_to_use = Ck_basis_list_for_opt
		if (
			args.use_orthogonal_context
			and j_token_pos > 0
			and len(fixed_prev_raw_embeds_rows) > 0
		):
			print(f"\n--- Creating Orthogonal Basis for Token {j_token_pos} ---")
			Ck_basis_list_to_use = []

			for layer_idx, original_basis in enumerate(Ck_basis_list_for_opt):
				# Use the LN embedding of the previous token as the direction to exclude
				prev_token_ln_embed = model_wrapper.get_input_to_block_after_ln(
					fixed_prev_raw_embeds_rows[-1].unsqueeze(0).unsqueeze(0), layer_idx
				).squeeze()  # (emb_dim,)

				orthogonal_basis = create_orthogonal_basis_excluding_direction(
					original_basis, prev_token_ln_embed, args.device
				)
				Ck_basis_list_to_use.append(orthogonal_basis)
				print(
					f"  Layer {layer_idx}: Original basis {original_basis.shape} -> Orthogonal basis {orthogonal_basis.shape}"
				)

		# Choose optimization method based on args
		if args.use_batch_reinit:
			reconstructed_raw_row = optimize_raw_embedding_batch_reinit(
				fixed_prev_raw_embeds_rows,
				j_token_pos,
				Ck_basis_list_to_use,
				model_wrapper,
				args,
				args.num_steps_x_opt,
				args.lr_x_opt,
				alpha_k_weights=[1.0] * len(Ck_basis_list_to_use),
				true_raw_embed_row=(
					true_raw_embed_for_this_token_row if args.initialize_x_from_gt else None
				),
				apply_norm_heuristic=args.apply_norm_heuristic,
				num_reinit=args.num_reinit,
			)
		else:
			reconstructed_raw_row = optimize_raw_embedding_for_token_at_pos(
				fixed_prev_raw_embeds_rows,
				j_token_pos,
				Ck_basis_list_to_use,
				model_wrapper,
				args,
				args.num_steps_x_opt,
				args.lr_x_opt,
				alpha_k_weights=[1.0] * len(Ck_basis_list_to_use),
				true_raw_embed_row=(
					true_raw_embed_for_this_token_row if args.initialize_x_from_gt else None
				),
				apply_norm_heuristic=args.apply_norm_heuristic,
			)

		# for token recovery, we need to compare raw embeddings against raw vocabulary embeddings
		rec_id, rec_str = recover_token_from_embedding(
			reconstructed_raw_row, j_token_pos, model_wrapper, args
		)
		if rec_id is None:
			print(f"Token recovery failed for pos {j_token_pos}. Stopping.")
			break
		recovered_token_ids.append(rec_id)
		recovered_token_strs.append(rec_str)
		actual_str = (
			model_wrapper.tokenizer.decode([actual_token_ids[j_token_pos]])
			if j_token_pos < len(actual_token_ids)
			else "N/A"
		)
		print(
			f"Recovered token {j_token_pos}: ID={rec_id}, Str='{rec_str}' (Actual: Str='{actual_str}')"
		)

		# compare GT vs closest token losses if requested
		if args.compare_gt_vs_closest_token_losses and j_token_pos < len(
			actual_token_ids
		):
			compare_gt_vs_closest_token_losses(
				actual_token_ids[j_token_pos],
				rec_id,
				j_token_pos,
				Ck_basis_list_to_use,
				model_wrapper,
				[1.0] * len(Ck_basis_list_to_use),
				args,
			)

		# get true raw embedding for comparison
		true_raw_embed_for_this_token_row = (
			true_raw_embeddings_full_sequence_cols[j_token_pos].T
			if j_token_pos < len(true_raw_embeddings_full_sequence_cols)
			else None
		)

		# raw embedding comparisons
		if true_raw_embed_for_this_token_row is not None:
			raw_mse = torch.nn.functional.mse_loss(
				reconstructed_raw_row.squeeze(),
				true_raw_embed_for_this_token_row.squeeze(),
			).item()
			raw_cos_sim = torch.nn.functional.cosine_similarity(
				reconstructed_raw_row.squeeze(),
				true_raw_embed_for_this_token_row.squeeze(),
				dim=0,
			).item()
			raw_l2_dist = torch.norm(
				reconstructed_raw_row.squeeze()
				- true_raw_embed_for_this_token_row.squeeze()
			).item()
			print(f"  Raw embedding MSE (reconstructed vs true): {raw_mse:.6f}")
			print(
				f"  Raw embedding cosine similarity (reconstructed vs true): {raw_cos_sim:.6f}"
			)
			print(
				f"  Raw embedding L2 distance (reconstructed vs true): {raw_l2_dist:.6f}"
			)

			# check mean and std of embeddings
			recon_raw = reconstructed_raw_row.squeeze()
			true_raw = true_raw_embed_for_this_token_row.squeeze()
			print(
				f"  Reconstructed raw embed - mean: {recon_raw.mean().item():.6f}, std: {recon_raw.std().item():.6f}"
			)
			print(
				f"  True raw embed - mean: {true_raw.mean().item():.6f}, std: {true_raw.std().item():.6f}"
			)
			print(
				f"  Mean difference: {(recon_raw.mean() - true_raw.mean()).item():.6f}"
			)
			print(f"  Std difference: {(recon_raw.std() - true_raw.std()).item():.6f}")

		# compute and display MSE between reconstructed and true LN embedding
		true_xj_LN_col_gt = true_X_LN0_full[0, j_token_pos, :].unsqueeze(0)  # (1, 768)
		# convert reconstructed raw to LN for comparison
		reconstructed_LN_row = model_wrapper.get_input_to_block_after_ln(
			reconstructed_raw_row.unsqueeze(0).unsqueeze(0), 0
		).squeeze(
			0
		)  # (1, 768)
		mse_recon_gt = torch.nn.functional.mse_loss(
			reconstructed_LN_row.squeeze(), true_xj_LN_col_gt.squeeze()
		).item()
		print(
			f"  MSE (reconstructed vs true LN embed) at pos {j_token_pos}: {mse_recon_gt:.6f}"
		)
		# display the cosine similarity too
		cos_sim_recon_gt = torch.nn.functional.cosine_similarity(
			reconstructed_LN_row.squeeze(),
			true_xj_LN_col_gt.squeeze(),
			dim=0,
		).item()
		print(
			f"  Cosine similarity (reconstructed vs true LN embed) at pos {j_token_pos}: {cos_sim_recon_gt:.6f}"
		)

		# check mean and std of LN embeddings
		recon_ln = reconstructed_LN_row.squeeze()
		true_ln = true_xj_LN_col_gt.squeeze()
		print(
			f"  Reconstructed LN embed - mean: {recon_ln.mean().item():.6f}, std: {recon_ln.std().item():.6f}"
		)
		print(
			f"  True LN embed - mean: {true_ln.mean().item():.6f}, std: {true_ln.std().item():.6f}"
		)
		print(f"  LN Mean difference: {(recon_ln.mean() - true_ln.mean()).item():.6f}")
		print(f"  LN Std difference: {(recon_ln.std() - true_ln.std()).item():.6f}")

		# debug: Show norms to understand the scale difference
		print(
			f"  Reconstructed raw embed norm: {torch.norm(reconstructed_raw_row).item():.4f}"
		)
		print(
			f"  Reconstructed LN embed norm: {torch.norm(reconstructed_LN_row).item():.4f}"
		)
		if true_raw_embed_for_this_token_row is not None:
			print(
				f"  True raw embed norm: {torch.norm(true_raw_embed_for_this_token_row).item():.4f}"
			)
		print(f"  True LN embed norm: {torch.norm(true_xj_LN_col_gt).item():.4f}")

		# Teacher forcing: use true raw embedding or reconstructed raw embedding
		next_context_raw_row = (
			model_wrapper.get_canonical_embedding_for_id_pos(
				actual_token_ids[j_token_pos], j_token_pos, args.device
			)
			if args.teacher_force_recovered_tokens
			else reconstructed_raw_row
		)
		fixed_prev_raw_embeds_rows.append(next_context_raw_row)

	print("\n--- Overall Iterative Recovery Result ---")
	actual_decoded_prefix = model_wrapper.tokenizer.decode(
		actual_token_ids[: len(recovered_token_ids)]
	)  # Compare only recovered length
	recovered_decoded_sequence = model_wrapper.tokenizer.decode(recovered_token_ids)
	print(f"Actual sequence    : {actual_decoded_prefix}")
	print(f"Recovered sequence : {recovered_decoded_sequence}")

	len_to_compare = min(
		len(recovered_token_ids), max_len_to_recover, len(actual_token_ids)
	)
	correct_count = 0
	if len_to_compare > 0:
		correct_count = sum(
			1
			for i in range(len_to_compare)
			if recovered_token_ids[i] == actual_token_ids[i]
		)
		accuracy = correct_count / len_to_compare
	else:
		accuracy = 0.0  # Avoid division by zero if no tokens were recovered or actual sequence is empty

	print(
		f"Token-level accuracy ({len_to_compare} tokens): {correct_count}/{len_to_compare} ({accuracy*100:.2f}%)"
	)


def create_orthogonal_basis_excluding_direction(
	original_basis, direction_to_exclude, device
):
	"""
	Create a new orthonormal basis that excludes a specific direction from the original basis.
	Uses Gram-Schmidt process for robust orthogonalization.

	Args:
		original_basis: (rank, emb_dim) - Original basis with rows as basis vectors
		direction_to_exclude: (emb_dim,) or (1, emb_dim) - Direction to exclude
		device: torch device

	Returns:
		new_basis: (new_rank, emb_dim) - New orthonormal basis excluding the direction
	"""
	direction_to_exclude = direction_to_exclude.squeeze()

	direction_norm = torch.norm(direction_to_exclude)  # unit vector
	if direction_norm < 1e-8:
		print(
			f"  [Gram-Schmidt] Warning: direction to exclude has very small norm {direction_norm:.2e}"
		)
		return original_basis

	u = direction_to_exclude / direction_norm

	# project each basis vector onto the direction and subtract it
	# proj_u(v) = <v, u> * u for each row v in original_basis
	dot_products = torch.matmul(
		original_basis, u
	)  # (rank,) - dot product of each basis vector with u
	projections = dot_products.unsqueeze(1) * u.unsqueeze(
		0
	)  # (rank, emb_dim) - projection of each basis vector onto u
	orthogonal_vectors = (
		original_basis - projections
	)  # (rank, emb_dim) - remove component along u

	# remove vectors that became nearly zero (they were parallel to the excluded direction)
	vector_norms = torch.norm(orthogonal_vectors, dim=1)  # (rank,)
	valid_indices = vector_norms > 1e-6 # keep only vectors with significant norm

	if not valid_indices.any():
		print(
			f"  [Gram-Schmidt] Warning: All basis vectors were parallel to excluded direction"
		)
		# Return a random orthogonal basis in this case
		random_basis = torch.randn(
			max(1, original_basis.shape[0] - 1), original_basis.shape[1], device=device
		)
		q, _ = torch.linalg.qr(random_basis.T)
		return q.T

	orthogonal_vectors = orthogonal_vectors[valid_indices]  # keep only valid vectors

	# use QR decomposition to get an orthonormal basis for the orthogonal subspace
	q, r = torch.linalg.qr(orthogonal_vectors.T)  # q: (emb_dim, new_rank)

	# filter out zero columns (rank deficient case)
	r_diag = torch.abs(torch.diag(r))
	non_zero_cols = r_diag > 1e-6
	if non_zero_cols.any():
		q = q[:, non_zero_cols]  # (emb_dim, actual_new_rank)

	new_basis = q.T  # (actual_new_rank, emb_dim) - rows are basis vectors

	# the new basis should be orthogonal to the excluded direction
	dot_products_check = torch.abs(torch.matmul(new_basis, u))  # (actual_new_rank,)
	max_dot_product = (
		torch.max(dot_products_check).item() if len(dot_products_check) > 0 else 0.0
	)

	print(
		f"  [Gram-Schmidt] Original basis: {original_basis.shape}, New basis: {new_basis.shape}"
	)
	print(
		f"  [Gram-Schmidt] Max dot product with excluded direction: {max_dot_product:.2e}"
	)
	print(f"  [Gram-Schmidt] Excluded direction norm: {direction_norm:.6f}")
	print(
		f"  [Gram-Schmidt] New basis rank: {torch.linalg.matrix_rank(new_basis).item()}"
	)

	# check that basis vectors are orthonormal
	if new_basis.shape[0] > 1:
		gram_matrix = torch.matmul(new_basis, new_basis.T)
		identity_diff = torch.norm(
			gram_matrix - torch.eye(new_basis.shape[0], device=device)
		)
		print(
			f"  [Gram-Schmidt] Orthonormality check (should be ~0): {identity_diff.item():.2e}"
		)

	return new_basis


def compute_span_distance_sq_batched(vectors, basis):
    """
    Computes the squared L2 distance of a BATCH of vectors to the subspace
    spanned by the rows of the basis matrix.
    
    Args:
        vectors (torch.Tensor): A batch of vectors, shape (batch_size, n).
        basis (torch.Tensor): The basis matrix, shape (r, n).
        
    Returns:
        torch.Tensor: Squared distances for each vector, shape (batch_size,).
    """
    
    projection_coeffs = torch.linalg.lstsq(basis.T, vectors.T).solution # (r, batch_size)
    projected_vectors = (basis.T @ projection_coeffs).T # (batch_size, n)
    
    # Calculate squared L2 distance between original and projected vectors
    # Norm is taken over the embedding dimension (dim=1)
    dist_sq = torch.sum((vectors - projected_vectors) ** 2, dim=1)
    
    return dist_sq


def calculate_loss_for_batch_x_iterative(
    batch_x_raw_params,  # (batch_size, emb_dim) BATCH of OPTIM VARIABLES
    fixed_prev_raw_embeds_rows,  # list[(1, emb_dim)]
    current_token_pos_idx,
    Ck_basis_list,
    model_wrapper: ModelWrapper,
    alpha_k_weights,
    args,
    debug_step=None,
):
    """
    Returns a TENSOR of losses (batch_size,) for a batch of raw-embedding variables.
    This version is vectorized to run in parallel on the GPU.
    """
    batch_size = batch_x_raw_params.shape[0]
    seq_len = len(fixed_prev_raw_embeds_rows) + 1
    
    if fixed_prev_raw_embeds_rows:
        fixed_context = torch.cat(fixed_prev_raw_embeds_rows, dim=0) 
        expanded_context = fixed_context.unsqueeze(0).expand(batch_size, -1, -1)
        H_raw_0 = torch.cat(
            [expanded_context, batch_x_raw_params.unsqueeze(1)], dim=1
        ) # (batch_size, seq_len, emb_dim)
    else:
        H_raw_0 = batch_x_raw_params.unsqueeze(1) # (batch_size, 1, emb_dim)

    total_loss = torch.zeros(batch_size, device=args.device, dtype=torch.float32)

    H_raw_k = H_raw_0
    for k in range(len(Ck_basis_list)):
        # --- span check for token j at layer k for the entire batch ---
        X_LN_k = model_wrapper.get_input_to_block_after_ln(H_raw_k, k)
        
        # select the vectors for the current token position from each item in the batch
        # shape: (batch_size, emb_dim)
        x_LN_k_j_batch = X_LN_k[:, current_token_pos_idx, :]
        
        dist_sq_batch = compute_span_distance_sq_batched(x_LN_k_j_batch, Ck_basis_list[k])
        
        alpha = alpha_k_weights[k] if k < len(alpha_k_weights) else alpha_k_weights[0]
        total_loss += alpha * dist_sq_batch 

        if k < len(Ck_basis_list) - 1:
            H_raw_k = model_wrapper.forward_sequence_through_block(
                H_raw_k,
                block_idx=k,
                current_seq_len=H_raw_k.shape[1],
                device=args.device,
                already_ln=False,
            )

    diversity_loss_batch = torch.zeros(batch_size, device=args.device, dtype=torch.float32)
    if args.add_diversity_loss and fixed_prev_raw_embeds_rows:
        for prev_emb_row in fixed_prev_raw_embeds_rows:
            prev_emb = prev_emb_row.squeeze()  # (emb_dim,)
            cos_sim = torch.nn.functional.cosine_similarity(
                batch_x_raw_params, prev_emb.unsqueeze(0).expand(batch_size, -1), dim=1
            )
            diversity_loss_batch += 0.1 * cos_sim.abs() ** 2
        total_loss += diversity_loss_batch

    return total_loss


def optimize_raw_embedding_batch_reinit(
	fixed_prev_raw_embeds_rows,
	current_token_pos_idx,
	Ck_basis_list,
	model_wrapper,
	args,
	num_steps,
	lr,
	alpha_k_weights,
	true_raw_embed_row=None,
	apply_norm_heuristic=False,
	num_reinit=5,
):
	"""
	Optimize raw embedding with multiple random initializations in PARALLEL.
	"""
	print(f"\n--- Batch Reinitialization for token at position {current_token_pos_idx} ---")
	print(f"  Running {num_reinit} parallel initializations")
	
	emb_dim = model_wrapper.emb_size
	
	# create a single batch tensor of random initializations
	batch_embeddings = torch.randn(
		num_reinit, emb_dim, requires_grad=True, device=args.device, dtype=torch.float32
	)
	
	if true_raw_embed_row is not None:
		batch_embeddings.data[0] = true_raw_embed_row.clone().float().squeeze()
		print(f"  Initialized first candidate with true raw embedding")
	
	optimizer = torch.optim.AdamW([batch_embeddings], lr=lr, weight_decay=1e-4)
	
	print(f"  Starting batch optimization (batch_size={num_reinit}, emb_dim={emb_dim}) with {num_steps} steps, lr={lr}")
	
	final_losses = torch.zeros(num_reinit, device=args.device, dtype=torch.float32)
	losses_history = [[] for _ in range(num_reinit)]
	
	initial_model_training_state = model_wrapper.model.training
	model_wrapper.model.eval()
	
	for step in range(num_steps):
		optimizer.zero_grad()
		
		batch_losses = calculate_loss_for_batch_x_iterative(
			batch_embeddings,
			fixed_prev_raw_embeds_rows,
			current_token_pos_idx,
			Ck_basis_list,
			model_wrapper,
			alpha_k_weights,
			args,
			debug_step=step if step < 3 else None,
		)
		
		# sum all losses for a single backpropagation step.
		total_batch_loss = torch.sum(batch_losses)
		
		if total_batch_loss.requires_grad:
			total_batch_loss.backward(retain_graph=True)
			optimizer.step()
			
			if apply_norm_heuristic and true_raw_embed_row is not None:
				with torch.no_grad():
					true_norm = torch.norm(true_raw_embed_row)
					current_norms = torch.norm(batch_embeddings, dim=1, keepdim=True)
					current_norms[current_norms == 0] = 1.0 
					scales = true_norm / current_norms
					batch_embeddings.data.mul_(scales)
		else:
			print(f"  Step {step}: No gradients. Stopping batch optimization.")
			break
			
		final_losses = batch_losses.detach()
		for i in range(num_reinit):
			losses_history[i].append(final_losses[i].item())
		
		if step % (num_steps // 10 if num_steps >= 100 else 20) == 0 or step == num_steps - 1:
			min_loss, max_loss, mean_loss = final_losses.min().item(), final_losses.max().item(), final_losses.mean().item()
			best_init_idx = final_losses.argmin().item()
			print(f"  Step {step:4d}: Loss range [{min_loss:.3e}, {max_loss:.3e}], Mean: {mean_loss:.3e}, Best init: {best_init_idx}")
		
		if final_losses.min().item() < 1e-6:
			print(f"  Early stopping at step {step} due to very small loss")
			break
			
	# restore model state
	if initial_model_training_state:
		model_wrapper.model.train()
	else:
		model_wrapper.model.eval()

	best_init_idx = torch.argmin(final_losses).item()
	best_embedding = batch_embeddings[best_init_idx:best_init_idx+1].detach()
	best_loss = final_losses[best_init_idx].item()

	print(f"  Best initialization: {best_init_idx} with loss {best_loss:.6e}")
	print(f"  Loss improvement: {torch.max(final_losses).item() - best_loss:.6e}")

	# Plot loss curves for all initializations if requested
	if args.save_loss_plot and current_token_pos_idx == 0:
		plt.figure(figsize=(12, 8))
		
		# Plot individual loss curves
		for init_idx in range(num_reinit):
			alpha = 1.0 if init_idx == best_init_idx else 0.6
			linewidth = 2 if init_idx == best_init_idx else 1
			label = f"Init {init_idx}" + (" (Best)" if init_idx == best_init_idx else "")
			plt.plot(losses_history[init_idx], alpha=alpha, linewidth=linewidth, label=label)
		
		plt.xlabel("Optimization Step")
		plt.ylabel("Loss")
		plt.title(f"Batch Reinitialization Loss Curves (Token {current_token_pos_idx}, {num_reinit} inits)")
		plt.legend()
		plt.grid(True)
		plt.yscale("log" if max(max(losses_history[i]) for i in range(num_reinit)) > 1e-3 else "linear")
		
		plot_filename = f"batch_reinit_loss_curves_tok{current_token_pos_idx}_{args.dataset}_{args.model_path.replace('/','_')}.png"
		try:
			plt.savefig(plot_filename)
			print(f"  Batch loss curves saved to {plot_filename}")
			plt.close()
		except Exception as e:
			print(f"  Error saving batch loss plot: {e}")

	return best_embedding


if __name__ == "__main__":
	main_raw_embedding_attack()

