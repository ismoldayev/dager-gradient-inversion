# sanity_checks.py
"""
Sanity check functions for gradient inversion attacks.
Contains functions to validate the correctness of embeddings and losses.
"""

import torch
import torch.nn.functional as F
from utils.models import ModelWrapper


def _calculate_total_loss_for_sequence(
    raw_embeds_cols, # list of (emb_dim, 1) tensors
    Ck_basis_list,
    model_wrapper,
    alpha_k_weights,
    args
):
    total_loss = torch.tensor(0.0, device=args.device, dtype=torch.float32)
    
    H_raw_k = torch.stack(
        [col.squeeze(1) for col in raw_embeds_cols], dim=1
    ).unsqueeze(0)
    H_raw_k = torch.stack(
        [col.squeeze(1) for col in raw_embeds_cols], dim=0
    ).unsqueeze(0)
    for k in range(len(Ck_basis_list)):
        X_LN_k = model_wrapper.get_input_to_block_after_ln(H_raw_k, k) # (1, seq_len, emb_dim)
        
        layer_k_loss_sum = torch.tensor(0.0, device=args.device, dtype=torch.float32)
        for token_idx in range(X_LN_k.shape[1]):
            x_LN_k_token_col = X_LN_k[0, token_idx, :].unsqueeze(1) # (emb_dim, 1)
            dist_sq = compute_span_distance_sq(x_LN_k_token_col, Ck_basis_list[k])
            layer_k_loss_sum += dist_sq
        
        alpha = alpha_k_weights[k] if k < len(alpha_k_weights) else alpha_k_weights[0]
        total_loss += alpha * layer_k_loss_sum

        if k < len(Ck_basis_list) - 1:
            H_raw_k = model_wrapper.forward_sequence_through_block(
                H_raw_k,
                block_idx=k,
                current_seq_len=H_raw_k.shape[1],
                device=args.device,
                already_ln=False, 
            )
            
    return total_loss


def project_onto_subspace_lstsq(vector_col, basis_matrix_cols):
	"""
	Project a vector onto the subspace spanned by the basis matrix columns.
	
	Args:
		vector_col: (emb_dim, 1) - Vector to project
		basis_matrix_cols: (rank, emb_dim) or (emb_dim, rank) - Basis matrix
		
	Returns:
		projected_vector: (emb_dim, 1) - Projected vector
	"""
	if basis_matrix_cols.numel() == 0 or basis_matrix_cols.shape[1] == 0:
		return torch.zeros_like(vector_col)
	
	# If basis is oriented as rows (r x D) instead of columns (D x r), transpose it
	if (
		basis_matrix_cols.shape[0] != vector_col.shape[0]
		and basis_matrix_cols.shape[1] == vector_col.shape[0]
	):
		basis_matrix_cols = basis_matrix_cols.T
	
	try:
		basis_float = basis_matrix_cols.float()
		vector_float = vector_col.float()
		ATA = (
			basis_float.T @ basis_float
			+ torch.eye(basis_float.shape[1], device=basis_float.device) * 1e-7
		)
		ATb = basis_float.T @ vector_float
		coeffs = torch.linalg.solve(ATA, ATb)
		return (basis_float @ coeffs).to(vector_col.dtype)
	except torch.linalg.LinAlgError:
		try:
			coeffs = torch.linalg.lstsq(
				basis_matrix_cols.float(), vector_col.float()
			).solution
			return (basis_matrix_cols.float() @ coeffs).to(vector_col.dtype)
		except Exception as e_lstsq:
			print(
				f"Projection lstsq also failed: {e_lstsq}, basis_shape: {basis_matrix_cols.shape}, vec_shape: {vector_col.shape}"
			)
			return torch.zeros_like(vector_col)


def compute_span_distance_sq(vector_col, basis_matrix_cols):
	"""
	Compute the squared distance from a vector to the subspace spanned by the basis.
	
	Args:
		vector_col: (emb_dim, 1) - Vector to compute distance for
		basis_matrix_cols: (rank, emb_dim) - Basis matrix with rows as basis vectors
		
	Returns:
		distance_sq: scalar tensor - Squared distance to subspace
	"""
	if basis_matrix_cols.numel() == 0 or basis_matrix_cols.shape[1] == 0:
		return torch.sum(vector_col**2)
	proj = project_onto_subspace_lstsq(vector_col, basis_matrix_cols)
	return torch.sum((vector_col - proj) ** 2)


def calculate_loss_directly_from_true_embeddings_FULL_CONTEXT(
	true_raw_embeddings_full_sequence_cols,  # these are X_raw_GT^(0)
	original_sequence_length,
	num_model_layers_to_constrain,  # this is len(Ck_basis_list) effectively
	Ck_basis_list,
	model_wrapper: ModelWrapper,
	alpha_k_weights,
	args,
	true_X_LN0_full,  # this is X_LN_GT^(0) derived from true_raw_embeddings_full_sequence_cols
	captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis=None,
):
	"""
	Calculate loss for true embeddings as a sanity check.
	
	This function performs two checks:
	1. Check loss for captured original LN activations (X_LN_orig)
	2. Check loss for re-propagated true raw embeddings (X_LN_reprop)
	
	Args:
		true_raw_embeddings_full_sequence_cols: List of true raw embeddings
		original_sequence_length: Length of the original sequence
		num_model_layers_to_constrain: Number of layers to constrain
		Ck_basis_list: List of basis matrices for each layer
		model_wrapper: Model wrapper instance
		alpha_k_weights: Weights for each layer
		args: Arguments object
		true_X_LN0_full: True LN embeddings at layer 0
		captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis: Captured original LN activations
		
	Returns:
		total_loss_reprop: Total loss for re-propagated embeddings
	"""
	print(f"\n--- SANITY CHECK 2 (FULL CONTEXT / MULTI-PART) ---")
	if (
		not true_raw_embeddings_full_sequence_cols
		or len(true_raw_embeddings_full_sequence_cols) != original_sequence_length
	):
		print(
			"  Sanity Check 2 (Overall) skipped: Mismatch in true_raw_embeddings_full_sequence_cols or original_sequence_length."
		)
		return torch.tensor(float("nan"))

	# --- Part A: Check loss for CAPTURED ORIGINAL X_LN_orig^(k) ---
	print(
		f"\n--- SANITY CHECK 2.A: Loss for CAPTURED ORIGINAL LN Activations (X_LN_orig) ---"
	)
	if (
		captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis is not None
		and len(captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis)
		== len(Ck_basis_list)
		and len(Ck_basis_list) == num_model_layers_to_constrain
	):
		total_loss_captured_orig_ln = torch.tensor(
			0.0, device=args.device, dtype=torch.float32
		)
		for k_layer_idx in range(num_model_layers_to_constrain):
			X_LN_k_CAPTURED_ORIG_tensor = (
				captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis[k_layer_idx]
			)
			if X_LN_k_CAPTURED_ORIG_tensor is None:
				print(
					f"  Sanity2.A Layer {k_layer_idx}: Skipped ( Captured X_LN_orig is None for this layer)."
				)
				continue
			print(f"  Sanity2.A Processing Layer {k_layer_idx}:")
			if (
				X_LN_k_CAPTURED_ORIG_tensor.ndim != 3
			):
				print(
					f"    ERROR: X_LN_k_CAPTURED_ORIG_tensor for layer {k_layer_idx} has unexpected shape: {X_LN_k_CAPTURED_ORIG_tensor.shape}. Expected 3D tensor (batch, seq_len, embed_dim). Skipping this layer for Sanity 2.A."
				)
				continue
			
			# extract the first sample from the batch and trim to original sequence length
			if X_LN_k_CAPTURED_ORIG_tensor.shape[0] > 1:
				# Take only the first sample from the batch
				X_LN_k_CAPTURED_ORIG_tensor = X_LN_k_CAPTURED_ORIG_tensor[0:1, :, :]
			
			# trim to original sequence length if necessary
			if X_LN_k_CAPTURED_ORIG_tensor.shape[1] > original_sequence_length:
				X_LN_k_CAPTURED_ORIG_tensor = X_LN_k_CAPTURED_ORIG_tensor[:, :original_sequence_length, :]
				
			print(
				f"    Shape of X_LN_k_CAPTURED_ORIG_tensor (after slicing): {X_LN_k_CAPTURED_ORIG_tensor.shape}"
			)
			print(
				f"    Norm of X_LN_k_CAPTURED_ORIG_tensor (all tokens): {torch.norm(X_LN_k_CAPTURED_ORIG_tensor).item():.4f}"
			)
			layer_k_loss_sum_orig = torch.tensor(
				0.0, device=args.device, dtype=torch.float32
			)
			num_tokens_in_captured_activation = X_LN_k_CAPTURED_ORIG_tensor.shape[1]
			tokens_to_check_for_this_layer = min(
				original_sequence_length, num_tokens_in_captured_activation
			)
			for token_idx_in_seq in range(tokens_to_check_for_this_layer):
				x_LN_k_s_captured_orig_col = X_LN_k_CAPTURED_ORIG_tensor[
					0, token_idx_in_seq, :
				].unsqueeze(1)
				current_Ck_basis = Ck_basis_list[k_layer_idx]
				loss_k_s_captured_orig = compute_span_distance_sq(
					x_LN_k_s_captured_orig_col, current_Ck_basis
				)
				if token_idx_in_seq == 0:
					print(
						f"    Token {token_idx_in_seq}: Norm x_LN_captured_orig: {torch.norm(x_LN_k_s_captured_orig_col).item():.4f}, Ck_basis norm: {torch.norm(current_Ck_basis).item():.4f}, Loss_term: {loss_k_s_captured_orig.item():.6e}"
					)
				layer_k_loss_sum_orig += loss_k_s_captured_orig
			current_alpha = (
				alpha_k_weights[k_layer_idx]
				if k_layer_idx < len(alpha_k_weights)
				else alpha_k_weights[0]
			)
			total_loss_captured_orig_ln += current_alpha * layer_k_loss_sum_orig
			print(
				f"  Sanity2.A Layer {k_layer_idx}: Summed token loss (from {tokens_to_check_for_this_layer} tokens) = {layer_k_loss_sum_orig.item():.6e}"
			)
		print(
			f"--- TOTAL LOSS for CAPTURED ORIGINAL LN Activations (Sanity Check 2.A): {total_loss_captured_orig_ln.item():.6e} ---"
		)
	else:
		print(
			f"\n--- SANITY CHECK 2.A: Skipped (No or misaligned captured original LN activations / Ck_basis list provided) ---"
		)
		if captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis is not None:
			print(
				f"    len(captured_X_LN_orig): {len(captured_X_LN_orig_for_Ck_layers_matched_to_Ck_basis)}, len(Ck_basis_list): {len(Ck_basis_list)}, num_model_layers_to_constrain: {num_model_layers_to_constrain}"
			)

	# --- Part B: Existing check for RE-PROPAGATED X_LN_reprop^(k) ---
	print(
		f"\n--- SANITY CHECK 2.B: Loss for RE-PROPAGATED True Raw Embeddings (X_LN_reprop) ---"
	)
	current_H_raw_reprop_input_to_LN = (
		torch.stack(
			[emb.squeeze(1) for emb in true_raw_embeddings_full_sequence_cols], dim=0
		)
		.unsqueeze(0)
		.float()
	)  # Shape (1, original_sequence_length, embed_dim)

	total_loss_reprop = torch.tensor(0.0, device=args.device, dtype=torch.float32)
	initial_model_training_state = model_wrapper.model.training  # Save model state
	model_wrapper.model.eval()  # Ensure eval mode for consistent propagation

	with torch.no_grad():  # Re-propagation part is for checking, no grads needed here
		for k_layer_idx in range(num_model_layers_to_constrain):
			print(f"  Sanity2.B Re-propagating for Layer {k_layer_idx}:")
			if k_layer_idx == 0:
				X_LN_k_reprop_tensor = true_X_LN0_full
			else:
				X_LN_k_reprop_tensor = model_wrapper.get_input_to_block_after_ln(
					current_H_raw_reprop_input_to_LN, k_layer_idx
				)
			print(f"    Shape of X_LN_k_reprop_tensor: {X_LN_k_reprop_tensor.shape}")
			print(
				f"    Norm of X_LN_k_reprop_tensor (all tokens): {torch.norm(X_LN_k_reprop_tensor).item():.4f}"
			)

			layer_k_loss_sum_reprop = torch.tensor(
				0.0, device=args.device, dtype=torch.float32
			)
			for token_idx_in_seq in range(original_sequence_length):
				x_LN_k_s_reprop_col = X_LN_k_reprop_tensor[
					0, token_idx_in_seq, :
				].unsqueeze(1)
				current_Ck_basis = Ck_basis_list[k_layer_idx]
				loss_k_s_direct_gt_reprop = compute_span_distance_sq(
					x_LN_k_s_reprop_col, current_Ck_basis
				)
				if token_idx_in_seq == 0:
					print(
						f"    Token {token_idx_in_seq}: Norm x_LN_reprop: {torch.norm(x_LN_k_s_reprop_col).item():.4f}, Ck_basis norm: {torch.norm(current_Ck_basis).item():.4f}, Loss_term (reprop): {loss_k_s_direct_gt_reprop.item():.6e}"
					)
				layer_k_loss_sum_reprop += loss_k_s_direct_gt_reprop

			current_alpha = (
				alpha_k_weights[k_layer_idx]
				if k_layer_idx < len(alpha_k_weights)
				else alpha_k_weights[0]
			)
			total_loss_reprop += current_alpha * layer_k_loss_sum_reprop
			print(
				f"  Sanity2.B Layer {k_layer_idx}: Summed token loss (reprop) = {layer_k_loss_sum_reprop.item():.6e}"
			)

			if k_layer_idx < num_model_layers_to_constrain - 1:
				H_raw_reprop_next_input_to_LN = (
					model_wrapper.forward_sequence_through_block(
						current_H_raw_reprop_input_to_LN,
						k_layer_idx,  # Block index
						original_sequence_length,
						args.device,
						already_ln=False,
					)
				)  # Output is H_raw_reprop^(k+1)
				print(
					f"    Sanity2.B: Re-Propagated H_raw_reprop^({k_layer_idx+1}). Norm: {torch.norm(H_raw_reprop_next_input_to_LN).item():.4f}"
				)
				# --- Comparison Point ---
				original_block_output_key = f"transformer.h.{k_layer_idx}.block_output"
				H_raw_orig_k_plus_1 = model_wrapper.captured_original_block_outputs.get(
					original_block_output_key
				)
				if H_raw_orig_k_plus_1 is not None:
					H_raw_orig_k_plus_1 = H_raw_orig_k_plus_1.to(args.device).float()
					# Extract the relevant portion for the current sequence (first sample, up to original_seq_len)
					H_raw_orig_k_plus_1_current = H_raw_orig_k_plus_1[0:1, :original_sequence_length, :]
					
					# Check if shapes match now
					if H_raw_reprop_next_input_to_LN.shape == H_raw_orig_k_plus_1_current.shape:
						mse_block_outputs = torch.nn.functional.mse_loss(
							H_raw_reprop_next_input_to_LN, H_raw_orig_k_plus_1_current
						)
						cos_sim_block_outputs = torch.nn.functional.cosine_similarity(
							H_raw_reprop_next_input_to_LN.view(-1),
							H_raw_orig_k_plus_1_current.view(-1),
							dim=0,
						)
						print(
							f"    COMPARISON for Block {k_layer_idx} Output (H_raw^({k_layer_idx+1})):"
						)
						print(
							f"      MSE(H_raw_reprop, H_raw_orig): {mse_block_outputs.item():.6e}"
						)
						print(
							f"      CosSim(H_raw_reprop, H_raw_orig): {cos_sim_block_outputs.item():.6f}"
						)
						if mse_block_outputs.item() > 1e-5:
							print(
								f"      !!!! SIGNIFICANT DIVERGENCE in Block {k_layer_idx} output !!!!"
							)
					else:
						print(
							f"    COMPARISON for Block {k_layer_idx} Output: Shape mismatch after slicing. "
							f"Reprop: {H_raw_reprop_next_input_to_LN.shape}, Orig: {H_raw_orig_k_plus_1_current.shape}"
						)
				else:
					print(
						f"    COMPARISON for Block {k_layer_idx} Output: H_raw_orig^({k_layer_idx+1}) not captured (key: {original_block_output_key})."
					)
				current_H_raw_reprop_input_to_LN = H_raw_reprop_next_input_to_LN
	
	if initial_model_training_state:
		model_wrapper.model.train()
	else:
		model_wrapper.model.eval()

	print(
		f"--- TOTAL LOSS for RE-PROPAGATED True Raw Embeddings (Sanity Check 2.B): {total_loss_reprop.item():.6e} ---"
	)
	return total_loss_reprop


def compare_gt_vs_closest_token_losses(
	true_token_id, 
	closest_token_id,
	token_pos,
	Ck_basis_list,
	model_wrapper,
	alpha_k_weights,
	args,
):
	"""
	Compare losses for the ground truth token vs the closest recovered token.
	This helps understand if the optimization is finding the right minimum.
	
	Args:
		true_token_id: Ground truth token ID
		closest_token_id: Closest recovered token ID
		token_pos: Position of the token in the sequence
		Ck_basis_list: List of basis matrices for each layer
		model_wrapper: Model wrapper instance
		alpha_k_weights: Weights for each layer
		args: Arguments object
	"""
	print(f"\n--- Loss Comparison for Token {token_pos}: GT vs Closest ---")
	
	# Get embeddings for both tokens
	gt_embedding = model_wrapper.get_canonical_embedding_for_id_pos(
		true_token_id, token_pos, args.device
	).T  # (1, emb_dim)
	
	closest_embedding = model_wrapper.get_canonical_embedding_for_id_pos(
		closest_token_id, token_pos, args.device  
	).T  # (1, emb_dim)
	
	# Calculate losses for both
	for token_type, token_id, embedding in [
		("GT", true_token_id, gt_embedding),
		("Closest", closest_token_id, closest_embedding)
	]:
		# Build sequence with just this token (simplified)
		H_raw = embedding.unsqueeze(0).float()  # (1, 1, emb_dim)
		
		total_loss = torch.tensor(0.0, device=args.device, dtype=torch.float32)
		layer_losses = []
		
		H_raw_k = H_raw
		for k in range(len(Ck_basis_list)):
			# Get LN input for this layer
			x_LN_k_col = model_wrapper.get_input_to_block_after_ln(
				H_raw_k, k
			)[0, 0, :].unsqueeze(1)  # (emb_dim, 1)
			
			# Compute distance to basis
			dist_sq = compute_span_distance_sq(x_LN_k_col, Ck_basis_list[k])
			alpha = alpha_k_weights[k] if k < len(alpha_k_weights) else alpha_k_weights[0]
			layer_loss = alpha * dist_sq
			total_loss += layer_loss
			layer_losses.append(layer_loss.item())
			
			# Propagate if not last layer
			if k < len(Ck_basis_list) - 1:
				H_raw_k = model_wrapper.forward_sequence_through_block(
					H_raw_k, block_idx=k, current_seq_len=1,
					device=args.device, already_ln=False
				)
		
		token_str = model_wrapper.tokenizer.decode([token_id])
		print(f"  {token_type} Token '{token_str}' (ID={token_id}): Total Loss = {total_loss.item():.6e}")
		print(f"    Layer losses: {[f'{l:.2e}' for l in layer_losses]}")
	
	print("--- End Loss Comparison ---\n")


@torch.no_grad()
def check_uniform_shift_invariance(
    raw_embed_cols,            # list of (emb_dim, 1) tensors
    Ck_basis_list,
    model_wrapper: ModelWrapper,
    alpha_k_weights,
    args,
    shift_value=123.456,
):
    """
    Prints the loss and LOGITS before/after adding a uniform shift c·1 to *every* embedding.
    If LayerNorm really makes the objective invariant, the two numbers should match.
    """
    initial_model_training_state = model_wrapper.model.training
    model_wrapper.model.eval()

    loss_orig = _calculate_total_loss_for_sequence(
        raw_embed_cols, Ck_basis_list, model_wrapper, alpha_k_weights, args
    )

    ones_col = torch.ones_like(raw_embed_cols[0])
    shifted_cols = [col + shift_value * ones_col for col in raw_embed_cols]

    loss_shift = _calculate_total_loss_for_sequence(
        shifted_cols, Ck_basis_list, model_wrapper, alpha_k_weights, args
    )

    print(f"\n=== UNIFORM-SHIFT CHECK ===")
    print("--- Custom Span-Distance Loss ---")
    print(f"loss(original)   = {loss_orig.item():.6e}")
    print(f"loss(+{shift_value}·1)   = {loss_shift.item():.6e}")
    print(f"absolute diff    = {abs(loss_orig.item() - loss_shift.item()):.3e}")

    print("\n--- Full Model Logit Comparison ---")
    
    # original sequence tensor for full forward pass
    # (batch_size, seq_len, emb_dim)
    original_sequence_tensor = torch.stack(
        [col.squeeze(1) for col in raw_embed_cols], dim=0
    ).unsqueeze(0).to(model_wrapper.model.dtype)

    # shifted sequence tensor
    shifted_sequence_tensor = torch.stack(
        [col.squeeze(1) for col in shifted_cols], dim=0
    ).unsqueeze(0).to(model_wrapper.model.dtype)

    outputs_orig = model_wrapper.model(inputs_embeds=original_sequence_tensor)
    logits_orig = outputs_orig.logits

    outputs_shift = model_wrapper.model(inputs_embeds=shifted_sequence_tensor)
    logits_shift = outputs_shift.logits

    logit_mse = F.mse_loss(logits_orig.float(), logits_shift.float()).item()
    logit_max_abs_diff = torch.max(torch.abs(logits_orig - logits_shift)).item()

    print(f"Logits MSE: {logit_mse:.6e}")
    print(f"Logits Max Abs Diff: {logit_max_abs_diff:.6e}")
    print("===========================\n")
    
    if initial_model_training_state:
        model_wrapper.model.train()
    
    return loss_orig, loss_shift