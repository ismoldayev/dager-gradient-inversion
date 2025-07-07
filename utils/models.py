#models.py

import os
import torch
import peft
import numpy as np
from utils.ext import update_causal_mask
from utils.partial_models import (
	add_partial_forward_gpt2,
	add_partial_forward_bert,
	add_partial_forward_llama,
)
from constants import config
from transformers import (
	AutoTokenizer,
	AutoModelForSequenceClassification,
	AutoModelForCausalLM,
)
import torch.nn as nn
from utils.functional import get_layer_decomp


class ModelWrapper:
	def __init__(self, args):
		assert args.model_path in [
			"bert-base-uncased",
			"gpt2",
			"openai-community/gpt2-large",
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		], "Model is not yet supported - add it to assertion list and specify implementation details"
		access_token = os.environ["HF_TOKEN"]
		self.args = args
		model_kwargs = (
			{"cache_dir": args.cache_dir} if args.cache_dir is not None else {}
		)

		model_kwargs["pretrained_model_name_or_path"] = (
			args.model_path
			if args.finetuned_path is None or args.train_method == "lora"
			else args.finetuned_path
		)
		model_kwargs["attn_implementation"] = args.attn_implementation

		if args.hidden_act is not None and args.model_path in [
			"gpt2",
			"openai-community/gpt2-large",
		]:
			model_kwargs["activation_function"] = args.hidden_act
		elif args.hidden_act is not None and args.model_path in [
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		]:
			model_kwargs["hidden_act"] = args.hidden_act

		if args.precision == "8bit":
			model_kwargs["load_in_8bit"] = True
		if args.precision == "half":
			model_kwargs["torch_dtype"] = torch.float16
		if args.precision == "double":
			model_kwargs["torch_dtype"] = torch.float64
		if args.task == "seq_class":
			self.model = AutoModelForSequenceClassification.from_pretrained(
				**model_kwargs
			)
		elif args.task == "next_token_pred":
			self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
		else:
			assert False
		g_cpu = torch.Generator(device=self.model.device)
		g_cpu.manual_seed(0)
		self.model.eval()
		self.tokenizer = AutoTokenizer.from_pretrained(
			args.model_path, use_fast=True, token=access_token, cache_dir=args.cache_dir
		)
		self.tokenizer.model_max_length = 512

		if args.pad == "left":
			self.tokenizer.padding_side = "left"

		if args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			self.start_token = None
			self.eos_token = self.model.config.eos_token_id
			self.layer_ids = list(range(4, 137, 12))

			if args.task == "seq_class":
				self.model.score.weight.data.normal_(
					mean=0.0, std=1e-3, generator=g_cpu
				)

			# Set padding token
			self.model.config.pad_token_id = self.model.config.eos_token_id
			self.pad_token = self.model.config.eos_token_id
			self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
			self.embeddings_weight_nopos = self.model.transformer.wte.weight.unsqueeze(
				0
			)

			self.emb_size = self.model.config.n_embd
			add_partial_forward_gpt2(self.model.transformer)

		elif args.model_path in ["bert-base-uncased"]:

			self.start_token = 101
			self.eos_token = 102
			self.pad_token = 0
			self.layer_ids = list(range(5, 190, 16))

			# Store embeddings
			bert_embeddings_weight = (
				self.model.bert.embeddings.word_embeddings.weight.unsqueeze(0)
			)
			bert_embeddings_weight_token = (
				self.model.bert.embeddings.token_type_embeddings.weight.unsqueeze(0)
			)

			self.embeddings_weight_nopos = (
				bert_embeddings_weight_token + bert_embeddings_weight[0][:, None, :]
			)[None, :, :, :]
			self.emb_size = self.model.config.hidden_size
			add_partial_forward_bert(self.model.bert)
		elif args.model_path in [
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		]:

			self.start_token = self.tokenizer.bos_token_id
			self.eos_token = self.tokenizer.eos_token_id
			if args.model_path in [
				"meta-llama/Llama-2-7b-hf",
				"meta-llama/Llama-2-70b-hf",
			]:
				self.tokenizer.add_special_tokens(
					{"pad_token": self.tokenizer.unk_token}
				)
				self.pad_token = self.tokenizer.unk_token_id
				self.model.config.pad_token_id = self.tokenizer.unk_token_id
			else:
				self.tokenizer.add_special_tokens(
					{"pad_token": self.tokenizer.eos_token}
				)
				self.pad_token = self.tokenizer.eos_token_id
				self.model.config.pad_token_id = self.tokenizer.eos_token_id

			if args.train_method == "lora" and args.finetuned_path is not None:
				lora_cfg = peft.LoraConfig(r=args.lora_r, target_modules=["q_proj"])
				self.model = peft.LoraModel(self.model, lora_cfg, "default")
				self.model.load_state_dict(
					torch.load(args.finetuned_path, map_location=torch.device("cpu"))
				)
				self.model = self.model.model
				self.layer_ids = list(range(0, 64, 2))
			else:
				if args.task == "seq_class":
					self.model.score.weight.data.normal_(mean=0.0, std=1e-3)
				# else:
				# self.model.lm_head.weight.data.normal_(mean=0.0, std=1e-6)

				if args.train_method == "lora":
					lora_cfg = peft.LoraConfig(r=args.lora_r, target_modules=["q_proj"])
					self.full_model = peft.LoraModel(self.model, lora_cfg, "default")
					self.model = self.full_model.model
					self.layer_ids = list(range(1, 64, 2))

				else:
					self.layer_ids = list(range(1, 281, 9))

			self.emb_size = self.model.config.hidden_size
			self.embeddings_weight_nopos = (
				self.model.model.embed_tokens.weight.unsqueeze(0)
			)
			add_partial_forward_llama(self.model.model)

		self.trainable_parameters = lambda: (
			param for param in self.model.parameters() if param.requires_grad
		)
		config["START_TOKEN"] = self.start_token
		config["EOS_TOKEN"] = self.eos_token
		config["PAD_TOKEN"] = self.pad_token
		self.set_model_device(args.device)
		self.captured_original_ln_activations = {}  # For activation capture
		self.captured_original_block_outputs = {}  # For block output capture

	def compute_grads_fed_avg(self, batch, labels, create_graph=False):
		og_weights = [param.data.clone() for param in self.model.parameters()]

		self.model.eval()
		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.avg_lr)

		n_minib = batch["input_ids"].shape[0] // self.args.b_mini
		print(n_minib)
		for _ in range(self.args.avg_epochs):
			for i in range(n_minib):
				print(batch["input_ids"].shape)
				b_mini = {
					k: batch[k][i * self.args.b_mini : (i + 1) * self.args.b_mini]
					for k in batch.keys()
				}
				y_mini = labels[:, i * self.args.b_mini : (i + 1) * self.args.b_mini]
				print(b_mini["input_ids"].shape, y_mini)
				optimizer.zero_grad()
				outs = self.model(**b_mini, labels=y_mini)
				outs.loss.backward()
				optimizer.step()

		grad = [
			-(param.data.detach() - og_weights[i])
			/ n_minib
			/ self.args.avg_lr
			/ self.args.avg_epochs
			for i, param in enumerate(self.model.parameters())
		]
		for i, param in enumerate(self.model.parameters()):
			param.data = og_weights[i]
		self.model.eval()
		return grad

	def compute_grads(self, batch, y_labels, create_graph=False):
		if self.args.grad_mode == "eval":
			self.model.eval()
		else:
			self.model.train()
		dev = y_labels.device

		# --- Start Activation Capturing Setup ---
		self.captured_original_ln_activations = {}
		self.captured_original_block_outputs = {}
		hooks = []

		num_layers_for_grads_hook = self.args.n_layers
		if hasattr(self.model.config, "num_hidden_layers"):
			num_total_model_layers = self.model.config.num_hidden_layers
			num_layers_to_hook = min(
				num_layers_for_grads_hook + 1, num_total_model_layers
			)
		else:
			num_layers_to_hook = num_layers_for_grads_hook + 1

		def get_ln_hook_fn(storage_key_for_activation):
			def hook_fn_ln(module, input_activations, output_activations):
				self.captured_original_ln_activations[storage_key_for_activation] = (
					output_activations.detach().clone()
				)

			return hook_fn_ln

		def get_block_output_hook_fn(storage_key_for_block_output):
			def hook_fn_block_output(module, input_activations, output_of_block):
				# output_of_block for GPT2Block is a tuple, the first element is the hidden_state
				if isinstance(output_of_block, tuple):
					self.captured_original_block_outputs[
						storage_key_for_block_output
					] = (output_of_block[0].detach().clone())
				else:
					self.captured_original_block_outputs[
						storage_key_for_block_output
					] = output_of_block.detach().clone()

			return hook_fn_block_output

		if "gpt2" in self.args.model_path:
			for k_idx in range(num_layers_to_hook):
				try:
					# LN output hook (already present)
					ln_storage_key = f"transformer.h.{k_idx}.attn.c_attn.weight"
					module_to_hook_ln = self.model.transformer.h[k_idx].ln_1
					hooks.append(
						module_to_hook_ln.register_forward_hook(
							get_ln_hook_fn(ln_storage_key)
						)
					)
					# Block output hook
					block_output_storage_key = f"transformer.h.{k_idx}.block_output"
					module_to_hook_block = self.model.transformer.h[k_idx]
					hooks.append(
						module_to_hook_block.register_forward_hook(
							get_block_output_hook_fn(block_output_storage_key)
						)
					)
				except (AttributeError, IndexError) as e:
					print(
						f"Warning: Could not register hook for GPT-2 layer {k_idx} (LN or Block): {e}"
					)
		# ...add elif for LLaMA, BERT, etc. if needed...
		# --- End Activation Capturing Setup ---

		if self.args.precision != "8bit":
			batch_on_grad_dev = {
				k: v.to(self.args.device_grad) for k, v in batch.items()
			}
			labels_on_grad_dev = y_labels.to(self.args.device_grad)
			self.model.to(self.args.device_grad)
		else:
			batch_on_grad_dev = batch
			labels_on_grad_dev = y_labels

		if self.args.task == "next_token_pred":
			loss_labels = torch.where(
				batch_on_grad_dev["attention_mask"].bool(),
				batch_on_grad_dev["input_ids"],
				-100,
			)
		else:
			loss_labels = labels_on_grad_dev

		grad = None
		if self.args.grad_b is None:
			if self.args.algo == "fedavg":
				print(
					"Warning: Activation capture with FedAvg needs careful review of hook placement."
				)
				grad = self.compute_grads_fed_avg(
					batch_on_grad_dev, loss_labels, create_graph
				)
			else:
				outs = self.model(**batch_on_grad_dev, labels=loss_labels)
				current_loss = outs.loss
				if self.args.loss == "mse":
					raise NotImplementedError(
						"MSE loss for grad computation needs review with hooks."
					)
				trainable_params = [
					p for p in self.trainable_parameters() if p.requires_grad
				]
				grad_tuple = torch.autograd.grad(
					current_loss,
					trainable_params,
					create_graph=create_graph,
					allow_unused=True,
				)
				grad_map = {p: g for p, g in zip(trainable_params, grad_tuple)}
				grad = [grad_map.get(p, None) for p in self.model.parameters()]
		else:
			print(
				"Warning: Activation capture with grad_b will capture from the LAST microbatch."
			)
			for param in self.model.parameters():
				if param.grad is not None:
					param.grad.zero_()
			minib_size = self.args.batch_size // self.args.grad_b
			for i in range(self.args.grad_b):
				mini_batch_input = {
					k: batch_on_grad_dev[k][i * minib_size : (i + 1) * minib_size]
					for k in batch.keys()
				}
				if self.args.task == "seq_class":
					current_loss_labels = loss_labels[
						:, i * minib_size : (i + 1) * minib_size
					]
				else:
					current_loss_labels = loss_labels[
						i * minib_size : (i + 1) * minib_size
					]
				outs = self.model(**mini_batch_input, labels=current_loss_labels)
				micro_loss = outs.loss
				(micro_loss / self.args.grad_b).backward()
			grad = tuple(
				[
					(
						param.grad.detach().cpu().clone()
						if param.grad is not None
						else None
					)
					for param in self.model.parameters()
				]
			)

		# --- Remove Hooks ---
		for h in hooks:
			h.remove()
		# --- End Remove Hooks ---

		self.set_model_device(dev)
		if self.args.precision != "8bit" and dev.type != self.args.device_grad:
			pass
		self.model.eval()
		return grad

	def set_model_device(self, device):
		if self.args.precision == "8bit":
			return
		if (
			self.args.model_path
			in [
				"meta-llama/Llama-2-7b-hf",
				"meta-llama/Llama-2-70b-hf",
				"meta-llama/Meta-Llama-3-8B",
				"meta-llama/Meta-Llama-3.1-8B",
				"meta-llama/Meta-Llama-3-70B",
			]
			and device != "cpu"
		):
			self.model.model.embed_tokens.to(device)
			self.model.model.rotary_emb.to(device)
			for i in range(self.args.n_layers):
				self.model.model.layers[i].to(device)
		else:
			self.model.to(device)

	def get_matrices_expansions(self, true_grads, B=None, tol=None):
		if B is None:
			max_rank = 0
			for i in self.layer_ids[:10]:
				grad = true_grads[i]
				if self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
					grad = grad.T
				if self.args.precision == "half":
					B = np.linalg.matrix_rank(grad.float().cpu(), tol=tol)
				else:
					B = np.linalg.matrix_rank(grad.cpu(), tol=tol)
				if max_rank < B:
					max_rank = B
			B = max_rank
		if self.args.algo == "fedavg":
			B += 60
		B = min(B, self.emb_size - self.args.rank_cutoff)

		R_Qs = []

		for i in range(self.args.n_layers):
			grad_Q = true_grads[self.layer_ids[i]]
			if self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
				grad_Q = grad_Q.T
			_, R_Q = get_layer_decomp(
				grad_Q, B=B, tol=tol, upcast=(self.args.precision == "half")
			)
			R_Q = R_Q.to(self.args.device)
			R_Qs.append(R_Q)
		return B, R_Qs

	def get_embeddings(self, pos=None):
		if self.args.model_path in ["bert-base-uncased"]:
			bert_embeddings_weight_position = (
				self.model.bert.embeddings.position_embeddings.weight.unsqueeze(0)
			)
			emb = (
				self.embeddings_weight_nopos.to(self.args.device)
				+ bert_embeddings_weight_position[0][pos : pos + 1, None, None, :]
			)
			emb = self.model.bert.embeddings.LayerNorm(emb)
			return emb

		elif self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			gpt_embeddings_weight_position = (
				self.model.transformer.wpe.weight.unsqueeze(0)
			)
			emb = (
				self.embeddings_weight_nopos.to(self.args.device)
				+ gpt_embeddings_weight_position[0][pos : pos + 1, None, :]
			)
			emb = self.model.transformer.h[0].ln_1(emb)
			return emb
		elif self.args.model_path in [
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		]:
			emb = self.embeddings_weight_nopos.to(self.args.device)
			return self.model.model.layers[0].input_layernorm(emb)

	def get_layer_inputs(
		self, sentences, token_type_ids=None, attention_mask=None, layers=1
	):
		if self.args.model_path in ["bert-base-uncased"]:
			# if token_type_ids is None:
			#     raise ValueError('Token type must be defined when model is BERT')
			# emb = self.model.bert.embeddings( input_ids=sentences, token_type_ids=token_type_ids )
			# layer_inputs = []
			# for i in range(layers):
			#     emb = self.model.bert.encoder.layer[i](emb)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
			#     layer_inputs.append(emb[ : , :-1, : ].clone())
			# return layer_inputs
			return self.model.bert.get_hidden_states(
				input_ids=sentences, token_type_ids=token_type_ids, n_layers=layers
			)

		elif self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			return self.model.transformer.get_hidden_states(
				input_ids=sentences, attention_mask=attention_mask, n_layers=layers
			)

		elif self.args.model_path in [
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		]:
			position_ids = (
				torch.arange(sentences.size(1))
				.unsqueeze(0)
				.repeat(sentences.size(0), 1)
				.to(self.args.device)
			)
			# if attention_mask is not None:
			#     first_item_idx = torch.argmax(attention_mask, dim=1).unsqueeze(1)
			#     position_ids = torch.maximum(position_ids - first_item_idx, torch.tensor(0).to(self.args.device))
			#     attention_mask = update_causal_mask(self.model.model, attention_mask, emb).to(self.args.device)

			# layer_inputs = []
			# for i in range(layers):
			#     emb = self.model.model.layers[i](emb, attention_mask=attention_mask, position_ids=position_ids)[0]# As end of sentence tokens have little gradient they are unreliable measures for sentence inclusion
			#     layer_inputs.append(self.model.model.layers[i+1].input_layernorm(emb))
			# return layer_inputs
			return self.model.model.get_hidden_states(
				input_ids=sentences,
				position_ids=position_ids,
				attention_mask=attention_mask,
				n_layers=layers,
			)

	def is_bert(self):
		return self.args.model_path in ["bert-base-uncased"]

	def is_decoder(self):
		return self.args.model_path in [
			"gpt2",
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"openai-community/gpt2-large",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		]

	def has_rope(self):
		return self.args.model_path in [
			"meta-llama/Llama-2-7b-hf",
			"meta-llama/Llama-2-70b-hf",
			"meta-llama/Meta-Llama-3-8B",
			"meta-llama/Meta-Llama-3.1-8B",
			"meta-llama/Meta-Llama-3-70B",
		]

	def has_bos(self):
		return self.start_token is not None

	def is_lower(self):
		return self.args.precision in ["8bit", "half"]

	def get_canonical_embedding_for_id_pos(self, token_id, position_idx, device):
		"""
		Generates the canonical 'raw' (pre-first-layernorm) embedding for a given token ID at a specific position.
		Output shape: (1, embed_dim)
		"""
		token_id_tensor = torch.tensor(
			[[token_id]], device=device, dtype=torch.long
		)  # Shape (1,1)

		with torch.no_grad():
			if self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
				word_emb = self.model.transformer.wte(
					token_id_tensor
				).float()  # (1,1,N)
				pos_ids = torch.tensor(
					[[position_idx]], device=device, dtype=torch.long
				)  # (1,1)
				pos_emb = self.model.transformer.wpe(pos_ids).float()  # (1,1,N)
				canonical_emb = (word_emb + pos_emb).squeeze(0)  # (1,N)
			elif self.has_rope():  # LLaMA-like
				# For RoPE models, the "raw" input to the first LN is just the word embedding
				canonical_emb = (
					self.model.model.embed_tokens(token_id_tensor).squeeze(0).float()
				)  # (1,N)
			elif self.is_bert():
				pos_ids = torch.tensor(
					[[position_idx]], device=device, dtype=torch.long
				)
				# Assuming token_type_id = 0 for general text
				token_type_ids = torch.zeros_like(
					token_id_tensor, device=device, dtype=torch.long
				)

				word_emb = self.model.bert.embeddings.word_embeddings(token_id_tensor)
				pos_emb = self.model.bert.embeddings.position_embeddings(pos_ids)
				type_emb = self.model.bert.embeddings.token_type_embeddings(
					token_type_ids
				)
				canonical_emb = (
					(word_emb + pos_emb + type_emb).squeeze(0).float()
				)  # (1,N)
			else:
				raise NotImplementedError(
					f"get_canonical_embedding_for_id_pos not implemented for {self.args.model_path}"
				)
		return canonical_emb

	def get_input_to_block_after_ln(
		self, raw_embeddings_sequence_tensor_batch_first, block_idx
	):
		"""
		Applies the appropriate LayerNorm before a given transformer block.
		Args:
			raw_embeddings_sequence_tensor_batch_first (torch.Tensor): Shape (1, SeqLen, EmbedDim)
																	Represents H_raw^(k) (output of previous block, or initial embeds)
			block_idx (int): Index of the block whose LN is to be applied.
		Returns:
			torch.Tensor: Shape (1, SeqLen, EmbedDim) - embeddings after LayerNorm (X_LN^(k))
		"""
		self.model.eval() 

		ln_output = None
		if self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			ln_layer = self.model.transformer.h[block_idx].ln_1
			ln_output = ln_layer(raw_embeddings_sequence_tensor_batch_first)
		elif self.has_rope():  # LLaMA-like
			ln_layer = self.model.model.layers[block_idx].input_layernorm
			ln_output = ln_layer(raw_embeddings_sequence_tensor_batch_first)
		elif self.is_bert():
			ln_output = raw_embeddings_sequence_tensor_batch_first  
		else:
			raise NotImplementedError(
				f"get_input_to_block_after_ln not implemented for {self.args.model_path}"
			)
		return ln_output

	def forward_sequence_through_block(
		self,
		ln_embeddings_sequence_tensor_batch_first,
		block_idx,
		current_seq_len,
		device,
		already_ln=False,
	):
		self.model.eval()
		hidden_states = ln_embeddings_sequence_tensor_batch_first

		position_ids_for_block = None

		if self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			layer_module = self.model.transformer.h[block_idx]
			# Always feed raw embeddings; let block apply its own LayerNorm
			attention_mask_arg = None
			if self.args.attn_implementation == "eager":
				attention_mask_arg = None
			elif self.args.attn_implementation == "sdpa":
				attention_mask_arg = None
			else:
				print(
					f"Warning: Unknown attn_implementation '{self.args.attn_implementation}', defaulting to None mask."
				)

			outputs = layer_module(
				hidden_states, attention_mask=attention_mask_arg, use_cache=False
			)
			sequence_output = outputs[0]
		elif self.has_rope():
			layer_module = self.model.model.layers[block_idx]
			position_ids_for_block = torch.arange(
				0, current_seq_len, dtype=torch.long, device=device
			).unsqueeze(0)
			_llama_attention_mask = None
			if current_seq_len > 1:
				llama_causal_mask = torch.full(
					(current_seq_len, current_seq_len),
					fill_value=torch.finfo(hidden_states.dtype).min,
					device=device,
				)
				llama_causal_mask = torch.triu(llama_causal_mask, diagonal=1)
				_llama_attention_mask = llama_causal_mask.unsqueeze(0).unsqueeze(0)

			outputs = layer_module(
				hidden_states,
				attention_mask=_llama_attention_mask,
				position_ids=position_ids_for_block,
				use_cache=False,
			)
			sequence_output = outputs[0]
		elif self.is_bert():
			layer_module = self.model.bert.encoder.layer[block_idx]
			causal_mask = torch.zeros(
				1,
				1,
				current_seq_len,
				current_seq_len,
				device=device,
				dtype=hidden_states.dtype,
			)
			causal_mask.masked_fill_(
				torch.triu(
					torch.ones(
						current_seq_len,
						current_seq_len,
						device=device,
						dtype=torch.bool,
					),
					diagonal=1,
				),
				float("-inf"),
			)
			attention_mask_for_block = causal_mask
			layer_outputs = layer_module(
				hidden_states, attention_mask=attention_mask_for_block
			)
			sequence_output = layer_outputs[0]
		else:
			raise NotImplementedError(
				f"forward_sequence_through_block not implemented for {self.args.model_path}"
			)
		return sequence_output  # Shape (1, SeqLen, EmbedDim)

	def forward_one_block_for_token(self, token_embedding_col_vec, block_index):
		self.model.eval()  # Ensure model is in eval mode
		current_device = token_embedding_col_vec.device

		# (1 x n) -> (1 x 1 x n)
		input_for_block = token_embedding_col_vec.T.unsqueeze(1)

		hidden_states = (
			input_for_block  # This is the input to the first layer of the block
		)

		if self.args.model_path in ["bert-base-uncased"]:
			attention_mask = torch.ones(
				1, 1, device=current_device, dtype=torch.long
			)  # (batch_size, seq_length)
			extended_attention_mask = self.model.bert.get_extended_attention_mask(
				attention_mask, input_for_block.size(), current_device
			)
			layer_module = self.model.bert.encoder.layer[block_index]
			layer_outputs = layer_module(
				hidden_states, attention_mask=extended_attention_mask
			)
			sequence_output = layer_outputs[0]  # (batch_size, seq_len, hidden_size)

		elif self.args.model_path in ["gpt2", "openai-community/gpt2-large"]:
			layer_module = self.model.transformer.h[block_index]
			outputs = layer_module(hidden_states, use_cache=False)
			sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)

		elif self.has_rope():  # LLaMa-like models
			
			position_ids = torch.zeros(
				1, 1, dtype=torch.long, device=current_device
			)  # (batch_size, seq_len)
			
			layer_module = self.model.model.layers[block_index]

			outputs = layer_module(
				hidden_states,
				attention_mask=None,
				position_ids=position_ids,
				use_cache=False,
			)
			sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
		else:
			raise NotImplementedError(
				f"forward_one_block_for_token not implemented for {self.args.model_path}"
			)

		# Output is (1, 1, n), so squeeze and transpose to get n x 1 column vector
		return sequence_output.squeeze(1).T
