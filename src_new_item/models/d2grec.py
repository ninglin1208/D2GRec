import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import torch.optim as optim
from logging import getLogger
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Assuming these modules exist in your project structure
from src_new_item.common.abstract_recommender import GeneralRecommender
from src_new_item.utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood
from src_new_item.utils.mi_estimator import *
from src_new_item.common.loss import MSELoss

class DiffusionRefiner(nn.Module):
    """
    ResMLP-based Diffusion Refiner (Interface consistent with original):
    - Replaces the original three-layer MLP with multi-layer Residual + FiLM conditioning.
    - Supports classifier-free guidance (CFG) and the original sampling process.
    - Directly compatible with existing cond designs (e.g., 4096+64 / 384+64).
    """
    def __init__(self, feat_dim, cond_dim, timesteps=20, cond_dropout=0.2, guidance_scale=3.0,
                 n_blocks=4, inner_mult=4, dropout=0.1):
        super().__init__()
        self.timesteps = timesteps
        self.cond_dropout = cond_dropout
        self.default_guidance_scale = float(guidance_scale)
        self.cond_dim = int(cond_dim)
        self.feat_dim = int(feat_dim)

        # Time Embedding (Keeping original 64 dimensions)
        self.t_embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

        # Conditional Encoding: Fuse [cond, t_emb] into a conditional hidden vector
        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim + 64, max(256, feat_dim)), nn.SiLU(),
            nn.Linear(max(256, feat_dim), max(256, feat_dim))
        )

        inner = max(128, min(1024, feat_dim * inner_mult // 2))  # Inner width, controlling parameter size

        class FiLMResBlock(nn.Module):
            def __init__(self, feat_dim, cond_hid, inner, dropout=0.1):
                super().__init__()
                self.norm = nn.LayerNorm(feat_dim)
                self.cond_to_film = nn.Linear(cond_hid, 2 * feat_dim)  # scale, shift
                self.fc1 = nn.Linear(feat_dim, inner * 2)              # SwiGLU: split -> silu(a)*b
                self.fc2 = nn.Linear(inner, feat_dim)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x, cond_h):
                h = self.norm(x)
                scale, shift = self.cond_to_film(cond_h).chunk(2, dim=-1)
                h = h * (1.0 + scale) + shift
                a, b = self.fc1(h).chunk(2, dim=-1)
                h = F.silu(a) * b
                h = self.fc2(h)
                return x + self.dropout(h)

        self.blocks = nn.ModuleList([FiLMResBlock(feat_dim, max(256, feat_dim), inner, dropout)
                                     for _ in range(n_blocks)])
        self.out = nn.Linear(feat_dim, feat_dim)

    def q_sample(self, x0, t, noise):
        device = x0.device
        betas = torch.linspace(1e-4, 0.02, steps=self.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alpha_bars[t].view(-1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

    def _pack_cond(self, x, cond, t):
        B = x.size(0)
        dev = x.device

        # 1) Handle cond shape: ensure it is [B, self.cond_dim]
        if cond is None:
            cond = torch.zeros(B, self.cond_dim, device=dev)
        elif cond.dim() != 2 or cond.size(1) != self.cond_dim:
            if cond.size(1) > self.cond_dim:
                cond = cond[:, :self.cond_dim]
            else:
                pad = self.cond_dim - cond.size(1)
                cond = torch.cat(
                    [cond, torch.zeros(cond.size(0), pad, device=dev, dtype=cond.dtype)],
                    dim=1
                )

        # 2) Handle timestep t: could be scalar / [1] / [B], broadcast to [B]
        if t.dim() == 0:
            t = t.expand(B)
        elif t.dim() == 1 and t.size(0) == 1 and B > 1:
            t = t.expand(B)

        # t is now guaranteed to be [B]
        t_emb = self.t_embed(t.float().view(-1, 1) / max(self.timesteps, 1))  # [B, 64]

        # Guard against edge cases
        if t_emb.size(0) == 1 and B > 1:
            t_emb = t_emb.expand(B, -1)

        # 3) Concatenate cond and t_emb to get conditional hidden vector
        cond_h = self.cond_proj(torch.cat([cond, t_emb], dim=-1))  # [B, cond_hid]
        return cond_h

    def forward(self, x0, cond, t, train_cfg=True):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        # Conditional dropout for classifier-free guidance
        if train_cfg and self.training and self.cond_dropout > 0:
            drop = (torch.rand(xt.size(0), 1, device=xt.device) < self.cond_dropout).float()
            cond = cond * (1.0 - drop)

        cond_h = self._pack_cond(xt, cond, t)
        h = xt
        for blk in self.blocks:
            h = blk(h, cond_h)
        eps_pred = self.out(h)
        return eps_pred, noise

    @torch.no_grad()
    def sample(self, x_init, cond, guidance_scale=None):
        return self.sample_cfg(x_init, cond, guidance_scale if guidance_scale is not None else self.default_guidance_scale)

    @torch.no_grad()
    def sample_cfg(self, x_init, cond, guidance_scale=None):
        gs = self.default_guidance_scale if guidance_scale is None else float(guidance_scale)
        device = x_init.device
        betas = torch.linspace(1e-4, 0.02, steps=self.timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        x = x_init

        # Prepare "unconditional" cond_h
        zero_cond = torch.zeros_like(cond)
        for t in reversed(range(self.timesteps)):
            # Conditional / Unconditional branches
            cond_h_c = self._pack_cond(x, cond, torch.tensor([t], device=device))
            cond_h_u = self._pack_cond(x, zero_cond, torch.tensor([t], device=device))

            # Run network for both branches
            def run_blocks(x, cond_h):
                h = x
                for blk in self.blocks:
                    h = blk(h, cond_h)
                return self.out(h)

            eps_c = run_blocks(x, cond_h_c)
            eps_u = run_blocks(x, cond_h_u)
            eps = eps_u + gs * (eps_c - eps_u)

            alpha, alpha_bar, beta = alphas[t], alpha_bars[t], betas[t]
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps)
            x = mean + (torch.sqrt(beta) * torch.randn_like(x) if t > 0 else 0.0)
        return x

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer Implementation"""

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        h: [N, in_features] Node features
        adj: [N, N] Adjacency matrix
        """
        Wh = torch.mm(h, self.W)  # [N, out_features]
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        return h_prime, attention

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return self.leakyrelu(torch.mm(all_combinations_matrix, self.a).view(N, N))


class AdaptiveKNNBuilder:
    """Adaptive KNN Graph Builder"""

    def __init__(self, base_k=10, min_k=3, max_k=50):
        self.base_k = base_k
        self.min_k = min_k
        self.max_k = max_k

    def get_adaptive_k(self, item_maturity):
        """Returns adaptive k value based on item maturity"""
        # Higher maturity -> larger k
        adaptive_k = self.min_k + (self.max_k - self.min_k) * item_maturity
        return torch.clamp(adaptive_k, self.min_k, self.max_k).int()

    def build_adaptive_knn(self, similarity_matrix, item_maturity_scores):
        """Builds adaptive KNN graph"""
        n_items = similarity_matrix.size(0)
        
        # Create adjacency matrix
        adj_matrix = torch.zeros_like(similarity_matrix)

        for i in range(n_items):
            # Get adaptive k for current item
            k_i = self.get_adaptive_k(item_maturity_scores[i]).item()

            # Find top k_i similar neighbors
            similarities = similarity_matrix[i]
            _, top_k_indices = torch.topk(similarities, k_i + 1)  # +1 to exclude self

            # Exclude self
            top_k_indices = top_k_indices[top_k_indices != i][:k_i]

            # Set adjacency
            adj_matrix[i, top_k_indices] = similarities[top_k_indices]

        # Symmetrize
        adj_matrix = (adj_matrix + adj_matrix.t()) / 2

        return adj_matrix


class D2GRec(GeneralRecommender):
    def __init__(self, config, dataset, logger=None):
        super(D2GRec, self).__init__(config, dataset)

        if 'n_users' in config and 'n_items' in config:
            self.n_users = config['n_users']
            self.n_items = config['n_items']

        self.config = config
        self.use_feature_diffusion = bool(config['use_feature_diffusion'])

        # ===== Two-stage generation ablation switches =====
        # Stage-1 coarse generator
        self.use_stage1_gen = bool(config['use_stage1_gen'])

        # Stage-2 diffusion refiner (inherits from use_feature_diffusion)
        self.use_stage2_gen = bool(config['use_stage2_gen'])

        # Master switch for missing modal generation
        if 'enable_missing_modal_generation' in config:
            self.enable_missing_modal_generation = bool(config['enable_missing_modal_generation'])
        else:
            self.enable_missing_modal_generation = self.use_stage1_gen or self.use_stage2_gen

        # ===== Improvement 1: Progressive Graph Integration Parameters =====
        self.progressive_integration = config['progressive_integration']
        self.maturity_threshold = config['maturity_threshold']
        self.integration_rate = config['integration_rate']

        # ===== Improvement 2: Adaptive KNN Parameters =====
        self.adaptive_knn = config['adaptive_knn']
        self.base_k = config['base_k']
        self.min_k = config['min_k']
        self.max_k = config['max_k']

        # ===== Improvement 3: Graph Attention Parameters =====
        self.use_graph_attention = config['use_graph_attention']
        self.attention_dropout = config['attention_dropout']

        self.cfg_scale = float(config['cfg_scale'])
        self.new_items = int(self.config['new_items'])
        
        if self.new_items:
            dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
            new_items_path = os.path.join(dataset_path, "new_items.npy")
            if os.path.exists(new_items_path):
                self.new_items_set = np.load(new_items_path)
                self.old_items_set = np.setdiff1d(np.arange(self.n_items), self.new_items_set)
            else:
                self.new_items_set = np.array([], dtype=int)
                self.old_items_set = np.arange(self.n_items)
        else:
            self.new_items_set = self.old_items_set = np.arange(self.n_items)

        self.logger = getLogger() if logger is None else logger
        self.embedding_dim = config['embedding_size']
        self.n_ui_layers = config['n_ui_layers']
        self.n_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']

        self.diff_timesteps = int(self.config['diff_timesteps'])
        self.msmlp_prompt_scale = float(self.config['msmlp_prompt_scale'])
        self.coarse_fuse_alpha = float(self.config['coarse_fuse_alpha'])

        # Short-term / Long-term interest parameters
        self.use_fixed_short_weight = bool(self.config['use_fixed_short_weight'])
        self.short_term_weight = float(self.config['short_term_weight'])
        self.gate_bias_init = float(self.config['gate_bias_init'])

        # New item related parameters
        self.new_item_alpha = float(config['new_item_alpha'])
        self.new_item_thresh = float(config['new_item_thresh'])
        self.new_item_warmup_epochs = int(config['new_item_warmup_epochs'])
        self.generate_only_new = bool(config['generate_only_new'])
        # gen_blend_alpha is redundant with new_item_alpha, but keeping for consistency if used elsewhere
        self.gen_blend_alpha = float(self.config['new_item_alpha'])

        self.freeze_pretrained_features = config['freeze_pretrained_features']
        self.freeze_encoders_after_epoch = config['freeze_encoders_after_epoch']
        self.mi_update_frequency = config['mi_update_frequency']
        self.adj_update_frequency = config['adj_update_frequency']
        self.current_epoch = 0

        # ===== Initialize Components =====
        # 1. Adaptive KNN Builder
        self.adaptive_knn_builder = AdaptiveKNNBuilder(
            base_k=self.base_k,
            min_k=self.min_k,
            max_k=self.max_k
        )

        # 2. Item Maturity Tracking
        self.item_interaction_counts = torch.zeros(self.n_items, device=self.device)
        self.item_maturity_scores = torch.zeros(self.n_items, device=self.device)

        # 3. Graph Attention Layers
        if self.use_graph_attention:
            self.image_gat = GraphAttentionLayer(
                self.embedding_dim, self.embedding_dim,
                dropout=self.attention_dropout
            ).to(self.device)
            self.text_gat = GraphAttentionLayer(
                self.embedding_dim, self.embedding_dim,
                dropout=self.attention_dropout
            ).to(self.device)

        # ==================== Prompt Learning Parameters ====================
        self.prompt_length = config['prompt_length']
        self.prompt_complete = nn.Parameter(torch.randn(self.prompt_length, 64))
        self.prompt_image_only = nn.Parameter(torch.randn(self.prompt_length, 64))
        self.prompt_text_only = nn.Parameter(torch.randn(self.prompt_length, 64))
        self.prompt_new_item = nn.Parameter(torch.randn(self.prompt_length, 64))
        nn.init.xavier_uniform_(self.prompt_new_item)
        nn.init.xavier_uniform_(self.prompt_complete)
        nn.init.xavier_uniform_(self.prompt_image_only)
        nn.init.xavier_uniform_(self.prompt_text_only)

        self.prompt_to_image_proj = nn.Linear(64, 4096).to(self.device)
        self.prompt_to_text_proj = nn.Linear(64, 384).to(self.device)

        # Multi-scale MLP components
        self.image_scale_1 = nn.Sequential(nn.Linear(4096, 2048), nn.ReLU(), nn.Linear(2048, 4096), nn.Tanh()).to(self.device)
        self.image_scale_2 = nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.Linear(1024, 4096), nn.Tanh()).to(self.device)
        self.image_scale_3 = nn.Sequential(nn.Linear(4096, 512), nn.ReLU(), nn.Linear(512, 4096), nn.Tanh()).to(self.device)
        self.text_scale_1 = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 384), nn.Tanh()).to(self.device)
        self.text_scale_2 = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 384), nn.Tanh()).to(self.device)
        self.text_scale_3 = nn.Sequential(nn.Linear(384, 64), nn.ReLU(), nn.Linear(64, 384), nn.Tanh()).to(self.device)
        self.image_fusion_attn = nn.Sequential(nn.Linear(4096 * 3, 3), nn.Softmax(dim=-1)).to(self.device)
        self.text_fusion_attn = nn.Sequential(nn.Linear(384 * 3, 3), nn.Softmax(dim=-1)).to(self.device)

        # Adaptive weighting parameters
        self.log_sigma_bpr = nn.Parameter(torch.zeros(1))
        self.log_sigma_align = nn.Parameter(torch.zeros(1))
        self.log_sigma_gen = nn.Parameter(torch.zeros(1))
        self.log_sigma_gan = nn.Parameter(torch.zeros(1))
        self.log_sigma_reg = nn.Parameter(torch.zeros(1))

        # Attention Fuser
        class AttentionFuser(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.query_proj = nn.Linear(embedding_dim, embedding_dim)
                self.key_proj = nn.Linear(embedding_dim, embedding_dim)
                self.value_proj = nn.Linear(embedding_dim, embedding_dim)
                self.scaler = math.sqrt(embedding_dim)

            def forward(self, id_emb, img_emb, txt_emb):
                q = self.query_proj(id_emb)
                modal_embs = torch.stack([id_emb, img_emb, txt_emb], dim=1)
                k = self.key_proj(modal_embs)
                v = self.value_proj(modal_embs)
                attn_weights = F.softmax(torch.bmm(q.unsqueeze(1), k.transpose(1, 2)) / self.scaler, dim=-1)
                return torch.bmm(attn_weights, v).squeeze(1)

        self.item_fuser = AttentionFuser(self.embedding_dim).to(self.device)

        class LightweightDiscriminator(nn.Module):
            def __init__(self, feat_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(feat_dim, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())

            def forward(self, x): return self.net(x)

        # GAN Generation
        self.discriminator_img = LightweightDiscriminator(4096).to(self.device)
        self.discriminator_txt = LightweightDiscriminator(384).to(self.device)
        self.adv_loss = nn.BCELoss()

        # Diffusion Refiner
        cond_img_dim = 4096 + 64
        cond_txt_dim = 384 + 64

        self.diff_img = DiffusionRefiner(feat_dim=4096, cond_dim=cond_img_dim, timesteps=self.diff_timesteps,
                                         guidance_scale=self.cfg_scale).to(self.device)
        self.diff_txt = DiffusionRefiner(feat_dim=384, cond_dim=cond_txt_dim, timesteps=self.diff_timesteps,
                                         guidance_scale=self.cfg_scale).to(self.device)

        # Verifier / Feedback parameters
        _z_candidate = self.config['z_dim']
        if _z_candidate is None:
            _z_candidate = self.config['latent_dim']
        self._verifier_zdim = int(_z_candidate)

        self.enc_proj_img = nn.Linear(4096, self._verifier_zdim).to(self.device)
        self.enc_proj_txt = nn.Linear(384, self._verifier_zdim).to(self.device)
        self.user2feat_img = nn.Linear(self.embedding_dim, 4096).to(self.device)
        self.user2feat_txt = nn.Linear(self.embedding_dim, 384).to(self.device)
        self.feat2prompt_img = nn.Linear(4096, 64).to(self.device)
        self.feat2prompt_txt = nn.Linear(384, 64).to(self.device)

        # Base Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Dynamic Weighting
        self.time_decay_lambda = 0.01
        self.enable_dynamic_weighting = True
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        if self.enable_dynamic_weighting:
            try:
                self.interaction_history = dataset.get_interaction_history()
                self.latest_timestamp = dataset.get_latest_timestamp()
                self.logger.info("Successfully loaded interaction history, dynamic preference weighting enabled.")
            except AttributeError:
                self.logger.warning(
                    "Dynamic preference weighting is enabled, but dataset object is missing required methods. Falling back to static weights.")
                self.enable_dynamic_weighting = False

        # Graph Structure Initialization
        self.n_nodes = self.n_users + self.n_items
        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix,
                                                      torch.Size((self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.norm_adj = self.norm_adj.to(self.device)
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(self.device)
        self.all_items = np.arange(self.n_items)

        # Projection Layers
        self.image_proj = nn.Linear(4096, self.embedding_dim).to(self.device)
        self.text_proj = nn.Linear(384, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)

        # Missing Modality Handling
        self.complete_items = np.arange(self.n_items)
        self.missing_modal = config['missing_modal']
        if self.missing_modal:
            self.preprocess_missing_modal(config)

        self.momentum_eta = float(config['momentum_eta'])  # Momentum update step

        # Modality Feature Initialization and Graph Construction
        # ========= Modality Feature Initialization =========
        self.image_embedding = None
        self.text_embedding = None

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=self.freeze_pretrained_features
            )

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=self.freeze_pretrained_features
            )

        # ========= Initial Graph Construction (Using the two embeddings above) =========
        if self.v_feat is not None:
            self._build_adaptive_image_graph()

        if self.t_feat is not None:
            self._build_adaptive_text_graph()

        torch.cuda.empty_cache()

        # Encoders and Other Network Components
        self.loss_nce = config['loss_nce']
        self.image_encoder = nn.Linear(self.v_feat.shape[1], self.embedding_dim).to(self.device)
        self.text_encoder = nn.Linear(self.t_feat.shape[1], self.embedding_dim).to(self.device)
        self.shared_encoder = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.image_encoder.weight)
        nn.init.xavier_uniform_(self.text_encoder.weight)
        nn.init.xavier_uniform_(self.shared_encoder.weight)

        # Preference Networks
        self.image_preference_ = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False).to(self.device)
        self.text_preference_ = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False).to(self.device)
        self.image_pref_encoder = nn.Linear(self.embedding_dim, 2 * 4096).to(self.device)
        self.text_pref_encoder = nn.Linear(self.embedding_dim, 2 * 384).to(self.device)
        nn.init.xavier_uniform_(self.image_preference_.weight)
        nn.init.xavier_uniform_(self.text_preference_.weight)

        # Precompute user preferences for image/text
        self.user_item_image_pref = torch.sparse.mm(self.adj.t(),
                                                    self.image_preference_(self.user_embedding.weight.to(self.device)))
        self.user_item_text_pref = torch.sparse.mm(self.adj.t(),
                                                   self.text_preference_(self.user_embedding.weight.to(self.device)))

        # Context and Generation Networks
        self.image_context_proj = nn.Linear(4096, self.embedding_dim).to(self.device)
        self.text_context_proj = nn.Linear(384, self.embedding_dim).to(self.device)
        self.fine_image_proj = nn.Linear(self.embedding_dim, 4096).to(self.device)
        self.fine_text_proj = nn.Linear(self.embedding_dim, 384).to(self.device)
        self.text_decoder = nn.Linear(self.embedding_dim, self.t_feat.shape[1]).to(self.device)
        self.image_decoder = nn.Linear(self.embedding_dim, self.v_feat.shape[1]).to(self.device)
        nn.init.xavier_uniform_(self.image_decoder.weight)
        nn.init.xavier_uniform_(self.text_decoder.weight)

        self.image_gen = nn.Sequential(nn.Linear(self.embedding_dim, 2048), nn.Tanh(), nn.Linear(2048, 4096),
                                       nn.Tanh()).to(self.device)
        self.text_gen = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
                                      nn.Linear(self.embedding_dim, 384), nn.Tanh()).to(self.device)

        # Cross-modal Translation
        self.image2text = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
                                        nn.Linear(self.embedding_dim, self.embedding_dim)).to(self.device)
        self.text2image = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
                                        nn.Linear(self.embedding_dim, self.embedding_dim)).to(self.device)

        # Query-Key Networks
        self.q_image_from_text = nn.Linear(384, 256).to(self.device)
        self.k_image_from_text = nn.Linear(4096, 256).to(self.device)
        self.q_text_from_image = nn.Linear(4096, 256).to(self.device)
        self.k_text_from_image = nn.Linear(384, 256).to(self.device)

        # Hyperparameters
        self.alpha = config['alpha']
        self.w_diff = config['w_diff']
        self.w_align = config['w_align']
        self.infoNCETemp = config['infoNCETemp']
        self.alignBMTemp = config['alignBMTemp']
        self.alignUITemp = config['alignUITemp']
        self.act_g = nn.Tanh()
        self.refresh_adj_counter = 0

        # Additional Projections
        self.image_from_text_proj = nn.Linear(4096, self.embedding_dim).to(self.device)
        self.text_from_image_proj = nn.Linear(384, self.embedding_dim).to(self.device)
        self.fine_image_gen = nn.Sequential(nn.Linear(4096 + 64, 1024), nn.ReLU(), nn.Linear(1024, 4096), nn.Tanh()).to(
            self.device)
        self.fine_text_gen = nn.Sequential(nn.Linear(384 + 64, 256), nn.ReLU(), nn.Linear(256, 384), nn.Tanh()).to(
            self.device)
        self.image_fuse_proj = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.text_fuse_proj = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.image_align_proj = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.text_align_proj = nn.Linear(self.embedding_dim, self.embedding_dim).to(self.device)
        self.init_mi_estimator()
        self.mse_loss = nn.MSELoss().to(self.device)

        # Optimizer Settings
        lr = config['learning_rate']
        weight_decay = config['weight_decay']

        # Core Parameters
        core_params = (list(self.user_embedding.parameters()) + list(self.item_id_embedding.parameters()) +
                       [self.prompt_complete, self.prompt_image_only, self.prompt_text_only] +
                       list(self.prompt_to_image_proj.parameters()) + list(self.prompt_to_text_proj.parameters()) +
                       list(self.image_scale_1.parameters()) + list(self.image_scale_2.parameters()) + list(
                    self.image_scale_3.parameters()) +
                       list(self.text_scale_1.parameters()) + list(self.text_scale_2.parameters()) + list(
                    self.text_scale_3.parameters()) +
                       list(self.image_fusion_attn.parameters()) + list(self.text_fusion_attn.parameters()) +
                       list(self.image_preference_.parameters()) + list(self.text_preference_.parameters()))

        # Freezable Parameters
        freezable_params = (list(self.image_encoder.parameters()) + list(self.text_encoder.parameters()) +
                            list(self.shared_encoder.parameters()) + list(self.image_proj.parameters()) + list(
                    self.text_proj.parameters()) +
                            list(self.item_fuser.parameters()) + list(self.image2text.parameters()) + list(
                    self.text2image.parameters()) +
                            list(self.q_image_from_text.parameters()) + list(self.k_image_from_text.parameters()) +
                            list(self.q_text_from_image.parameters()) + list(self.k_text_from_image.parameters()))

        # Add Graph Attention Parameters if enabled
        if self.use_graph_attention:
            core_params += list(self.image_gat.parameters()) + list(self.text_gat.parameters())

        self.optimizer_main = optim.Adam(core_params + freezable_params, lr=lr, weight_decay=weight_decay)
        self.optimizer_g = optim.Adam(list(self.fine_image_gen.parameters()) + list(self.fine_text_gen.parameters()) +
                                      list(self.image_pref_encoder.parameters()) + list(
            self.text_pref_encoder.parameters()),
                                      lr=1e-4, weight_decay=1e-5)
        self.optimizer_d = optim.Adam(
            list(self.discriminator_img.parameters()) + list(self.discriminator_txt.parameters()),
            lr=2e-4, weight_decay=1e-5)

        if getattr(self, 'use_feature_diffusion', False):
            self.freeze_modules([self.discriminator_img, self.discriminator_txt])

        self.freezable_modules = [self.image_encoder, self.text_encoder, self.shared_encoder, self.image_proj,
                                  self.text_proj, self.item_fuser,
                                  self.image2text, self.text2image, self.q_image_from_text, self.k_image_from_text,
                                  self.q_text_from_image, self.k_text_from_image]

        # Disentangled Encoding/Decoding
        self.z_dim = self.embedding_dim
        self.ib_beta_shared = config['ib_beta_shared']
        self.ib_beta_private = config['ib_beta_private']

        # ===== IB Module Switches & Weights (For Ablation) =====
        # Master Switch: Enable IB Disentanglement Module
        self.use_ib = bool(config['use_ib'])

        # Sub-switches
        self.use_ib_kl = bool(config['use_ib_kl'])                   # KL Regularization
        self.use_ib_disentangle = bool(config['use_ib_disentangle']) # Disentanglement Loss
        self.use_ib_latent_recon = bool(config['use_ib_latent_recon'])  # Latent Space Reconstruction Consistency

        # IB Internal Loss Weights
        self.w_disentangle = float(config['w_disentangle'])
        self.w_latent_recon = float(config['w_latent_recon'])

        self.ib_heads = nn.ModuleDict({
            's_img': nn.Linear(self.z_dim, self.z_dim * 2),
            's_txt': nn.Linear(self.z_dim, self.z_dim * 2),
            'p_img': nn.Linear(self.z_dim, self.z_dim * 2),
            'p_txt': nn.Linear(self.z_dim, self.z_dim * 2),
        }).to(self.device)
        for m in self.ib_heads.values():
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)

        self.img_shared_enc = nn.Sequential(nn.Linear(self.embedding_dim, self.z_dim), nn.Tanh()).to(self.device)
        self.txt_shared_enc = nn.Sequential(nn.Linear(self.embedding_dim, self.z_dim), nn.Tanh()).to(self.device)
        self.img_private_enc = nn.Sequential(nn.Linear(self.embedding_dim, self.z_dim), nn.Tanh()).to(self.device)
        self.txt_private_enc = nn.Sequential(nn.Linear(self.embedding_dim, self.z_dim), nn.Tanh()).to(self.device)

        # === Modality Specific Linear Transforms for Graph Construction W_m^{sh}, W_m^{sp} ===
        self.W_img_sh = nn.Linear(self.z_dim, self.embedding_dim).to(self.device)
        self.W_img_sp = nn.Linear(self.z_dim, self.embedding_dim).to(self.device)
        self.W_txt_sh = nn.Linear(self.z_dim, self.embedding_dim).to(self.device)
        self.W_txt_sp = nn.Linear(self.z_dim, self.embedding_dim).to(self.device)
        for m in [self.W_img_sh, self.W_img_sp, self.W_txt_sh, self.W_txt_sp]:
            nn.init.xavier_uniform_(m.weight)

        # Control relative weight of shared/specific in graph construction
        self.graph_lambda_img = float(config['graph_lambda_img']) if 'graph_lambda_img' in config else 0.5
        self.graph_lambda_txt = float(config['graph_lambda_txt']) if 'graph_lambda_txt' in config else 0.5

        self.dec_image = nn.Sequential(
            nn.Linear(self.z_dim * 2, 1024), nn.ReLU(),
            nn.Linear(1024, 4096)
        ).to(self.device)
        self.dec_text = nn.Sequential(
            nn.Linear(self.z_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, 384)
        ).to(self.device)

        # Stage-1 Generator
        self.stage1_shared_gen = nn.Sequential(
            nn.Linear(self.z_dim + 64, self.z_dim), nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim)
        ).to(self.device)
        self.stage1_priv_img_gen = nn.Sequential(
            nn.Linear(self.z_dim + 64, self.z_dim), nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim)
        ).to(self.device)
        self.stage1_priv_txt_gen = nn.Sequential(
            nn.Linear(self.z_dim + 64, self.z_dim), nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim)
        ).to(self.device)

        # Short-term / Long-term Interest Modeling
        self.max_seq_len = config['max_seq_len']
        self.short_encoder = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        nn.init.constant_(self.gate_mlp[-1].bias, 1.0)
        nn.init.constant_(self.gate_mlp[-1].bias,
                          0.0 if self.use_fixed_short_weight else self.gate_bias_init)

        # User channel weights (ID / Image / Text), personalized per user
        self.modal_weight_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, 3)  # [w_id, w_img, w_txt]
        ).to(self.device)

        # ===== Stage-2 Latent Space Diffusion =====
        cond_latent = 64 + self.embedding_dim
        self.diff_shared = DiffusionRefiner(
            feat_dim=self.z_dim, cond_dim=cond_latent,
            timesteps=self.diff_timesteps, guidance_scale=self.cfg_scale
        ).to(self.device)
        self.diff_priv_img = DiffusionRefiner(
            feat_dim=self.z_dim, cond_dim=cond_latent,
            timesteps=self.diff_timesteps, guidance_scale=self.cfg_scale
        ).to(self.device)
        self.diff_priv_txt = DiffusionRefiner(
            feat_dim=self.z_dim, cond_dim=cond_latent,
            timesteps=self.diff_timesteps, guidance_scale=self.cfg_scale
        ).to(self.device)

        # -- Projection for two types of conditions (both compressed to embedding_dim) --
        # Shared Chain: Existing Modal Shared Semantics + Item ID Embedding
        self.cond_proj_shared = nn.Linear(self.embedding_dim * 2, self.embedding_dim).to(self.device)
        # Private Chain: User Preference (Short+Long) + Item ID Embedding
        self.cond_proj_user = nn.Linear(self.embedding_dim * 2, self.embedding_dim).to(self.device)

        # Global Prompt: 64d, time-independent prior
        self.global_prompt = nn.Parameter(torch.randn(64), requires_grad=True)

        # Add new parameters to optimizer
        self.optimizer_main.add_param_group({
            "params": list(self.img_shared_enc.parameters())
                      + list(self.txt_shared_enc.parameters())
                      + list(self.img_private_enc.parameters())
                      + list(self.txt_private_enc.parameters())
                      + list(self.dec_image.parameters())
                      + list(self.dec_text.parameters())
                      + list(self.stage1_shared_gen.parameters())
                      + list(self.stage1_priv_img_gen.parameters())
                      + list(self.stage1_priv_txt_gen.parameters())
                      + list(self.W_img_sh.parameters())
                      + list(self.W_img_sp.parameters())
                      + list(self.W_txt_sh.parameters())
                      + list(self.W_txt_sp.parameters())
        })
        # Diffusion Optimizer
        self.optimizer_diff = optim.Adam(
            list(getattr(self, "diff_img", nn.Sequential()).parameters()) +
            list(getattr(self, "diff_txt", nn.Sequential()).parameters()) +
            list(self.diff_shared.parameters()) + list(self.diff_priv_img.parameters()) + list(
                self.diff_priv_txt.parameters()),
            lr=1e-4, weight_decay=1e-5
        )

        # Move to Device
        self.logger.info(f"Recursively moving all model components to device: {self.device}")
        if not self.enable_missing_modal_generation:
            self.freeze_modules([self.diff_img, self.diff_txt,
                                 self.diff_shared, self.diff_priv_img, self.diff_priv_txt])
        self.to(self.device)
        self.logger.info("All components successfully moved to device.")

        # Save dataset pointer
        self._dataset = dataset

        # Ensure user sequence cache is built
        self._build_user_seq_cache(dataset)

        self.use_graph_attention = False  # Force disable high-memory GAT path

    def _get_user_modal_weights(self, user_emb_batch):
        # user_emb_batch: [B, d]
        w = self.modal_weight_head(user_emb_batch)  # [B, 3]
        return torch.softmax(w, dim=-1)  # Sum of each row is 1

    def pre_epoch_processing(self):
        # Can clear counters here if needed, leaving empty for now
        return None

    def post_epoch_processing(self):
        """Called after each epoch to perform progressive graph updates based on maturity"""
        self.current_epoch += 1

        info = f"[D2GRec] Epoch {self.current_epoch} end."

        # Update only if progressive integration is enabled & missing modalities/new items exist
        if self.missing_modal and self.progressive_integration:
            # Maturity threshold should be in config
            self.maturity_trigger_threshold = float(self.config['maturity_trigger_threshold'])

            maturity_scores = self._compute_item_maturity()
            if (maturity_scores >= self.maturity_trigger_threshold).any():
                # Only update items reaching the threshold
                self.update_adj(maturity_mask=(maturity_scores >= self.maturity_trigger_threshold))
                info += f" Updated adjacency matrix (progressive_integration={self.progressive_integration})."

        return info

    def _compute_item_maturity(self):
        """Compute item maturity score (3 dimensions: Interaction Count + Graph Connectivity + Generation Quality)"""
        # 1. Maturity based on interaction count
        interaction_maturity = torch.clamp(self.item_interaction_counts / 100.0, 0, 1)

        # 2. Maturity based on neighbor connections (Graph Connectivity)
        if hasattr(self, 'image_adj') and hasattr(self, 'text_adj'):
            image_degree = torch.sparse.sum(self.image_adj, dim=1).to_dense()
            text_degree = torch.sparse.sum(self.text_adj, dim=1).to_dense()
            degree_maturity = torch.clamp((image_degree + text_degree) / (2 * self.max_k), 0, 1)
        else:
            degree_maturity = torch.zeros_like(interaction_maturity)

        # 3. Maturity based on generation quality (Cycle Consistency)
        if self.missing_modal and hasattr(self, 'image2text') and hasattr(self, 'text2image'):
            generation_quality_maturity = self._compute_generation_quality()
        else:
            generation_quality_maturity = torch.zeros_like(interaction_maturity)

        # Combined Maturity Score (Weighted)
        w_interact = float(self.config['maturity_weight_interaction'])
        w_degree = float(self.config['maturity_weight_degree'])
        w_quality = float(self.config['maturity_weight_quality'])

        self.item_maturity_scores = (
            w_interact * interaction_maturity +
            w_degree * degree_maturity +
            w_quality * generation_quality_maturity
        )

        return self.item_maturity_scores

    @torch.no_grad()
    def _compute_generation_quality(self):
        """
        Evaluate generation quality based on Cycle Consistency:
        - Missing Image Only: Validate generated image with real text (Gen Image -> Recon Text vs Real Text)
        - Missing Text Only: Validate generated text with real image (Gen Text -> Recon Image vs Real Image)
        - Both Missing: Degrade to interaction count proxy

        Calculation is identical for training and inference as missing modality flags are predetermined.
        """
        quality_scores = torch.ones(self.n_items, device=self.device)

        if not self.missing_modal:
            return quality_scores

        # Get current modality embeddings (encoded)
        item_image_emb = torch.sigmoid(
            self.shared_encoder(torch.sigmoid(self.image_encoder(self.image_embedding.weight)))
        )
        item_text_emb = torch.sigmoid(
            self.shared_encoder(torch.sigmoid(self.text_encoder(self.text_embedding.weight)))
        )

        # ===== Case 1: Missing Image Only, Real Text Exists =====
        only_v_missing = np.setdiff1d(self.missing_items_v, self.missing_items_t)
        if len(only_v_missing) > 0:
            idx_v = torch.from_numpy(only_v_missing).long().to(self.device)

            # Generated Image Embedding
            gen_img_emb = item_image_emb[idx_v]

            # Real Text Embedding (Exists in train/infer)
            real_txt_emb = item_text_emb[idx_v]

            # Cycle Consistency: Gen Image -> Recon Text
            reconstructed_txt = self.image2text(gen_img_emb)

            # Cycle Error (MSE)
            cycle_error = F.mse_loss(
                reconstructed_txt, real_txt_emb, reduction='none'
            ).mean(dim=-1)

            # Convert to quality score: exp(-error), normalized to [0,1]
            max_error = cycle_error.max()
            if max_error > 1e-6:
                quality_v = torch.exp(-cycle_error / max_error)
            else:
                quality_v = torch.ones_like(cycle_error)

            quality_scores[idx_v] = quality_v

        # ===== Case 2: Missing Text Only, Real Image Exists =====
        only_t_missing = np.setdiff1d(self.missing_items_t, self.missing_items_v)
        if len(only_t_missing) > 0:
            idx_t = torch.from_numpy(only_t_missing).long().to(self.device)

            # Generated Text Embedding
            gen_txt_emb = item_text_emb[idx_t]

            # Real Image Embedding
            real_img_emb = item_image_emb[idx_t]

            # Cycle Consistency: Gen Text -> Recon Image
            reconstructed_img = self.text2image(gen_txt_emb)

            # Cycle Error
            cycle_error = F.mse_loss(
                reconstructed_img, real_img_emb, reduction='none'
            ).mean(dim=-1)

            # Convert to quality score
            max_error = cycle_error.max()
            if max_error > 1e-6:
                quality_t = torch.exp(-cycle_error / max_error)
            else:
                quality_t = torch.ones_like(cycle_error)

            quality_scores[idx_t] = quality_t

        # ===== Case 3: Both Missing -> Cannot use cycle consistency, degrade to interaction count =====
        both_missing = np.intersect1d(self.missing_items_v, self.missing_items_t)
        if len(both_missing) > 0:
            idx_both = torch.from_numpy(both_missing).long().to(self.device)

            # Use interaction count as quality proxy
            interaction_quality = torch.clamp(
                self.item_interaction_counts[idx_both] / 100.0, 0, 1
            )
            quality_scores[idx_both] = interaction_quality

        return quality_scores

    def _get_modal_graph_features(self):
        """
        Build modality features for KNN based on IB disentangled shared / specific factors:
        Returns (img_feat, txt_feat), each [n_items, d].
        If IB is disabled, degrades to using original modality embeddings directly.
        """
        # Fallback to original features if IB is off
        if not getattr(self, 'use_ib', False):
            img_feat = self.image_embedding.weight.detach()
            txt_feat = self.text_embedding.weight.detach()
            return img_feat, txt_feat

        with torch.no_grad():
            # Disentangled Encoding: Get z_s_img, z_v, z_s_txt, z_t
            z_s_img, z_v, z_s_txt, z_t, _ = self.encode_disentangled_ib()

            # Image Modality Graph Features g_i^{(img,0)}
            img_feat = self.W_img_sh(z_s_img) + self.graph_lambda_img * self.W_img_sp(z_v)
            # Text Modality Graph Features g_i^{(txt,0)}
            txt_feat = self.W_txt_sh(z_s_txt) + self.graph_lambda_txt * self.W_txt_sp(z_t)

        return img_feat.detach(), txt_feat.detach()

    def _two_stage_generate(self, base_shared, base_priv, prompt,
                            cond_shared, decoder, stage1_priv_gen):
        """
        base_shared/base_priv: Shared / Private latent from IB encoding
        prompt: 64d prompt
        cond_shared: diffusion condition (shared latent or feat condition)
        decoder: dec_image / dec_text
        stage1_priv_gen: stage1_priv_img_gen / stage1_priv_txt_gen
        """
        # ---------- Stage-1: coarse ----------
        if self.use_stage1_gen:
            z_s_coarse = self.stage1_shared_gen(torch.cat([base_shared, prompt], dim=-1))
            z_p_coarse = stage1_priv_gen(torch.cat([base_priv, prompt], dim=-1))
        else:
            # w/o Stage-1: Use IB latent directly as coarse init
            z_s_coarse = base_shared
            z_p_coarse = base_priv

        # ---------- Stage-2: diffusion refine ----------
        if self.use_stage2_gen:
            z_s_refined = self.diff_shared.sample_cfg(z_s_coarse, cond_shared)
            feat = decoder(torch.cat([z_s_refined, z_p_coarse], dim=-1))
        else:
            # w/o Stage-2: Decode coarse directly
            feat = decoder(torch.cat([z_s_coarse, z_p_coarse], dim=-1))

        return feat

    def _build_adaptive_image_graph(self):
        """Build adaptive image modality graph (IB Disentangled Version)"""
        # 1. Use g_img combined from IB disentangled shared/specific as graph features
        img_feat, _ = self._get_modal_graph_features()
        image_adj = build_sim(img_feat)

        if self.adaptive_knn:
            maturity_scores = self._compute_item_maturity()
            image_adj = self.adaptive_knn_builder.build_adaptive_knn(image_adj, maturity_scores)
        else:
            image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)

        # Handle missing modalities
        if self.missing_modal and hasattr(self, 'missing_items_v') and len(self.missing_items_v) > 0:
            if self.progressive_integration:
                # Progressive integration: Isolate only immature new items
                immature_missing = self._get_immature_items(self.missing_items_v)
                image_adj[immature_missing, :] = 0.0
                image_adj[:, immature_missing] = 0.0
                image_adj.fill_diagonal_(1.0)
            else:
                # Traditional method: Complete isolation
                image_adj[self.missing_items_v, :] = 0.0
                image_adj[:, self.missing_items_v] = 0.0
                image_adj.fill_diagonal_(1.0)

        self.image_adj = compute_normalized_laplacian(image_adj).to(self.device).to_sparse_coo()
        del image_adj

    def _build_adaptive_text_graph(self):
        """Build adaptive text modality graph"""
        _, txt_feat = self._get_modal_graph_features()
        text_adj = build_sim(txt_feat)

        if self.adaptive_knn:
            maturity_scores = self._compute_item_maturity()
            text_adj = self.adaptive_knn_builder.build_adaptive_knn(text_adj, maturity_scores)
        else:
            text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)

        # Handle missing modalities
        if self.missing_modal and hasattr(self, 'missing_items_t') and len(self.missing_items_t) > 0:
            if self.progressive_integration:
                # Progressive integration: Isolate only immature new items
                immature_missing = self._get_immature_items(self.missing_items_t)
                text_adj[immature_missing, :] = 0.0
                text_adj[:, immature_missing] = 0.0
                text_adj.fill_diagonal_(1.0)
            else:
                # Traditional method: Complete isolation
                text_adj[self.missing_items_t, :] = 0.0
                text_adj[:, self.missing_items_t] = 0.0
                text_adj.fill_diagonal_(1.0)

        self.text_adj = compute_normalized_laplacian(text_adj).to(self.device).to_sparse_coo()
        del text_adj

    def _get_immature_items(self, item_indices):
        """Get indices of immature items"""
        maturity_scores = self._compute_item_maturity()
        immature_mask = maturity_scores[item_indices] < self.maturity_threshold
        return item_indices[immature_mask.cpu().numpy()]

    def _progressive_graph_propagation(self, features, adj_matrix, modality="image"):
        """Graph propagation (Low memory version: uses sparse multiplication only, no NxN attention)"""
        # Directly use sparse adjacency matrix multiplication to avoid to_dense and N^2 memory overhead
        return torch.sparse.mm(adj_matrix, features)

    def update_item_interactions(self, items):
        """Update item interaction counts"""
        unique_items, counts = torch.unique(items, return_counts=True)
        self.item_interaction_counts[unique_items] += counts.float()

    def progressive_update_adj(self):
        """Progressively update adjacency matrix"""
        print(f"Epoch {self.current_epoch}: Progressive adjacency matrix update")

        with torch.no_grad():
            # Recompute maturity
            maturity_scores = self._compute_item_maturity()

            # Find items that recently became mature
            newly_mature = (maturity_scores >= self.maturity_threshold) & (
                    maturity_scores < self.maturity_threshold + 0.1
            )

            if newly_mature.any():
                # Move to CPU, convert to python int list for CPU-only logic
                mature_indices = torch.where(newly_mature)[0].cpu().tolist()

                # Convert missing indices to set for faster lookup
                missing_v_set = set(map(int, np.array(getattr(self, "missing_items_v", []))))
                missing_t_set = set(map(int, np.array(getattr(self, "missing_items_t", []))))

                # Gradually add connections for newly mature items
                for idx in mature_indices:
                    if idx in missing_v_set:
                        self._gradually_connect_item(idx, modality="image")
                    if idx in missing_t_set:
                        self._gradually_connect_item(idx, modality="text")

    def _gradually_connect_item(self, item_idx, modality="image"):
        """Gradually add graph connections for a single item"""
        with torch.no_grad():
            if modality == "image":
                # Recompute similarity
                similarities = torch.cosine_similarity(
                    self.image_embedding.weight[item_idx:item_idx + 1],
                    self.image_embedding.weight,
                    dim=1
                )
                adj_matrix = self.image_adj.to_dense()
            else:
                similarities = torch.cosine_similarity(
                    self.text_embedding.weight[item_idx:item_idx + 1],
                    self.text_embedding.weight,
                    dim=1
                )
                adj_matrix = self.text_adj.to_dense()

            # Get adaptive k value
            maturity = self.item_maturity_scores[item_idx]
            k = self.adaptive_knn_builder.get_adaptive_k(maturity).item()

            # Find top k neighbors
            _, top_k_indices = torch.topk(similarities, k + 1)
            top_k_indices = top_k_indices[top_k_indices != item_idx][:k]

            # Progressive update: control update strength with integration_rate
            new_connections = similarities[top_k_indices] * self.integration_rate
            adj_matrix[item_idx, top_k_indices] = new_connections
            adj_matrix[top_k_indices, item_idx] = new_connections

            # Update corresponding adjacency matrix
            if modality == "image":
                self.image_adj = compute_normalized_laplacian(adj_matrix).to_sparse_coo()
            else:
                self.text_adj = compute_normalized_laplacian(adj_matrix).to_sparse_coo()

    def mge(self):
        item_image_emb = F.sigmoid(self.image_encoder(self.image_embedding.weight))
        item_text_emb = F.sigmoid(self.text_encoder(self.text_embedding.weight))

        item_image_emb = F.sigmoid(self.shared_encoder(item_image_emb))
        item_text_emb = F.sigmoid(self.shared_encoder(item_text_emb))

        return item_image_emb, item_text_emb

    def cge(self, user_emb, item_emb, adj):
        ego_embeddings = torch.cat((user_emb, item_emb), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        user_embeddings, item_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        del ego_embeddings, side_embeddings

        return user_embeddings, item_embedding

    def _broadcast_prompt(self, n: int, device):
        """
        Broadcast self.prompt_complete to match latent variable entries.
        - If prompt_complete is [L,dim] (multi-token), average over tokens to get [1,dim] global prompt.
        - If [dim], convert to [1,dim].
        Then expand to [n,dim].
        """
        P = getattr(self, "prompt_complete", None)
        if P is None:
            return None

        # P: [L, dim] or [dim]
        if P.dim() == 1:
            # [dim] -> [1,dim]
            P = P.unsqueeze(0)
        elif P.dim() == 2 and P.size(0) > 1:
            # [L,dim] (multi-token) -> Average to get global prompt: [1,dim]
            P = P.mean(dim=0, keepdim=True)

        # P.shape[0] is now 1, safe to expand to [n,dim]
        P = P.expand(n, P.size(1)).contiguous()
        return P.to(device)

    def _build_shared_condition(self,
                                item_idx: torch.Tensor,
                                z_s_img: torch.Tensor,
                                z_s_txt: torch.Tensor):
        """
        Stage-2 Shared Chain Condition:
        Use item's shared latent on image/text side as semantic condition.
        item_idx: [K]
        """
        if item_idx.numel() == 0:
            return None

        # [K,2*z_dim] -> [K, z_dim]
        core = torch.cat([z_s_img[item_idx], z_s_txt[item_idx]], dim=-1)
        core = self.cond_proj_shared(core)  # [K, embedding_dim]

        # Here n = K (number of items)
        prompt = self._broadcast_prompt(core.size(0), core.device)  # [K,64] or None
        if prompt is None:
            return core

        # Final condition: [K, 64 + embedding_dim]
        return torch.cat([prompt, core], dim=-1)

    def _build_private_condition(self,
                                 user_batch_emb: torch.Tensor,
                                 pos_items: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Stage-2 Private Chain Condition:
        Use "User Preference + Item Semantics" as condition.
        user_batch_emb: [B, d]  User preference after gating, corresponds to pos_items
        pos_items:      [B]     Positive sample item IDs in current batch
        """
        if user_batch_emb is None or user_batch_emb.numel() == 0:
            return None

        device = user_batch_emb.device
        pos_items = pos_items.to(device)

        # Use item_id_embedding for item semantics
        item_repr = self.item_id_embedding(pos_items)  # [B, d]

        # Concat User Preference + Item Semantics -> [B, 2d]
        core = torch.cat([user_batch_emb, item_repr], dim=-1)  # [B, 2*d]
        core = self.cond_proj_user(core)  # [B, d]

        # Concat 64d prompt -> [B, 64 + d]
        prompt = self._broadcast_prompt(core.size(0), device)
        if prompt is None:
            return core
        return torch.cat([prompt, core], dim=-1)

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction

        # Update interaction counts
        self.update_item_interactions(torch.cat([pos_items, neg_items]))

        # ---------------- Two-stage Generation Ablation Switches ----------------
        # Stage-1 coarse generator
        use_stage1 = getattr(self, "use_stage1_gen", True)
        # Stage-2 diffusion refiner (Default to use_feature_diffusion)
        use_stage2 = getattr(self, "use_stage2_gen", getattr(self, "use_feature_diffusion", False))

        enable_gen = (
                getattr(self, "enable_missing_modal_generation", False)
                and (use_stage1 or use_stage2)
        )
        # ---------------------------------------------------

        # Base Embeddings
        user_collab_emb, item_collab_emb = self.cge(
            self.user_embedding.weight,
            self.item_id_embedding.weight,
            self.norm_adj
        )

        # Multimodal Features (Stage-0)
        item_image_emb, item_text_emb = self.mge()

        # ===== IB Disentanglement (Master Switch + 3 Sub-switches) =====
        loss_disentangle = torch.tensor(0.0, device=self.device)
        loss_ib = torch.tensor(0.0, device=self.device)
        loss_recon_latent = torch.tensor(0.0, device=self.device)
        # Stage-2 Diffusion Loss (Shared + Private)
        loss_diff_stage2 = torch.tensor(0.0, device=self.device)

        if self.use_ib:
            # Disentangled Encoding
            z_s_img, z_v, z_s_txt, z_t, kl_dict = self.encode_disentangled_ib()

            # Disentanglement Loss (Shared Consistency + Private Orthogonality)
            if self.use_ib_disentangle:
                loss_disentangle = self.disentangle_losses(z_s_img, z_v, z_s_txt, z_t)

            # Information Bottleneck KL Loss
            if self.use_ib_kl:
                loss_kl_shared = 0.5 * (kl_dict['kl_s_img'] + kl_dict['kl_s_txt'])
                loss_kl_private = 0.5 * (kl_dict['kl_p_img'] + kl_dict['kl_p_txt'])
                loss_ib = (
                        self.ib_beta_shared * loss_kl_shared +
                        self.ib_beta_private * loss_kl_private
                )

            # Latent Space Reconstruction Consistency
            if self.use_ib_latent_recon:
                recon_img = self.decode_modalities(z_s_img, z_v, modality="image")
                recon_txt = self.decode_modalities(z_s_txt, z_t, modality="text")
                loss_recon_latent = (
                        F.mse_loss(recon_img, self.image_embedding.weight) +
                        F.mse_loss(recon_txt, self.text_embedding.weight)
                )

        # Use improved graph propagation (Modal Graph + Maturity Gating)
        for _ in range(min(self.n_mm_layers, 2)):
            item_image_emb = self._progressive_graph_propagation(item_image_emb, self.image_adj, "image")
            item_text_emb = self._progressive_graph_propagation(item_text_emb, self.text_adj, "text")

        # User Modal Features: Aggregate items to user along interaction graph
        user_image_emb = torch.sparse.mm(self.adj, item_image_emb) * self.num_inters[:self.n_users]
        user_text_emb = torch.sparse.mm(self.adj, item_text_emb) * self.num_inters[:self.n_users]

        # ===== Alignment Loss: Image/Text + Collaborative =====
        all_items_in_batch = torch.unique(torch.cat((pos_items, neg_items))).cpu().numpy()
        tv_index = np.setdiff1d(
            all_items_in_batch,
            np.union1d(self.missing_items_t, self.missing_items_v)
        )

        loss_alignment = self.InfoNCE(
            item_image_emb[tv_index],
            item_text_emb[tv_index],
            temperature=self.infoNCETemp
        )

        loss_modality_alignment = self.InfoNCE(
            user_collab_emb[users],
            item_collab_emb[pos_items],
            temperature=self.alignUITemp
        )
        loss_modality_alignment += self.InfoNCE(
            user_image_emb[users] + user_text_emb[users],
            item_image_emb[pos_items] + item_text_emb[pos_items],
            temperature=self.alignUITemp
        )
        loss_modality_alignment += self.InfoNCE(
            item_collab_emb[pos_items],
            item_image_emb[pos_items] + item_text_emb[pos_items],
            temperature=self.alignBMTemp
        )
        loss_modality_alignment += self.InfoNCE(
            user_collab_emb[users],
            user_image_emb[users] + user_text_emb[users],
            temperature=self.alignBMTemp
        )

        loss_align_total = self.w_align * (loss_alignment + loss_modality_alignment)

        # ===== Generation and Reconstruction Loss (Stage-1 Coarse Gen) =====
        # >>> Ablation: If use_stage1=False, this part is 0
        loss_gen_total = torch.tensor(0.0, device=self.device)
        if enable_gen and use_stage1:
            t_index = np.setdiff1d(all_items_in_batch, self.missing_items_t)
            v_index = np.setdiff1d(all_items_in_batch, self.missing_items_v)

            generated_image_from_text = self.text2image(self.perturb(item_text_emb))
            generated_text_from_image = self.image2text(self.perturb(item_image_emb))

            loss_generation = MSELoss(generated_image_from_text[v_index], item_image_emb[v_index])
            loss_generation += MSELoss(generated_text_from_image[t_index], item_text_emb[t_index])

            loss_reconstruction = self.calculate_recon_loss(
                image=item_image_emb.detach(),
                text=item_text_emb.detach()
            )
            loss_gen_total = loss_generation + loss_reconstruction
        # =======================================================

        # Regularization Loss
        loss_regularization = self.reg_loss(
            user_collab_emb[users],
            item_collab_emb[pos_items],
            item_collab_emb[neg_items],
            item_image_emb[pos_items],
            item_text_emb[pos_items]
        )

        # ===== BPR + Long/Short Interest =====
        item_final_emb = item_collab_emb + (item_image_emb + item_text_emb) / 2
        user_long_all = user_collab_emb + (user_image_emb + user_text_emb) / 2
        user_short_batch = self._encode_short_term(item_final_emb, users)

        # Gated fusion to get user preference
        user_long_batch = user_long_all[users]
        if self.use_fixed_short_weight:
            gate_alpha = torch.full(
                (user_short_batch.size(0), 1),
                float(self.short_term_weight),
                device=self.device
            )
        else:
            gate_alpha = torch.sigmoid(
                self.gate_mlp(torch.cat([user_long_batch, user_short_batch], dim=1))
            )
        user_batch_emb = gate_alpha * user_short_batch + (1.0 - gate_alpha) * user_long_batch

        # ===== Stage-2 Diffusion Refinement (IB Latent Space) =====
        # >>> Ablation: If use_stage2=False, this part is 0 (and no momentum write-back)
        if enable_gen and use_stage2 and self.use_ib and hasattr(self, "diff_shared"):
            # 1) Shared Chain: condition = Existing Modality Shared Semantics (Train only on items with both image/text)
            if tv_index.size > 0:
                idx_torch = torch.from_numpy(tv_index).long().to(self.device)
                cond_shared = self._build_shared_condition(idx_torch, z_s_img, z_s_txt)
                if cond_shared is not None:
                    t_shared = torch.randint(
                        0, self.diff_timesteps,
                        (idx_torch.size(0),),
                        device=self.device
                    )
                    eps_pred_s, noise_s = self.diff_shared(
                        z_s_img[idx_torch], cond_shared, t_shared, train_cfg=True
                    )
                    loss_shared = F.mse_loss(eps_pred_s, noise_s)
                else:
                    loss_shared = torch.tensor(0.0, device=self.device)
            else:
                loss_shared = torch.tensor(0.0, device=self.device)

            # 2) Private Chain: condition = User Preference (Current batch user-item positive pairs)
            cond_priv = self._build_private_condition(user_batch_emb, pos_items)
            if cond_priv is not None:
                t_img = torch.randint(
                    0, self.diff_timesteps,
                    (pos_items.size(0),),
                    device=self.device
                )
                eps_pred_v, noise_v = self.diff_priv_img(
                    z_v[pos_items], cond_priv, t_img, train_cfg=True
                )
                t_txt = torch.randint(
                    0, self.diff_timesteps,
                    (pos_items.size(0),),
                    device=self.device
                )
                eps_pred_t, noise_t = self.diff_priv_txt(
                    z_t[pos_items], cond_priv, t_txt, train_cfg=True
                )
                loss_private = 0.5 * (
                        F.mse_loss(eps_pred_v, noise_v) +
                        F.mse_loss(eps_pred_t, noise_t)
                )
            else:
                loss_private = torch.tensor(0.0, device=self.device)

            loss_diff_stage2 = 0.5 * (loss_shared + loss_private)

            # ===== Momentum Write-back during Training (Only execute if Stage-2 is ON) =====
            if self.missing_modal and self.training:
                maturity_scores = self._compute_item_maturity()

                # --- Write back for items missing Image ---
                if len(self.missing_items_v) > 0:
                    batch_v_missing = np.intersect1d(pos_items.cpu().numpy(), self.missing_items_v)
                    if len(batch_v_missing) > 0:
                        idx_v_missing = torch.from_numpy(batch_v_missing).long().to(self.device)
                        gen_img_feat = self.dec_image(
                            torch.cat([z_s_img[idx_v_missing], z_v[idx_v_missing]], dim=-1)
                        )
                        old_img_feat = self.image_embedding.weight.data[idx_v_missing]
                        rho_v = maturity_scores[idx_v_missing].unsqueeze(1)
                        self.image_embedding.weight.data[idx_v_missing] = (
                                (1 - self.momentum_eta * rho_v) * old_img_feat +
                                self.momentum_eta * rho_v * gen_img_feat.detach()
                        )

                # --- Write back for items missing Text ---
                if len(self.missing_items_t) > 0:
                    batch_t_missing = np.intersect1d(pos_items.cpu().numpy(), self.missing_items_t)
                    if len(batch_t_missing) > 0:
                        idx_t_missing = torch.from_numpy(batch_t_missing).long().to(self.device)
                        gen_txt_feat = self.dec_text(
                            torch.cat([z_s_txt[idx_t_missing], z_t[idx_t_missing]], dim=-1)
                        )
                        old_txt_feat = self.text_embedding.weight.data[idx_t_missing]
                        rho_t = maturity_scores[idx_t_missing].unsqueeze(1)
                        self.text_embedding.weight.data[idx_t_missing] = (
                                (1 - self.momentum_eta * rho_t) * old_txt_feat +
                                self.momentum_eta * rho_t * gen_txt_feat.detach()
                        )
        # =======================================================

        # === User Personalized Modal Weights ===
        w = self._get_user_modal_weights(user_batch_emb)  # [B,3] -> (w_id, w_img, w_txt)

        # Per-modal Scoring (Pos/Neg)
        pos_id = (user_batch_emb * item_collab_emb[pos_items]).sum(dim=1)
        pos_img = (user_batch_emb * item_image_emb[pos_items]).sum(dim=1)
        pos_txt = (user_batch_emb * item_text_emb[pos_items]).sum(dim=1)

        neg_id = (user_batch_emb * item_collab_emb[neg_items]).sum(dim=1)
        neg_img = (user_batch_emb * item_image_emb[neg_items]).sum(dim=1)
        neg_txt = (user_batch_emb * item_text_emb[neg_items]).sum(dim=1)

        pos_scores = w[:, 0] * pos_id + w[:, 1] * pos_img + w[:, 2] * pos_txt
        neg_scores = w[:, 0] * neg_id + w[:, 1] * neg_img + w[:, 2] * neg_txt

        pos_item_batch_emb = item_final_emb[pos_items]
        neg_item_batch_emb = item_final_emb[neg_items]
        loss_bpr = self.bpr_loss(user_batch_emb, pos_item_batch_emb, neg_item_batch_emb)

        total_loss = (
                loss_bpr +
                loss_gen_total +
                loss_align_total +
                loss_regularization +
                self.w_disentangle * loss_disentangle +
                self.w_latent_recon * loss_recon_latent +
                loss_ib +
                self.w_diff * loss_diff_stage2  # Diffusion loss (0 if use_stage2=False)
        )

        return total_loss

    def update_adj(self, maturity_mask=None):
        """Update adjacency matrix (now uses progressive method)"""
        # Store mask if passed externally
        if maturity_mask is not None:
            self.maturity_mask = maturity_mask

        if self.progressive_integration:
            self.progressive_update_adj()
        else:
            # Traditional update method
            self._traditional_update_adj()

    def _traditional_update_adj(self):
        """Traditional adjacency matrix update (Based on IB Disentangled Graph Features)"""
        print(f"Epoch {self.current_epoch}: Traditional adjacency matrix update")
        with torch.no_grad():
            t_index = self.missing_items_t
            v_index = self.missing_items_v

            # 1. Get current modal features for graph construction (g_img, g_txt)
            img_feat, txt_feat = self._get_modal_graph_features()
            img_feat = img_feat.detach()
            txt_feat = txt_feat.detach()

            # ===== Image Adjacency Update =====
            self.image_adj = self.image_adj.cpu().to_dense()
            torch.cuda.empty_cache()

            # Build similarity / KNN based on IB features
            image_adj = build_sim(img_feat)
            if self.adaptive_knn:
                maturity_scores = self._compute_item_maturity()
                image_adj = self.adaptive_knn_builder.build_adaptive_knn(image_adj, maturity_scores)
            else:
                image_adj = build_knn_neighbourhood(image_adj, topk=self.knn_k)
            image_adj = compute_normalized_laplacian(image_adj).cpu()

            # EMA Update for missing items rows only
            self.image_adj[v_index] = (
                    image_adj[v_index] * self.alpha
                    + self.image_adj[v_index] * (1 - self.alpha)
            )
            self.image_adj = self.image_adj.to_sparse_coo()
            del image_adj

            # ===== Text Adjacency Update =====
            self.text_adj = self.text_adj.cpu().to_dense()
            torch.cuda.empty_cache()

            text_adj = build_sim(txt_feat)
            if self.adaptive_knn:
                maturity_scores = self._compute_item_maturity()
                text_adj = self.adaptive_knn_builder.build_adaptive_knn(text_adj, maturity_scores)
            else:
                text_adj = build_knn_neighbourhood(text_adj, topk=self.knn_k)
            text_adj = compute_normalized_laplacian(text_adj).cpu()

            # EMA Update for missing items rows only
            self.text_adj[t_index] = (
                    text_adj[t_index] * self.alpha
                    + self.text_adj[t_index] * (1 - self.alpha)
            )
            self.text_adj = self.text_adj.to_sparse_coo()
            del text_adj

            torch.cuda.empty_cache()
            self.image_adj = self.image_adj.to(self.device)
            self.text_adj = self.text_adj.to(self.device)

    def _build_user_seq_cache(self, dataset):
        try:
            result = dataset.get_user_sequences(self.max_seq_len)
            if isinstance(result, tuple):
                # Assume first element is seq dict
                seq_dict = result[0]
            else:
                seq_dict = result
            if not isinstance(seq_dict, dict):
                raise TypeError("Expected dict or tuple with dict as first element.")
        except Exception as e:
            self.logger.warning(f'get_user_sequences not available ({e}); falling back to empty sequences.')
            seq_dict = {}

        n, L = self.n_users, self.max_seq_len
        self._user_seq = torch.full((n, L), fill_value=-1, dtype=torch.long)
        self._user_seq_len = torch.zeros(n, dtype=torch.long)

        for u in range(n):
            seq = seq_dict.get(u, [])
            l = min(len(seq), L)
            if l > 0:
                self._user_seq[u, -l:] = torch.as_tensor(seq[-l:], dtype=torch.long)
                self._user_seq_len[u] = l

    def _encode_short_term(self, item_emb, users):
        device = self.device

        users_cpu = users.detach().cpu()
        seq = self._user_seq[users_cpu].to(device)
        lens = self._user_seq_len[users_cpu]

        B = users.size(0)
        if lens.sum().item() == 0:
            return torch.zeros(B, self.embedding_dim, device=device)

        idx = seq.clone()
        idx[idx < 0] = 0
        seq_emb = item_emb[idx]

        out = torch.zeros(B, self.embedding_dim, device=device)
        mask = lens > 0
        if mask.any():
            seq_emb_nz = seq_emb[mask]
            lens_nz = lens[mask].cpu()
            packed = pack_padded_sequence(seq_emb_nz, lens_nz, batch_first=True, enforce_sorted=False)
            _, h = self.short_encoder(packed)
            out[mask] = h[-1]
        return out

    def freeze_modules(self, modules):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        print(f"Froze parameters for {len(modules)} modules")

    def encode_disentangled_ib(self):
        item_image_emb = torch.sigmoid(
            self.shared_encoder(torch.sigmoid(self.image_encoder(self.image_embedding.weight))))
        item_text_emb = torch.sigmoid(self.shared_encoder(torch.sigmoid(self.text_encoder(self.text_embedding.weight))))

        h_s_img = self.img_shared_enc(item_image_emb)
        h_v = self.img_private_enc(item_image_emb)
        h_s_txt = self.txt_shared_enc(item_text_emb)
        h_t = self.txt_private_enc(item_text_emb)

        def split_mu_logvar(head, h):
            out = head(h)
            mu, logvar = out.chunk(2, dim=-1)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            return mu, logvar

        mu_s_img, logvar_s_img = split_mu_logvar(self.ib_heads['s_img'], h_s_img)
        mu_v, logvar_v = split_mu_logvar(self.ib_heads['p_img'], h_v)
        mu_s_txt, logvar_s_txt = split_mu_logvar(self.ib_heads['s_txt'], h_s_txt)
        mu_t, logvar_t = split_mu_logvar(self.ib_heads['p_txt'], h_t)

        z_s_img = self._gaussian_reparam(mu_s_img, logvar_s_img)
        z_v = self._gaussian_reparam(mu_v, logvar_v)
        z_s_txt = self._gaussian_reparam(mu_s_txt, logvar_s_txt)
        z_t = self._gaussian_reparam(mu_t, logvar_t)

        kl_dict = {
            'kl_s_img': self._kl_normal(mu_s_img, logvar_s_img),
            'kl_s_txt': self._kl_normal(mu_s_txt, logvar_s_txt),
            'kl_p_img': self._kl_normal(mu_v, logvar_v),
            'kl_p_txt': self._kl_normal(mu_t, logvar_t),
        }
        return z_s_img, z_v, z_s_txt, z_t, kl_dict

    def _gaussian_reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _kl_normal(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = 0.5 * (mu.pow(2) + torch.exp(logvar) - 1.0 - logvar).sum(dim=-1)
        return kl.mean()

    def decode_modalities(self, z_s, z_priv, modality="image"):
        if modality == "image":
            return self.dec_image(torch.cat([z_s, z_priv], dim=-1))
        else:
            return self.dec_text(torch.cat([z_s, z_priv], dim=-1))

    def disentangle_losses(self, z_s_img, z_v, z_s_txt, z_t):
        loss_shared_consistency = F.mse_loss(z_s_img, z_s_txt)

        def orthogonal_penalty(a, b):
            a_n = F.normalize(a, dim=-1)
            b_n = F.normalize(b, dim=-1)
            return torch.mean((a_n * b_n).sum(dim=-1).pow(2))

        loss_ortho = orthogonal_penalty(z_s_img, z_v) + orthogonal_penalty(z_s_txt, z_t)
        return loss_shared_consistency + 0.1 * loss_ortho

    # ====== Added: IB Disentanglement Statistics (For Plotting) ======

    @torch.no_grad()
    def get_ib_stats(self, sample_ratio: float = 0.1):
        """
        Sample a subset of items to calculate three cosine similarity curves in IB latent space:
          1) Shared vs Private (Image): E_g_img vs E_s_img   ≈  z_s_img vs z_v
          2) Shared vs Private (Text): E_g_txt vs E_s_txt   ≈  z_s_txt vs z_t
          3) Shared vs Shared (Cross-modal): E_g_img vs E_g_txt ≈  z_s_img vs z_s_txt

        Returns a dict with mean/std for each curve.
        """
        # If IB is off, skip
        if not getattr(self, "use_ib", False):
            return None

        # Reuse encode_disentangled_ib to get latent vectors for all items
        z_s_img, z_v, z_s_txt, z_t, _ = self.encode_disentangled_ib()
        n_items = z_s_img.size(0)
        device = z_s_img.device

        # Random sample to save time
        if sample_ratio <= 0 or sample_ratio >= 1:
            idx = torch.arange(n_items, device=device)
        else:
            sample_size = max(1, int(n_items * sample_ratio))
            perm = torch.randperm(n_items, device=device)
            idx = perm[:sample_size]

        z_s_img = z_s_img[idx]
        z_v = z_v[idx]
        z_s_txt = z_s_txt[idx]
        z_t = z_t[idx]

        def _cos_stats(a: torch.Tensor, b: torch.Tensor):
            """Calculate mean/variance for a cosine similarity curve"""
            a_n = F.normalize(a, dim=-1)
            b_n = F.normalize(b, dim=-1)
            sim = (a_n * b_n).sum(dim=-1)  # [B]
            return {
                "mean": sim.mean().item(),
                "std": sim.std(unbiased=False).item()
            }

        stats = {
            "E_g_img_vs_E_s_img": _cos_stats(z_s_img, z_v),
            "E_g_txt_vs_E_s_txt": _cos_stats(z_s_txt, z_t),
            "E_g_img_vs_E_g_txt": _cos_stats(z_s_img, z_s_txt),
        }
        return stats

    def preprocess_missing_modal(self, config):
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.missing_ratio = config['missing_ratio']
        missing_info_path = os.path.join(dataset_path, f"missing_items_{self.missing_ratio}.npy")
        self.logger.info(f"Attempting to load missing modality file from: {missing_info_path}")

        if not os.path.exists(missing_info_path):
            self.logger.warning(f"Missing modality file not found. The model will run with complete modalities.")
            self.missing_items = {'t': np.array([]), 'v': np.array([]), 'all': np.array([])}
            self.missing_items_t = np.array([])
            self.missing_items_v = np.array([])
            self.missing_modal = True
            return

        self.missing_items = np.load(missing_info_path, allow_pickle=True).item()
        self.logger.info(f"Successfully loaded missing modality info. Keys: {list(self.missing_items.keys())}")

        t_keys = self.missing_items.get('t', np.array([]))
        v_keys = self.missing_items.get('v', np.array([]))
        all_keys = self.missing_items.get('all', np.array([]))
        self.missing_items_t = np.unique(np.concatenate((all_keys, t_keys))).astype(int)
        self.missing_items_v = np.unique(np.concatenate((all_keys, v_keys))).astype(int)

        self.logger.info(f"Total unique items with missing text modality: {len(self.missing_items_t)}")
        self.logger.info(f"Total unique items with missing visual modality: {len(self.missing_items_v)}")

        self.complete_items = np.setdiff1d(np.arange(self.n_items),
                                           np.union1d(self.missing_items_v, self.missing_items_t))
        self.items_tv = np.setdiff1d(np.arange(self.n_items), np.union1d(self.missing_items_t, self.missing_items_v))

        non_missing_item_t = np.setdiff1d(self.all_items, self.missing_items_t)
        non_missing_item_v = np.setdiff1d(self.all_items, self.missing_items_v)
        self.mean_fill_old_items = bool(config["mean_fill_old_items"])
        if self.mean_fill_old_items:
            if self.v_feat is not None and len(non_missing_item_v) > 0 and len(self.missing_items_v) > 0:
                image_mean = self.v_feat[non_missing_item_v].mean(dim=0)
                fill_v = np.setdiff1d(self.missing_items_v, getattr(self, "new_items_set", np.array([], dtype=int)))
                if len(fill_v) > 0:
                    self.v_feat.data[fill_v] = image_mean
                    self.logger.info(f"Filled {len(fill_v)} missing visual features with mean value (old items only).")

            if self.t_feat is not None and len(non_missing_item_t) > 0 and len(self.missing_items_t) > 0:
                text_mean = self.t_feat[non_missing_item_t].mean(dim=0)
                fill_t = np.setdiff1d(self.missing_items_t, getattr(self, "new_items_set", np.array([], dtype=int)))
                if len(fill_t) > 0:
                    self.t_feat.data[fill_t] = text_mean
                    self.logger.info(f"Filled {len(fill_t)} missing text features with mean value (old items only).")

        else:
           self.logger.info("Skip mean fill for old items (generation ablation).")

    def init_mi_estimator(self):
        self.item_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)
        self.user_image_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)
        self.item_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)
        self.user_text_estimator = CLUBSample(self.embedding_dim, self.embedding_dim, 64).to(self.device)

        params = list(self.item_image_estimator.parameters()) + list(self.user_image_estimator.parameters()) + \
                 list(self.item_text_estimator.parameters()) + list(self.user_text_estimator.parameters())

        self.optimizer_club = torch.optim.Adam(params, lr=1e-4)

    def full_sort_predict(self, interaction):
        users, _ = interaction

        # ====== Base Collaborative Embedding ======
        user_embeddings, item_embedding = self.cge(
            self.user_embedding.weight, self.item_id_embedding.weight, self.norm_adj
        )

        maturity_scores = self._compute_item_maturity()

        # ====== Two-stage Generation Switches ======
        use_stage1 = getattr(self, "use_stage1_gen", False)
        use_stage2 = getattr(self, "use_stage2_gen", getattr(self, "use_feature_diffusion", True))

        # ====== Stage-1/2 Online Generation: Only for "New & Missing" Items (Once) ======
        if (getattr(self, "use_ib", False)
                and getattr(self, "enable_missing_modal_generation", False)
                and (use_stage1 or use_stage2)
                and (not use_stage2 or hasattr(self, "diff_shared"))
                and not getattr(self, "_stage2_infer_done", False)):

            # Only for: New Items INTERSECT Missing Modality
            new_items = getattr(self, "new_items_set", np.array([], dtype=int))
            miss_v = getattr(self, "missing_items_v", np.array([], dtype=int))
            miss_t = getattr(self, "missing_items_t", np.array([], dtype=int))

            if self.generate_only_new:
                gen_img_items = np.intersect1d(new_items, miss_v)
                gen_txt_items = np.intersect1d(new_items, miss_t)
            else:
                gen_img_items = miss_v
                gen_txt_items = miss_t

            if gen_img_items.size > 0 or gen_txt_items.size > 0:
                # 1) Encode all items in IB space
                z_s_img, z_v, z_s_txt, z_t, _ = self.encode_disentangled_ib()

                # 2) Precompute shared semantics for Stage-2 condition
                item_image_sem = torch.sigmoid(
                    self.shared_encoder(torch.sigmoid(self.image_encoder(self.image_embedding.weight)))
                )
                item_text_sem = torch.sigmoid(
                    self.shared_encoder(torch.sigmoid(self.text_encoder(self.text_embedding.weight)))
                )

                # ---------------- (a) Generate Missing [Image] ----------------
                if gen_img_items.size > 0:
                    idx_img = torch.from_numpy(gen_img_items).long().to(self.device)
                    K_img = idx_img.size(0)

                    # Prompt
                    prompt_img = self._broadcast_prompt(K_img, self.device)  # [K,64]

                    # Base Latent: Use Text side latent as condition
                    base_shared = z_s_txt[idx_img]  # [K,z_dim]
                    base_priv = z_t[idx_img]  # [K,z_dim]

                    # ---- Stage-1: Coarse Shared/Private ----
                    if use_stage1:
                        z_s_img_coarse = self.stage1_shared_gen(
                            torch.cat([base_shared, prompt_img], dim=-1)
                        )
                        z_v_coarse = self.stage1_priv_img_gen(
                            torch.cat([base_priv, prompt_img], dim=-1)
                        )
                    else:
                        # w/o Stage-1: Use base latent directly
                        z_s_img_coarse = base_shared
                        z_v_coarse = base_priv

                    # ---- Stage-2: Refine Shared (Optional) ----
                    if use_stage2:
                        sem_txt = item_text_sem[idx_img]  # [K, embedding_dim]
                        cond_shared_img = torch.cat([prompt_img, sem_txt], dim=-1)  # [K,64+emb]
                        z_s_img_refined = self.diff_shared.sample_cfg(z_s_img_coarse, cond_shared_img)
                        gen_image_feat = self.dec_image(
                            torch.cat([z_s_img_refined, z_v_coarse], dim=-1)
                        )  # [K,4096]
                    else:
                        # w/o Stage-2: Decode coarse directly
                        gen_image_feat = self.dec_image(
                            torch.cat([z_s_img_coarse, z_v_coarse], dim=-1)
                        )  # [K,4096]

                    # Momentum update back to Image Embedding (Overwrite only new & missing)
                    old_feat_img = self.image_embedding.weight.data[idx_img]
                    rho_img = maturity_scores[idx_img].unsqueeze(1)  # [K,1]
                    self.image_embedding.weight.data[idx_img] = (
                            (1 - self.momentum_eta * rho_img) * old_feat_img +
                            self.momentum_eta * rho_img * gen_image_feat
                    )

                # ---------------- (b) Generate Missing [Text] ----------------
                if gen_txt_items.size > 0:
                    idx_txt = torch.from_numpy(gen_txt_items).long().to(self.device)
                    K_txt = idx_txt.size(0)

                    prompt_txt = self._broadcast_prompt(K_txt, self.device)  # [K,64]

                    # Base Latent: Use Image side latent as condition
                    base_shared_txt = z_s_img[idx_txt]  # [K,z_dim]
                    base_priv_txt = z_v[idx_txt]  # [K,z_dim]

                    # ---- Stage-1: Coarse Shared/Private ----
                    if use_stage1:
                        z_s_txt_coarse = self.stage1_shared_gen(
                            torch.cat([base_shared_txt, prompt_txt], dim=-1)
                        )
                        z_t_coarse = self.stage1_priv_txt_gen(
                            torch.cat([base_priv_txt, prompt_txt], dim=-1)
                        )
                    else:
                        z_s_txt_coarse = base_shared_txt
                        z_t_coarse = base_priv_txt

                    # ---- Stage-2: Refine Shared (Optional) ----
                    if use_stage2:
                        sem_img = item_image_sem[idx_txt]  # [K, embedding_dim]
                        cond_shared_txt = torch.cat([prompt_txt, sem_img], dim=-1)
                        z_s_txt_refined = self.diff_shared.sample_cfg(z_s_txt_coarse, cond_shared_txt)
                        gen_text_feat = self.dec_text(
                            torch.cat([z_s_txt_refined, z_t_coarse], dim=-1)
                        )  # [K,384]
                    else:
                        gen_text_feat = self.dec_text(
                            torch.cat([z_s_txt_coarse, z_t_coarse], dim=-1)
                        )  # [K,384]

                    # Momentum update back to Text Embedding
                    old_feat_txt = self.text_embedding.weight.data[idx_txt]
                    rho_txt = maturity_scores[idx_txt].unsqueeze(1)
                    self.text_embedding.weight.data[idx_txt] = (
                            (1 - self.momentum_eta * rho_txt) * old_feat_txt +
                            self.momentum_eta * rho_txt * gen_text_feat
                    )

            # Flag: Stage-2 inference done for this round
            self._stage2_infer_done = True

        # ====== Continue Inference with (possibly updated) features ======
        item_image, item_text = self.mge()

        item_image_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.image_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]
        item_text_filter = torch.sparse.mm(
            self.adj.t(), F.tanh(self.text_preference_(self.user_embedding.weight))
        ) * self.num_inters[self.n_users:]

        # Use improved graph propagation
        for _ in range(min(self.n_mm_layers, 2)):
            item_image = self._progressive_graph_propagation(item_image, self.image_adj, "image")
            item_text = self._progressive_graph_propagation(item_text, self.text_adj, "text")

        item_image = torch.mul(item_image, item_image_filter)
        item_text = torch.mul(item_text, item_text_filter)

        user_image = torch.sparse.mm(self.adj, item_image) * self.num_inters[:self.n_users]
        user_text = torch.sparse.mm(self.adj, item_text) * self.num_inters[:self.n_users]

        user_long_all = user_embeddings + ((user_image + user_text) / 2) / 2
        item_emb_final = item_embedding + ((item_image + item_text) / 2) / 2

        user_short_batch = self._encode_short_term(item_emb_final, users)
        user_long_batch = user_long_all[users]

        if self.use_fixed_short_weight:
            gate_alpha = torch.full(
                (user_short_batch.size(0), 1),
                float(self.short_term_weight),
                device=self.device
            )
        else:
            gate_alpha = torch.sigmoid(self.gate_mlp(torch.cat([user_long_batch, user_short_batch], dim=1)))
        user_emb_batch = gate_alpha * user_short_batch + (1.0 - gate_alpha) * user_long_batch

        # === User Modal Weights (Consistent with Training) ===
        w = self._get_user_modal_weights(user_emb_batch)  # [B,3]

        # Per-modal Scoring
        score_id = user_emb_batch @ item_emb_final.T
        score_img = user_emb_batch @ item_image.T
        score_txt = user_emb_batch @ item_text.T

        # Weighted Sum
        scores = (
                w[:, 0:1] * score_id +
                w[:, 1:2] * score_img +
                w[:, 2:3] * score_txt
        )  # [B,N]

        return scores

    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)

        if not self.enable_dynamic_weighting:
            print("Constructing Adjacency Matrix: Using Static Weights (Value=1).")
            inter_M = self.interaction_matrix
            inter_M_t = self.interaction_matrix.transpose()
            data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
            data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
            for key, value in data_dict.items():
                A[key] = value
        else:
            print(f"Constructing Adjacency Matrix: Using Dynamic Weights (Decay Factor lambda={self.time_decay_lambda}).")
            for user_id, item_id, timestamp, rating in self.interaction_history:
                time_diff = self.latest_timestamp - timestamp
                recency_weight = np.exp(-self.time_decay_lambda * time_diff)
                intensity_weight = rating
                final_weight = recency_weight * intensity_weight

                A[user_id, item_id + self.n_users] = final_weight
                A[item_id + self.n_users, user_id] = final_weight

        sumArr = A.sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag[diag == 1e-7] = 0
        diag = np.power(diag, -0.5)
        diag[np.isinf(diag)] = 0
        D = sp.diags(diag)

        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return sumArr, torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def calculate_reg_loss(self, user_emb, pos_items_emb, neg_item_emb, image_emb, text_emb):
        loss_reg = self.reg_loss(user_emb, pos_items_emb, neg_item_emb) * 1e-5
        loss_reg += self.reg_loss(image_emb) * 0.1
        loss_reg += self.reg_loss(text_emb) * 0.1
        return loss_reg

    def calculate_recon_loss(self, image, text):
        item_image_recon = self.image_decoder(self.perturb(image.detach()))
        item_text_recon = self.text_decoder(self.perturb(text.detach()))

        loss = 0
        loss += F.mse_loss(item_image_recon, self.image_embedding.weight) * 0.1
        loss += F.mse_loss(item_text_recon, self.text_embedding.weight) * 0.1
        return loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def bpr_loss(self, users, pos_items, neg_items):
        if len(pos_items.shape) == 2:
            pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
            neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        else:
            pos_scores = torch.einsum("ik, ijk -> ij", users, pos_items)
            neg_scores = torch.einsum("ik, ijk -> ij", users, neg_items)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def InfoNCE(self, view1, view2, temperature=0.4):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def forward(self):
        pass

    def perturb(self, x):
        noise = torch.rand_like(x).to(self.device)
        x = x + torch.sign(x) * F.normalize(noise, dim=-1) * 0.1
        return x
