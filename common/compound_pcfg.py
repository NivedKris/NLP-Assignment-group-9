"""Compound PCFG (Kim, Dyer & Rush, ACL 2019) - a device-agnostic PyTorch port of
harvardnlp/compound-pcfg (models.py + PCFG.py) with a vectorised inside algorithm.

Architecture (assignment Task 2 settings):
  nonterminals NT=30, preterminals T=60, z in R^64, symbol embeddings 256,
  inference network: word embeddings 256 -> 1-layer BiLSTM (hidden 256) -> max-pool -> (mu, logvar)
  root rules  S -> A  : MLP on [root_emb; z]: Linear(320,256) -> 2 residual ReLU layers (256) -> Linear
  emissions   T -> w  : MLP on [t_emb; z]   : Linear(320,256) -> 2 residual ReLU layers (256) -> Linear
  binary rules A -> B C on [nt_emb; z]:
      rule_layers=1 : one Linear layer (exactly the official models.py)          <- main run
      rule_layers=2 : Linear(320,256) -> ReLU -> Linear (literal reading of the assignment's
                      "2-layer MLP, hidden 256, ReLU" applied to the binary rules) <- ablation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e9


class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100, out_dim=100):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class CompPCFG(nn.Module):
    def __init__(self, vocab, h_dim=256, w_dim=256, z_dim=64, state_dim=256,
                 t_states=60, nt_states=30, rule_hidden=256, rule_layers=1):
        super().__init__()
        self.state_dim, self.z_dim = state_dim, z_dim
        self.nt_states, self.t_states = nt_states, t_states
        self.all_states = nt_states + t_states
        self.t_emb = nn.Parameter(torch.randn(t_states, state_dim))
        self.nt_emb = nn.Parameter(torch.randn(nt_states, state_dim))
        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        if rule_layers == 2:   # literal assignment reading: 2-layer MLP, hidden 256, ReLU
            self.rule_mlp = nn.Sequential(nn.Linear(state_dim + z_dim, rule_hidden), nn.ReLU(),
                                          nn.Linear(rule_hidden, self.all_states ** 2))
        else:                  # official code: a single linear layer
            self.rule_mlp = nn.Linear(state_dim + z_dim, self.all_states ** 2)
        self.root_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, nt_states))
        self.enc_emb = nn.Embedding(vocab, w_dim)
        self.enc_rnn = nn.LSTM(w_dim, h_dim, bidirectional=True, num_layers=1, batch_first=True)
        self.enc_params = nn.Linear(h_dim * 2, z_dim * 2)
        self.vocab_mlp = nn.Sequential(nn.Linear(z_dim + state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, vocab))

    # ---- inference network q(z | x)
    def enc(self, x):
        h, _ = self.enc_rnn(self.enc_emb(x))
        params = self.enc_params(h.max(1)[0])
        return params[:, :self.z_dim], params[:, self.z_dim:]

    @staticmethod
    def kl(mean, logvar):
        return (-0.5 * (logvar - mean.pow(2) - logvar.exp() + 1)).sum(1)

    # ---- grammar given z
    def grammar(self, x, z):
        b, n = x.shape
        root = F.log_softmax(self.root_mlp(torch.cat([self.root_emb.expand(b, -1), z], 1)), 1)
        nt_in = torch.cat([self.nt_emb.unsqueeze(0).expand(b, -1, -1),
                           z.unsqueeze(1).expand(b, self.nt_states, self.z_dim)], 2)
        rule = F.log_softmax(self.rule_mlp(nt_in), 2).view(
            b, self.nt_states, self.all_states, self.all_states)
        t_in = torch.cat([self.t_emb.unsqueeze(0).expand(b, -1, -1),
                          z.unsqueeze(1).expand(b, self.t_states, self.z_dim)], 2)
        emit = F.log_softmax(self.vocab_mlp(t_in), 2)                  # b x T x V
        unary = torch.gather(emit, 2, x.unsqueeze(1).expand(b, self.t_states, n)).transpose(1, 2)
        return unary, rule, root                                       # unary: b x n x T

    def forward(self, x, argmax=False, use_mean=False):
        mean, logvar = self.enc(x)
        kl = self.kl(mean, logvar)
        z = mean if use_mean else mean + (0.5 * logvar).exp() * torch.randn_like(mean)
        unary, rule, root = self.grammar(x, z)
        log_z = inside(unary, rule, root, self.nt_states)
        if argmax:
            with torch.no_grad():
                spans, tags = viterbi(unary, rule, root, self.nt_states)
            return -log_z, kl, spans, tags
        return -log_z, kl


def _blocks(rule, NT):
    """Split b x NT x S x S rule scores by child type. A width-1 child is always a
    preterminal (states NT..S-1), a wider child always a nonterminal (states 0..NT-1)."""
    return {("T", "T"): rule[:, :, NT:, NT:], ("T", "N"): rule[:, :, NT:, :NT],
            ("N", "T"): rule[:, :, :NT, NT:], ("N", "N"): rule[:, :, :NT, :NT]}


def _children(chart, unary, W, N):
    """For spans of width W (N of them): list of (k-list, left b N K X, right b N K Y, key)."""
    def side(width, start):
        return unary[:, start:start + N] if width == 1 else chart[width][:, start:start + N]
    groups = {}
    for k in range(1, W):
        key = ("T" if k == 1 else "N", "T" if W - k == 1 else "N")
        groups.setdefault(key, []).append(k)
    out = []
    for key, ks in groups.items():
        left = torch.stack([side(k, 0) for k in ks], 2)
        right = torch.stack([side(W - k, k) for k in ks], 2)
        out.append((ks, left, right, key))
    return out


def inside(unary, rule, root, NT):
    """log p(x) by the inside algorithm, vectorised over all spans of equal width.
    unary: b x n x T (log), rule: b x NT x S x S (log), root: b x NT (log). Returns b.
    Sums are done in probability space with max-shifts (log-sum-exp trick)."""
    b, n, T = unary.shape
    # per-parent shift so that the largest rule probability of every A is 1
    m_rule = rule.reshape(b, NT, -1).max(-1)[0]                         # b NT
    rp_all = (rule - m_rule[:, :, None, None]).exp()
    blocks = {key: blk.permute(0, 2, 1, 3).reshape(b, blk.size(2), NT * blk.size(3))
              for key, blk in _blocks(rp_all, NT).items()}               # [b, B, (A, C)]
    chart = {}                                                          # width -> b x (n-W+1) x NT
    for W in range(2, n + 1):
        N = n - W + 1
        parts = []
        for ks, left, right, key in _children(chart, unary, W, N):
            K, X, Y = len(ks), left.size(-1), right.size(-1)
            ml, mr = left.max(-1, keepdim=True)[0], right.max(-1, keepdim=True)[0]
            lp, rp = (left - ml).exp(), (right - mr).exp()
            tmp = torch.bmm(lp.reshape(b, N * K, X), blocks[key]).view(b, N, K, NT, Y)
            v = (tmp * rp.unsqueeze(3)).sum(-1)                                   # b N K NT
            parts.append(v.clamp_min(1e-38).log() + ml + mr)
        chart[W] = torch.logsumexp(torch.cat(parts, 2), 2) + m_rule[:, None, :]  # b N NT
    return torch.logsumexp(chart[n][:, 0] + root, 1)


@torch.no_grad()
def viterbi(unary, rule, root, NT):
    """Max-scoring binary tree. Returns (spans, tags): spans[b] = list of (i, j, label)
    inclusive, post-order with the root span last; tags[b] = list of preterminal ids."""
    b, n, T = unary.shape
    S = NT + T
    blocks = _blocks(rule, NT)
    offset = {"T": NT, "N": 0}
    chart, bp = {}, {}
    for W in range(2, n + 1):
        N = n - W + 1
        best = unary.new_full((b, N, NT), -float("inf"))
        best_k = torch.zeros(b, N, NT, dtype=torch.long, device=unary.device)
        best_b = torch.zeros_like(best_k)
        best_c = torch.zeros_like(best_k)
        for ks, left, right, key in _children(chart, unary, W, N):
            blk = blocks[key]
            X, Y = blk.size(2), blk.size(3)
            for j, k in enumerate(ks):
                cand = left[:, :, j, None, :, None] + right[:, :, j, None, None, :] + blk[:, None]
                m, idx = cand.reshape(b, N, NT, X * Y).max(-1)                      # b N NT
                upd = m > best
                best = torch.where(upd, m, best)
                best_k = torch.where(upd, torch.full_like(best_k, k), best_k)
                best_b = torch.where(upd, idx // Y + offset[key[0]], best_b)
                best_c = torch.where(upd, idx % Y + offset[key[1]], best_c)
        chart[W] = best
        bp[W] = (best_k.cpu().numpy(), best_b.cpu().numpy(), best_c.cpu().numpy())
    tags_all = unary.argmax(-1).cpu().numpy()  # only used when n == 1
    roots = (chart[n][:, 0] + root).argmax(1).cpu().numpy() if n > 1 else None
    out_spans, out_tags = [], []
    for bb in range(b):
        spans, tags = [], [None] * n

        def rec(i, W, a):
            if W == 1:
                tags[i] = a - NT
                return
            k = int(bp[W][0][bb, i, a])
            rec(i, k, int(bp[W][1][bb, i, a]))
            rec(i + k, W - k, int(bp[W][2][bb, i, a]))
            spans.append((i, i + W - 1, a))

        if n == 1:
            tags[0] = int(tags_all[bb, 0])
        else:
            rec(0, n, int(roots[bb]))
        out_spans.append(spans)
        out_tags.append(tags)
    return out_spans, out_tags
