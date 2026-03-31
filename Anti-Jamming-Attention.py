import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. The Core Artifact: Clockfield Attention
# ==========================================
class TopologicalClockfieldAttention(nn.Module):
    def __init__(self, embed_dim, tau=50.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau = tau 
        self.scale = 1.0 / math.sqrt(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)

    def forward(self, x, return_gamma=False):
        Q = self.q_proj(x) * self.scale
        K = self.k_proj(x)
        V = self.v_proj(x)

        # S matrix: Compares every time-step against every other time-step
        S = torch.matmul(Q, K.conj().transpose(-2, -1))
        attn_base = F.softmax(S.real, dim=-1)

        if self.tau > 0:
            beta = (S.imag) ** 2
            gamma = 1.0 / (1.0 + self.tau * beta) ** 2
            attn_weights = attn_base * gamma
        else:
            gamma = torch.ones_like(attn_base)
            attn_weights = attn_base

        out = torch.matmul(attn_weights.to(torch.cfloat), V)
        if return_gamma: return out, gamma
        return out

# ==========================================
# 2. Sequence Classifier (Mean Pooling)
# ==========================================
class RadarSequenceClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, tau):
        super().__init__()
        self.attn = TopologicalClockfieldAttention(embed_dim, tau=tau)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x, return_gamma=False):
        attn_out, gamma = self.attn(x, return_gamma=True)
        
        # Pool the sequence together. 
        # (Jammed tokens in the Clockfield should output 0 and not corrupt the mean)
        pooled = attn_out.mean(dim=1) 
        
        x_cat = torch.cat([pooled.real, pooled.imag], dim=-1)
        logits = self.classifier(x_cat)
        
        if return_gamma: return logits, gamma
        return logits

# ==========================================
# 3. Targeted EW Jamming Simulator (V3 - LOUD JAMMER)
# ==========================================
def generate_quarantine_data(n_samples, seq_len, embed_dim, n_classes, noise_std, jamming=False):
    """Generates a sequence where the second half is hit with a HIGH-POWER jammer."""
    class_phases = torch.linspace(0, 2*np.pi, n_classes+1)[:-1]
    X = torch.zeros(n_samples, seq_len, embed_dim, dtype=torch.cfloat)
    Y = torch.randint(0, n_classes, (n_samples,))

    for i in range(n_samples):
        cls = Y[i]
        phase = class_phases[cls] + torch.randn(seq_len, embed_dim) * noise_std
        amp = 0.8 + 0.4 * torch.rand(seq_len, embed_dim) # Normal quiet signal
        
        # TARGETED HIGH-POWER JAMMING
        if jamming:
            jam_start = seq_len // 2
            # 1. Scramble the phase
            phase[jam_start:, :] += torch.randn(seq_len - jam_start, embed_dim) * 5.0
            # 2. Blast the power (Amplitude x10) to blind the receiver
            amp[jam_start:, :] *= 10.0 
            
        X[i] = amp * torch.exp(1j * phase)
    return X, Y

# ==========================================
# 4. Training Engine
# ==========================================
def train_model(model, X, Y, epochs=75):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None and p.grad.is_conj():
                p.grad = p.grad.resolve_conj()
        optimizer.step()

def evaluate(model, X, Y):
    model.eval()
    with torch.no_grad():
        logits, gamma = model(X, return_gamma=True)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == Y).float().mean().item()
        return acc, gamma.flatten().cpu().numpy()

# ==========================================
# 5. The Quarantine Experiment
# ==========================================
if __name__ == "__main__":
    print("=======================================================")
    print("⚡ CLOCKFIELD V2: TOPOLOGICAL QUARANTINE TEST ⚡")
    print("=======================================================\n")

    SEQ_LEN = 16
    EMBED_DIM = 8
    NUM_CLASSES = 4
    TRAIN_NOISE = 0.2

    print("1. Generating Clean Training Data...")
    X_train, Y_train = generate_quarantine_data(800, SEQ_LEN, EMBED_DIM, NUM_CLASSES, TRAIN_NOISE, jamming=False)

    model_standard = RadarSequenceClassifier(EMBED_DIM, NUM_CLASSES, tau=0.0)
    model_clockfield = RadarSequenceClassifier(EMBED_DIM, NUM_CLASSES, tau=50.0)

    print("2. Training Standard Complex Transformer...")
    train_model(model_standard, X_train, Y_train)

    print("3. Training Clockfield Transformer (τ=50.0)...")
    train_model(model_clockfield, X_train, Y_train)

    print("\n4. Deploying (50% of the sequence is hit with Extreme Jamming)...")
    X_test, Y_test = generate_quarantine_data(400, SEQ_LEN, EMBED_DIM, NUM_CLASSES, TRAIN_NOISE, jamming=True)

    acc_std, _ = evaluate(model_standard, X_test, Y_test)
    acc_clk, gamma_clk = evaluate(model_clockfield, X_test, Y_test)

    print("\n================ RESULTS ================")
    print(f"Standard Attention Accuracy:   {acc_std * 100:.1f}%")
    print(f"Clockfield Attention Accuracy: {acc_clk * 100:.1f}%")
    print("=========================================\n")