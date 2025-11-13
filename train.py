import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from src.common import load_data, prepare_train_data, generate_submission


# Configuration
MODEL_PATH = "models/mlp_baseline.pth"
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=2048):
        super().__init__()
        # Encoder: project to higher dimensional space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Middle layers: learn complex transformations
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Decoder: project to output space
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Encode to hidden space
        h = self.encoder(x)
        
        # Process with residual connection
        h = h + self.middle(h)
        
        # Decode to output space
        out = self.decoder(h)
        
        return out


def train_model(model, train_loader, val_loader, device, epochs, lr):
    """Train the MLP model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)

            loss = F.mse_loss(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = F.mse_loss(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")

    return model


def main():
    print("1. Loading data...")
    # Load data
    train_data = load_data("data/train/train.npz")
    X, y, label = prepare_train_data(train_data)
    DATASET_SIZE = len(X)
    print(f"   Dataset size: {DATASET_SIZE:,}")
    
    # Split train/val
    print("2. Splitting train/validation...")
    n_train = int(0.9 * len(X))
    TRAIN_SPLIT = torch.zeros(len(X), dtype=torch.bool)
    TRAIN_SPLIT[:n_train] = 1
    X_train, X_val = X[TRAIN_SPLIT], X[~TRAIN_SPLIT]
    y_train, y_val = y[TRAIN_SPLIT], y[~TRAIN_SPLIT]

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    print(f"   Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")

    # Initialize model
    print("3. Initializing model...")
    model = MLP().to(DEVICE)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {DEVICE}")

    # Train
    print("\n4. Training...")
    model = train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR)

    # Load best model for evaluation
    print("\n5. Loading best model...")
    model.load_state_dict(torch.load(MODEL_PATH))

    # Generate submission
    print("\n6. Generating submission...")
    test_data = load_data("data/test/test.clean.npz")

    test_embds = test_data['captions/embeddings']
    test_embds = torch.from_numpy(test_embds).float()

    with torch.no_grad():
        pred_embds = model(test_embds.to(DEVICE)).cpu()

    submission = generate_submission(test_data['captions/ids'], pred_embds, 'submission.csv')
    print(f"\n✓ Model saved to: {MODEL_PATH}")
    print(f"✓ Submission saved to: submission.csv")


if __name__ == "__main__":
    main()

