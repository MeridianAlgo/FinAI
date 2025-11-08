"""Neural language model for next-word prediction"""
import numpy as np
import pickle
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier


class LanguageModel:
    """Simple neural language model for next-word prediction"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        """Train the language model"""
        print(f"Training language model on {len(X)} sequences...")
        
        # Use MLPClassifier for next-word prediction
        self.model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_dim, self.hidden_dim),
            activation='relu',
            max_iter=epochs,
            random_state=42,
            verbose=True,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        print("✓ Language model training complete")
    
    def train_incremental(self, batches_iter, epochs: int = 1, classes: np.ndarray = None):
        """Incremental (streaming) training using SGDClassifier with log-loss.
        batches_iter should yield (X, y) numpy arrays.
        """
        if classes is None:
            raise ValueError("classes array (0..vocab_size-1) must be provided for incremental training")
        
        import time
        from datetime import timedelta
        
        print("Training language model incrementally (streaming)...")
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            learning_rate="optimal",
            random_state=42,
        )
        first = True
        total = 0
        start_time = time.time()
        epoch_start_time = start_time
        
        for ep in range(epochs):
            batch_count = 0
            epoch_total = 0
            
            for X, y in batches_iter:
                if first:
                    self.model.partial_fit(X, y, classes=classes)
                    first = False
                else:
                    self.model.partial_fit(X, y)
                
                total += len(X)
                epoch_total += len(X)
                batch_count += 1
                
                # Print progress every 50 batches
                if batch_count % 50 == 0:
                    elapsed = time.time() - epoch_start_time
                    batches_per_sec = batch_count / elapsed if elapsed > 0 else 0
                    sequences_per_sec = epoch_total / elapsed if elapsed > 0 else 0
                    
                    print(f"  Epoch {ep+1}/{epochs}: {batch_count} batches, {epoch_total:,} sequences | "
                          f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                          f"Speed: {sequences_per_sec:.0f} seq/s")
            
            # Epoch complete
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            
            if epochs > 1:
                avg_time_per_epoch = total_time / (ep + 1)
                remaining_epochs = epochs - (ep + 1)
                eta = avg_time_per_epoch * remaining_epochs if remaining_epochs > 0 else 0
                
                print(f"  [OK] Epoch {ep+1}/{epochs} complete | "
                      f"Time: {timedelta(seconds=int(epoch_time))} | "
                      f"Total: {timedelta(seconds=int(total_time))} | "
                      f"ETA: {timedelta(seconds=int(eta))}")
            else:
                print(f"  [OK] Epoch {ep+1}/{epochs} complete | "
                      f"Time: {timedelta(seconds=int(epoch_time))}")
            
            epoch_start_time = time.time()
        
        total_time = time.time() - start_time
        self.is_trained = True
        print(f"[OK] Incremental training complete on ~{total:,} sequences | "
              f"Total time: {timedelta(seconds=int(total_time))}")
    
    def predict_next(self, sequence: List[int], temperature: float = 1.0, top_k: int = 50) -> int:
        """Predict next token given a sequence"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get probabilities for all tokens
        sequence_array = np.array([sequence])
        probas = self.model.predict_proba(sequence_array)[0]
        
        # Apply temperature
        if temperature != 1.0:
            probas = np.power(probas, 1.0 / temperature)
            probas = probas / np.sum(probas)
        
        # Top-k sampling
        if top_k > 0:
            top_k_indices = np.argsort(probas)[-top_k:]
            top_k_probas = probas[top_k_indices]
            top_k_probas = top_k_probas / np.sum(top_k_probas)
            next_token = np.random.choice(top_k_indices, p=top_k_probas)
        else:
            next_token = np.random.choice(len(probas), p=probas)
        
        return int(next_token)
    
    def save(self, path: str):
        """Save model to disk"""
        if self.is_trained:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'LanguageModel':
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)
