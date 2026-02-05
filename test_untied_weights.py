"""Test model with untied weights"""

import torch

from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM


def test_untied_weights():
    """Test that the model works with tie_word_embeddings=False"""
    print("Testing model with untied weights...")

    config = FinAINextConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,  # Disable weight tying
    )

    try:
        model = FinAINextForCausalLM(config)
        print("✓ Model initialized successfully!")

        tied_keys = model._tied_weights_keys()
        print(f"✓ _tied_weights_keys returns: {tied_keys}")
        assert tied_keys == {}, "Should return empty dict when not tying weights"
        print("✓ Correctly returns empty dict for untied weights")

        # Test forward pass
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model(input_ids)
        print(f"✓ Forward pass successful! Output shape: {output.logits.shape}")

        print("\n✅ Untied weights test passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_untied_weights()
    exit(0 if success else 1)
