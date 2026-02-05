from fin_ai.model.configuration_next import FinAINextConfig
from fin_ai.model.modeling_next import FinAINextForCausalLM


def test_initialization():
    config = FinAINextConfig(
        vocab_size=1000, hidden_size=128, num_layers=2, tie_word_embeddings=True
    )
    print("Initializing model...")
    try:
        model = FinAINextForCausalLM(config)
        print("Model initialized successfully!")

        # Check if weights are tied
        input_embeds = model.get_input_embeddings().weight
        output_embeds = model.get_output_embeddings().weight

        if input_embeds is output_embeds:
            print("Weights are tied correctly.")
        else:
            print(
                "Weights are NOT tied (this might be expected if post_init doesn't tie them automatically but tie_weights does)."
            )
            # In transformers, post_init calls tie_weights if tie_word_embeddings is True
            if input_embeds is output_embeds:
                print("Wait, they ARE tied.")
            else:
                model.tie_weights()
                if (
                    model.get_input_embeddings().weight
                    is model.get_output_embeddings().weight
                ):
                    print("Weights tied after explicit tie_weights().")
                else:
                    print("Weights still not tied after explicit tie_weights().")

    except Exception as e:
        print(f"Initialization failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_initialization()
