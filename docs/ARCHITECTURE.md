# FinAI Architecture

## System Overview

FinAI is a modular neural language model that generates financial advice through **pure next-word prediction**. Unlike the previous version with preloaded responses, this system learns from your training data and generates text word-by-word.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User Input                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Main Application                         │
│                   (src/core/finai.py)                       │
│  • Orchestrates all components                              │
│  • Manages chat loop                                        │
│  • Handles training workflow                                │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐    ┌──────────────────────────────┐
│  Conversation Context  │    │      Text Generator          │
│  (src/core/context.py) │    │ (src/models/text_generator)  │
│  • Stores history      │    │  • Generates responses       │
│  • Provides context    │    │  • Applies sampling          │
└────────────────────────┘    └──────────┬───────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────────┐
                              │     Language Model           │
                              │ (src/models/language_model)  │
                              │  • Neural network            │
                              │  • Predicts next word        │
                              │  • Trained on your data      │
                              └──────────┬───────────────────┘
                                         │
                                         ▼
                              ┌──────────────────────────────┐
                              │       Tokenizer              │
                              │   (src/data/tokenizer.py)    │
                              │  • Text ↔ Numbers            │
                              │  • Vocabulary management     │
                              └──────────────────────────────┘
```

## Data Flow

### Training Phase

```
Training Data (text file)
    │
    ▼
┌─────────────────────┐
│  Dataset Loader     │  Load and parse text
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Tokenizer        │  Build vocabulary, convert text → tokens
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Sequence Preparation│  Create input-output pairs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Language Model     │  Train neural network
└─────────┬───────────┘
          │
          ▼
    Save to disk
```

### Generation Phase

```
User Input: "how can i save money"
    │
    ▼
┌─────────────────────┐
│  Add Context        │  Include conversation history
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Tokenize          │  Convert to token IDs
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Predict Next Word  │  Use language model
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Apply Sampling     │  Temperature, top-k
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Repeat N Times     │  Generate full response
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Decode Tokens     │  Convert back to text
└─────────┬───────────┘
          │
          ▼
    Display Response
```

## Module Breakdown

### 1. Configuration (`src/config.py`)

**Purpose**: Central configuration for all parameters

**Key Settings**:
- `VOCAB_SIZE`: Maximum vocabulary size (10,000)
- `MAX_SEQUENCE_LENGTH`: Context window (50 tokens)
- `TEMPERATURE`: Generation randomness (0.7)
- `TOP_K`: Sampling diversity (50)
- `HIDDEN_DIM`: Neural network size (256)

### 2. Data Processing (`src/data/`)

#### Tokenizer (`tokenizer.py`)

**Purpose**: Convert between text and numerical tokens

**Key Methods**:
- `fit(texts)`: Build vocabulary from training data
- `encode(text)`: Text → token IDs
- `decode(indices)`: Token IDs → text
- `save()/load()`: Persistence

**Special Tokens**:
- `<PAD>`: Padding for sequences
- `<UNK>`: Unknown words
- `<START>`: Sequence start
- `<END>`: Sequence end

#### Dataset Loader (`dataset_loader.py`)

**Purpose**: Load and prepare training data

**Key Methods**:
- `load_from_file()`: Read text file
- `prepare_sequences()`: Create input-output pairs

**Process**:
1. Load text lines
2. Tokenize each line
3. Create sequences: [w1, w2, w3] → predict w4
4. Pad sequences to fixed length

### 3. Models (`src/models/`)

#### Language Model (`language_model.py`)

**Purpose**: Neural network for next-word prediction

**Architecture**:
- Input: Sequence of token IDs
- Hidden layers: 2 layers of 256 neurons
- Output: Probability distribution over vocabulary

**Key Methods**:
- `train()`: Train on sequences
- `predict_next()`: Predict next token
- `save()/load()`: Persistence

**Prediction Process**:
1. Take input sequence
2. Forward pass through network
3. Get probability for each word
4. Apply temperature scaling
5. Sample from top-k candidates

#### Text Generator (`text_generator.py`)

**Purpose**: Generate text using language model

**Key Methods**:
- `generate()`: Generate text from prompt

**Generation Process**:
1. Encode prompt to tokens
2. For each position:
   - Predict next token
   - Add to sequence
   - Update context window
3. Decode tokens to text

**Sampling Strategies**:
- **Temperature**: Controls randomness
  - Low (0.1-0.5): Conservative, predictable
  - Medium (0.6-0.9): Balanced
  - High (1.0-2.0): Creative, diverse
- **Top-K**: Sample from K most likely words
  - Prevents unlikely words
  - Maintains quality

### 4. Core Application (`src/core/`)

#### Context Manager (`context.py`)

**Purpose**: Manage conversation history

**Key Methods**:
- `add_message()`: Add user/assistant message
- `get_context_string()`: Get recent history as text
- `clear()`: Reset conversation

**Features**:
- Stores last N messages
- Provides context for generation
- Timestamps each message

#### Main Application (`finai.py`)

**Purpose**: Orchestrate all components

**Key Methods**:
- `initialize()`: Load models, setup
- `train_from_file()`: Train on dataset
- `generate_response()`: Create response
- `chat()`: Main interaction loop

**Workflow**:
1. Load or train models
2. Enter chat loop
3. For each user input:
   - Add to context
   - Generate response
   - Display to user

## Key Design Principles

### 1. Modularity

Each component has a single responsibility:
- Tokenizer: Text ↔ Numbers
- Model: Prediction
- Generator: Text generation
- Context: History management

### 2. No Preloaded Responses

Unlike the old system:
- ❌ No template responses
- ❌ No hardcoded answers
- ✅ Pure next-word prediction
- ✅ Learns from your data

### 3. Trainable

You provide the training data:
- Financial advice examples
- Conversational format
- Your domain knowledge

### 4. Configurable

Easy to adjust:
- Model size
- Generation parameters
- Vocabulary size
- Context window

## Comparison: Old vs New

### Old Architecture (finai_old.py)

```
User Input
    │
    ▼
Intent Classifier (ML)
    │
    ▼
Route to Handler (18 handlers)
    │
    ▼
Template Response + Rules
    │
    ▼
Fill in Variables
    │
    ▼
Return Preloaded Text
```

**Problems**:
- All responses hardcoded
- Can't learn new patterns
- Inflexible
- Not true language generation

### New Architecture

```
User Input
    │
    ▼
Add Context
    │
    ▼
Tokenize
    │
    ▼
Predict Next Word (repeat)
    │
    ▼
Generate Full Response
    │
    ▼
Return Dynamic Text
```

**Benefits**:
- Learns from data
- Generates novel responses
- Flexible and extensible
- True language modeling

## Performance Considerations

### Memory Usage

- **Vocabulary**: ~10K words × 4 bytes = 40KB
- **Model**: ~256×256×2 weights = ~500KB
- **Context**: Last 10 messages = ~10KB
- **Total**: ~1MB (very lightweight)

### Speed

- **Training**: 1-5 minutes for 1000 examples
- **Generation**: ~100ms per response
- **Loading**: <1 second

### Scalability

To improve quality:
1. Add more training data (500-5000 examples)
2. Increase HIDDEN_DIM (256 → 512)
3. Increase VOCAB_SIZE (10K → 20K)
4. Increase MAX_SEQUENCE_LENGTH (50 → 100)

## Future Enhancements

Possible improvements:
1. **Better Model**: Use LSTM/Transformer instead of MLP
2. **Beam Search**: Better generation quality
3. **Fine-tuning**: Update model with user feedback
4. **Multi-turn Context**: Better conversation tracking
5. **Attention Mechanism**: Focus on relevant context
6. **Embeddings**: Pre-trained word vectors

## Summary

The new FinAI is a **true neural language model** that:
- ✅ Generates text word-by-word
- ✅ Learns from your training data
- ✅ No preloaded responses
- ✅ Modular and maintainable
- ✅ Configurable and extensible

It's a complete rewrite focused on **real language modeling** rather than template-based responses.
