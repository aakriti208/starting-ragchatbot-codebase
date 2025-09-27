# Local LLM Setup Guide

This guide explains how to configure the RAG chatbot to use local LLM models instead of the Anthropic API.

## Supported Local LLM Providers

### 1. Ollama (Recommended)

**Installation:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (examples)
ollama pull llama3.2          # 3B parameters - fast, decent quality
ollama pull llama3.2:3b       # Same as above
ollama pull llama3.1:8b       # 8B parameters - better quality
ollama pull mistral           # Alternative model
ollama pull codellama         # Good for code-related tasks

# Start Ollama server (runs on http://localhost:11434)
ollama serve
```

**Configuration (.env file):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

### 2. LocalAI

**Installation with Docker:**
```bash
# Basic setup
docker run -p 8080:8080 --name local-ai -ti localai/localai:latest

# With GPU support (requires NVIDIA Docker)
docker run -p 8080:8080 --gpus all --name local-ai -ti localai/localai:latest-gpu
```

**Configuration (.env file):**
```env
LLM_PROVIDER=localai
LOCALAI_BASE_URL=http://localhost:8080
LOCALAI_MODEL=gpt-3.5-turbo
```

### 3. LM Studio (GUI Alternative)

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a model through the GUI
3. Start the local server (usually on port 1234)
4. Use LocalAI provider settings with appropriate URL

## Setup Steps

1. **Install dependencies:**
   ```bash
   uv add openai  # Required for Ollama and LocalAI providers
   ```

2. **Configure your .env file:**
   ```bash
   cp .env.example .env
   # Edit .env with your preferred provider settings
   ```

3. **Start your local LLM server** (see provider-specific instructions above)

4. **Run the application:**
   ```bash
   ./run.sh
   ```

## Model Recommendations

### For Ollama:

- **llama3.2** (3B) - Good balance of speed and quality, ~2GB RAM
- **llama3.1:8b** (8B) - Better quality, requires ~8GB RAM
- **mistral** (7B) - Alternative with good performance
- **codellama** - Specialized for code-related queries

### Performance Considerations:

- **3B models**: Fast responses, adequate for basic Q&A
- **7-8B models**: Better reasoning, slower responses
- **13B+ models**: Best quality, requires significant resources

## Troubleshooting

### Ollama Issues:

1. **"Failed to connect to Ollama"**
   - Ensure Ollama server is running: `ollama serve`
   - Check if model is installed: `ollama list`
   - Verify port 11434 is available

2. **Model not found**
   - Pull the model: `ollama pull llama3.2`
   - Check available models: `ollama list`

3. **Slow responses**
   - Try a smaller model (3B instead of 8B)
   - Ensure sufficient RAM is available

### LocalAI Issues:

1. **Connection refused**
   - Verify Docker container is running: `docker ps`
   - Check port 8080 is mapped correctly

2. **Model loading errors**
   - Ensure the model name matches what's available in LocalAI
   - Check LocalAI logs: `docker logs local-ai`

### General Issues:

1. **"OpenAI package required"**
   - Install the dependency: `uv add openai`

2. **High memory usage**
   - Use smaller models
   - Close other applications
   - Consider cloud-based solutions for limited hardware

## Switching Between Providers

You can easily switch between local and cloud providers by changing the `LLM_PROVIDER` in your `.env` file:

```env
# Use Anthropic Claude
LLM_PROVIDER=anthropic

# Use local Ollama
LLM_PROVIDER=ollama

# Use LocalAI
LLM_PROVIDER=localai
```

No code changes are required - just restart the application after updating the configuration.

## Performance Comparison

| Provider | Setup Complexity | Response Speed | Quality | Resource Usage |
|----------|------------------|----------------|---------|----------------|
| Anthropic | Easy | Fast | Excellent | None (API) |
| Ollama 3B | Medium | Fast | Good | ~2GB RAM |
| Ollama 8B | Medium | Medium | Very Good | ~8GB RAM |
| LocalAI | Hard | Variable | Variable | Variable |

## Privacy Benefits

Local LLMs provide several privacy advantages:
- No data sent to external APIs
- Full control over model and data
- No API costs or rate limits
- Works offline once models are downloaded