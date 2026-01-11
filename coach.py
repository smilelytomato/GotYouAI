"""
LLM-powered coaching and suggestion generation module.
Handles model inference and response parsing for GotYouAI coaching assistant.
"""

import os
import torch
import random
import time
from typing import Dict, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .utils import extract_suggestions, remove_suggestions_from_text


# =========================================================
# Prompt Loading Functions
# =========================================================

def _load_prompt_file(filename: str) -> str:
    """Load a prompt text file."""
    prompts_dir = Path(__file__).parent / "prompts"
    file_path = prompts_dir / filename
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"[WARNING] Prompt file is empty: {file_path}")
            return content
    except FileNotFoundError:
        print(f"[ERROR] Prompt file not found: {file_path}")
        raise FileNotFoundError(f"Prompt file not found: {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load prompt file {file_path}: {e}")
        raise


def build_user_prompt(conversation_text: str, state: str) -> str:
    """Build the user prompt for the LLM."""
    template = _load_prompt_file("user_prompt_template.txt")
    return template.format(conversation_text=conversation_text, state=state)


def build_state_specific_prompt(state: str) -> str:
    """Build the system prompt with state-specific instructions."""
    base_prompt = _load_prompt_file("base_prompt.txt")
    
    # Load state-specific prompt
    state_file_map = {
        "cooking": "cooking_state.txt",
        "wait": "wait_state.txt",
        "cooked": "cooked_state.txt"
    }
    
    state_file = state_file_map.get(state, "cooked_state.txt")
    state_prompt = _load_prompt_file(state_file)
    
    return base_prompt + "\n\n" + state_prompt


# =========================================================
# Model Loading (Cached)
# =========================================================

_model = None
_tokenizer = None
_pipeline = None
_model_loading = False
_model_loaded = False


def is_model_loaded() -> bool:
    """Check if model is already loaded."""
    return _model_loaded and _model is not None


def get_model():
    """Get or create model and tokenizer (cached for performance)."""
    global _model, _tokenizer, _pipeline, _model_loading, _model_loaded
    
    if _model is None or _tokenizer is None:
        _model_loading = True
        # Using Qwen2.5-3B-Instruct: publicly accessible instruction model (~3B parameters)
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print(f"[DEBUG] Loading model: {model_name}")
        
        try:
            # Check if CUDA is available
            use_cuda = torch.cuda.is_available()
            device = 0 if use_cuda else -1
            
            print(f"[DEBUG] Using device: {'CUDA' if use_cuda else 'CPU'}")
            
            # Load tokenizer
            _tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not set (required for some models)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            # Gemma models may need token to be set
            if hasattr(_tokenizer, 'pad_token_id') and _tokenizer.pad_token_id is None:
                _tokenizer.pad_token_id = _tokenizer.eos_token_id
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True
            }
            
            # Try to use 8-bit quantization for faster inference (if available)
            try:
                from transformers import BitsAndBytesConfig
                if use_cuda:
                    # Use 8-bit quantization on GPU for faster inference
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
                    model_kwargs.update({
                        "quantization_config": quantization_config,
                        "device_map": "auto"
                    })
                else:
                    model_kwargs.update({
                        "torch_dtype": torch.float32
                    })
            except ImportError:
                # bitsandbytes not available, use standard loading
                if use_cuda:
                    model_kwargs.update({
                        "torch_dtype": torch.float16,
                        "device_map": "auto"
                    })
                else:
                    model_kwargs.update({
                        "torch_dtype": torch.float32
                    })
            
            _model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not use_cuda:
                _model = _model.to("cpu")
            
            # Create pipeline for easier generation
            _pipeline = pipeline(
                "text-generation",
                model=_model,
                tokenizer=_tokenizer,
                device=device,
                torch_dtype=torch.float16 if use_cuda else torch.float32
            )
            
            print(f"[DEBUG] Model loaded successfully")
            _model_loaded = True
            _model_loading = False
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    return _pipeline, _tokenizer


# =========================================================
# Public API
# =========================================================

def generate_coaching(
    conversation_text: str,
    state: str,
    hf_api_token: Optional[str] = None  # Kept for backward compatibility, not used
) -> Dict[str, any]:
    """
    Generate coaching feedback and message suggestions using Qwen2.5-3B-Instruct model.
    """
    
    # Build prompts with error handling
    try:
        system_prompt = build_state_specific_prompt(state)
        user_prompt = build_user_prompt(conversation_text, state)
        print(f"[DEBUG] System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}")
    except Exception as e:
        print(f"[ERROR] Failed to load prompts: {e}")
        return {
            "coaching": f"Failed to load prompts: {str(e)}. Please check that prompt files exist.",
            "suggestions": [],
            "state": state
        }

    llm_output = None
    error_message = None

    try:
        llm_output = call_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    except RuntimeError as e:
        error_message = str(e)
        print(f"[MODEL ERROR] {e}")
    except Exception as e:
        error_message = f"Model inference failed: {str(e)}"
        print(f"[MODEL ERROR] {e}")

    if not llm_output:
        print("[DEBUG] LLM output is None or empty")
        if error_message:
            print(f"[DEBUG] Error details: {error_message}")
        # Return error result if LLM fails
        return {
            "coaching": error_message or "Unable to generate coaching. Please check the model and try again.",
            "suggestions": [],
            "state": state
        }
    
    print(f"[DEBUG] LLM Output length: {len(llm_output)}")
    print(f"[DEBUG] LLM Output preview: {llm_output[:500]}")

    suggestions = extract_suggestions(llm_output)
    print(f"[DEBUG] Extracted suggestions count: {len(suggestions)}")
    print(f"[DEBUG] Extracted suggestions: {suggestions}")
    
    if not suggestions:
        print(f"[DEBUG] WARNING: No suggestions found in {{}} format!")
        # Check if there are any curly brackets at all
        if '{' in llm_output or '}' in llm_output:
            print(f"[DEBUG] Found curly brackets but extraction failed. Full output: {llm_output}")
    
    if len(suggestions) > 9:
        suggestions = suggestions[:9]

    coaching_text = remove_suggestions_from_text(llm_output).strip()
    if not coaching_text:
        coaching_text = "Review the conversation and respond appropriately."

    return {
        "coaching": coaching_text,
        "suggestions": suggestions,
        "state": state
    }


# =========================================================
# Qwen Model Inference
# =========================================================

def call_model(
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Generate text using TinyLlama-1.1B-Chat model.
    Uses the tokenizer's chat template for proper formatting.
    """
    
    try:
        pipeline, tokenizer = get_model()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")
    
    # Format prompt using tokenizer's chat template (recommended approach)
    # Qwen uses a specific chat format with system/user roles
    try:
        # Try using the tokenizer's apply_chat_template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual formatting for Qwen
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    except Exception as e:
        print(f"[WARNING] Failed to use chat template, using fallback: {e}")
        # Fallback to manual formatting for Qwen
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    print(f"[DEBUG] Formatted prompt length: {len(formatted_prompt)}")
    print(f"[DEBUG] Prompt preview: {formatted_prompt[:200]}...")
    
    try:
        # Set random seed for variation (use current time + random for uniqueness)
        # This ensures different outputs even with same prompt
        seed = int(time.time() * 1000) % (2**31) + random.randint(0, 1000)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Generate text with higher temperature for more variation
        # Reduced max_new_tokens from 300 to 200 for faster generation
        outputs = pipeline(
            formatted_prompt,
            max_new_tokens=200,  # Reduced from 300 for faster generation
            temperature=0.8,  # Increased from 0.55 for more variation
            top_p=0.95,  # Increased from 0.9 for more diversity
            do_sample=True,
            repetition_penalty=1.15,
            return_full_text=False,
            truncation=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract generated text
        if isinstance(outputs, list) and len(outputs) > 0:
            generated = outputs[0].get("generated_text", "").strip()
        elif isinstance(outputs, dict):
            generated = outputs.get("generated_text", "").strip()
        else:
            generated = str(outputs).strip()
        
        print(f"[DEBUG] Generated text length: {len(generated)}")
        print(f"[DEBUG] Generated text preview: {generated[:500]}")
        
        if not generated:
            raise RuntimeError("Model generated empty output")
        
        return generated
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        raise RuntimeError(f"Text generation failed: {str(e)}")