"""
Writing Coach: Automatic English Writing Evaluation System
Uses simplified multi-agent approach for comprehensive writing feedback.
"""

import os
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import httpx
from pathlib import Path

# Import configuration management
from config import get_config, validate_environment

# Set environment variable to avoid OpenMP conflict with EasyOCR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Configure logging
config = get_config()
logging.basicConfig(level=getattr(logging, config.log_level), format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Structure for writing evaluation output"""
    style_and_topic: str
    strengths: list
    weaknesses: list
    improvement_suggestions: list
    refined_sample: str
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class WritingCoachSystem:
    """
    Simplified writing evaluation system using multiple LLM providers.
    Compatible with OpenAI, Gemini, Anthropic, Azure, Ollama, and other providers.
    Includes OCR functionality for image-to-text conversion.
    """

    def __init__(self, provider: Optional[str] = None, model_filter: Optional[List[str]] = None):
        """
        Initialize the writing coach system.
        
        Args:
            provider: LLM provider to use (openai, gemini, anthropic, azure, ollama, etc.)
                     If None, uses default provider from config
            model_filter: List of model names to filter (e.g., ['gpt-4', 'mistral'])
        """
        # Validate environment configuration
        if not validate_environment():
            raise ValueError("Environment configuration is invalid. Please check your .env file.")
        
        self.config = get_config()
        self.provider = provider or self.config.default_llm_provider
        
        # Get LLM configuration for the specified provider
        try:
            self.llm_config = self.config.get_llm_config(self.provider)
        except ValueError as e:
            # Fall back to available providers
            available = self.config.get_available_providers()
            if available:
                logger.warning(f"Provider '{self.provider}' not available. Using '{available[0]}'")
                self.provider = available[0]
                self.llm_config = self.config.get_llm_config(self.provider)
            else:
                raise ValueError("No LLM providers are configured. Please set API keys in your .env file.")
        
        # Initialize client based on provider
        self.client = self._init_client()
        self.model = self.llm_config.get('model', 'default')
        
        logger.info(f"✓ WritingCoachSystem initialized with provider: {self.provider}, model: {self.model}")
        logger.info(f"📷 OCR: Using {self.provider.title()} Vision for image text extraction")

    def _init_client(self):
        """Initialize client based on provider type"""
        try:
            api_type = self.llm_config.get('api_type', 'openai')
            
            if api_type == 'openai' or self.provider == 'openai':
                return self._init_openai_client()
            elif api_type == 'gemini' or self.provider == 'gemini':
                return self._init_gemini_client()
            elif api_type == 'anthropic' or self.provider == 'anthropic':
                return self._init_anthropic_client()
            elif api_type == 'azure' or self.provider == 'azure':
                return self._init_azure_client()
            elif api_type == 'open_ai' or self.provider == 'ollama':
                return self._init_ollama_client()
            elif api_type == 'qwen' or self.provider == 'qwen':
                return self._init_qwen_client()
            elif api_type == 'openai' or self.provider == 'deepseek':
                return self._init_deepseek_client()
            else:
                # Generic OpenAI-compatible client
                return self._init_generic_client()
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            return None

    def _init_openai_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            api_key = self.llm_config.get('api_key')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            client = OpenAI(
                api_key=api_key,
                base_url=self.llm_config.get('api_base', 'https://api.openai.com/v1')
            )
            self.use_openai_sdk = True
            return client
        except ImportError:
            logger.warning("OpenAI SDK not installed. Falling back to HTTP client.")
            return self._init_generic_client()

    def _init_gemini_client(self):
        """Initialize Gemini client"""
        try:
            import google.generativeai as genai
            api_key = self.llm_config.get('api_key')
            if not api_key:
                raise ValueError("Gemini API key not found")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.llm_config.get('model', 'gemini-pro'))
            self.use_gemini_sdk = True
            return model
        except ImportError:
            logger.error("Google Generative AI SDK not installed. Install with: pip install google-generativeai")
            raise ImportError("Google Generative AI SDK required for Gemini")

    def _init_anthropic_client(self):
        """Initialize Anthropic client"""
        try:
            import anthropic
            api_key = self.llm_config.get('api_key')
            if not api_key:
                raise ValueError("Anthropic API key not found")
            
            client = anthropic.Anthropic(api_key=api_key)
            self.use_anthropic_sdk = True
            return client
        except ImportError:
            logger.error("Anthropic SDK not installed. Install with: pip install anthropic")
            raise ImportError("Anthropic SDK required for Claude")

    def _init_azure_client(self):
        """Initialize Azure OpenAI client"""
        try:
            from openai import AzureOpenAI
            api_key = self.llm_config.get('api_key')
            endpoint = self.llm_config.get('api_base')
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API key and endpoint required")
            
            client = AzureOpenAI(
                api_key=api_key,
                api_version=self.llm_config.get('api_version', '2024-02-15-preview'),
                azure_endpoint=endpoint
            )
            self.use_azure_sdk = True
            return client
        except ImportError:
            logger.warning("OpenAI SDK not installed. Falling back to HTTP client.")
            return self._init_generic_client()

    def _init_ollama_client(self):
        """Initialize Ollama client"""
        self._httpx_client = httpx.Client(timeout=60.0)
        self.use_ollama = True
        return 'ollama'

    def _init_qwen_client(self):
        """Initialize Qwen client"""
        try:
            from openai import OpenAI
            
            api_key = self.llm_config.get('api_key')
            base_url = self.llm_config.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            
            if not api_key:
                raise ValueError("Qwen API key not found")
            
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.use_qwen_sdk = True
            return client
        except ImportError:
            logger.warning("OpenAI SDK not installed. Falling back to HTTP client.")
            return self._init_generic_client()

    def _init_deepseek_client(self):
        """Initialize DeepSeek client"""
        try:
            from openai import OpenAI
            
            api_key = self.llm_config.get('api_key')
            base_url = self.llm_config.get('api_base', 'https://api.deepseek.com')
            
            if not api_key:
                raise ValueError("DeepSeek API key not found")
            
            client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self.use_deepseek_sdk = True
            return client
        except ImportError:
            logger.warning("OpenAI SDK not installed. Falling back to HTTP client.")
            return self._init_generic_client()

    def _init_generic_client(self):
        """Initialize generic HTTP client for OpenAI-compatible APIs"""
        self._httpx_client = httpx.Client(timeout=60.0)
        self.use_generic = True
        return 'generic'

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Call LLM with system and user message based on provider"""
        try:
            # OpenAI SDK (OpenAI, Azure)
            if getattr(self, 'use_openai_sdk', False) or getattr(self, 'use_azure_sdk', False):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content

            # Gemini SDK
            elif getattr(self, 'use_gemini_sdk', False):
                # Combine system and user prompts for Gemini
                combined_prompt = f"System: {system_prompt}\n\nUser: {user_message}"
                response = self.client.generate_content(combined_prompt)
                return response.text

            # Anthropic SDK
            elif getattr(self, 'use_anthropic_sdk', False):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message}
                    ]
                )
                return response.content[0].text

            # Ollama HTTP API
            elif getattr(self, 'use_ollama', False):
                return self._call_ollama(system_prompt, user_message)

            # Qwen SDK (OpenAI compatible)
            elif getattr(self, 'use_qwen_sdk', False):
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content

            # Generic OpenAI-compatible HTTP API
            else:
                return self._call_generic_api(system_prompt, user_message)

        except Exception as e:
            logger.error(f"LLM call failed for {self.provider}: {e}")
            return ""

    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        """Call Ollama API"""
        try:
            api_base = self.llm_config.get('api_base', 'http://localhost:11434/v1').rstrip('/')
            
            # Remove /v1 suffix for Ollama's native API
            if api_base.endswith('/v1'):
                base_root = api_base[:-3]
            else:
                base_root = api_base
            
            url = f"{base_root}/api/generate"
            prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_message}"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = self._httpx_client.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    response_text = data.get('response', '')
                    if response_text:
                        return response_text.strip()
                return ''
            else:
                try:
                    err = response.json()
                except Exception:
                    err = response.text
                logger.error(f"Ollama API call failed: {response.status_code} - {err}")
                
                if response.status_code == 404:
                    logger.error(f"Model '{self.model}' not found. Pull it with: ollama pull {self.model}")
                return ''
                
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""

    def _call_generic_api(self, system_prompt: str, user_message: str) -> str:
        """Call generic OpenAI-compatible API"""
        try:
            api_base = self.llm_config.get('api_base', 'https://api.openai.com/v1').rstrip('/')
            url = f"{api_base}/chat/completions"
            
            headers = {"Content-Type": "application/json"}
            api_key = self.llm_config.get('api_key')
            if api_key and api_key != 'NA':
                headers['Authorization'] = f"Bearer {api_key}"
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            response = self._httpx_client.post(url, headers=headers, json=payload)
            
            if response.status_code in (200, 201):
                data = response.json()
                choices = data.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    return message.get('content', '').strip()
            
            try:
                err = response.json()
            except Exception:
                err = response.text
            logger.error(f"Generic API call failed: {response.status_code} - {err}")
            return ''
            
        except Exception as e:
            logger.error(f"Generic API call failed: {e}")
            return ""


    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using model-appropriate OCR functionality
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from the image
        """
        try:
            # Validate image file exists
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Validate file extension
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in valid_extensions:
                raise ValueError(f"Unsupported image format: {file_ext}. Supported: {valid_extensions}")
            
            logger.info(f"📷 Extracting text from image: {image_path}")
            logger.info(f"🤖 Using {self.provider} for OCR...")
            
            # Extract text using appropriate model
            if self.provider == 'qwen':
                extracted_text = self._extract_text_with_qwen_vision(image_path)
            elif self.provider in ['gemini']:
                extracted_text = self._extract_text_with_gemini_vision(image_path)
            else:
                # For other providers, try to use the generic multimodal capability
                extracted_text = self._extract_text_with_multimodal_model(image_path)
            
            # Basic text cleanup
            extracted_text = self._clean_ocr_text(extracted_text)
            
            logger.info(f"✓ Extracted {len(extracted_text)} characters from image")
            
            if not extracted_text.strip():
                logger.warning("⚠️ No text detected in image")
                return "No text could be detected in the image. Please ensure the image contains clear, readable text."
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"✗ OCR extraction failed: {e}")
            raise

    def _extract_text_with_qwen_vision(self, image_path: str) -> str:
        """Extract text using Qwen Vision"""
        try:
            from openai import OpenAI
            import base64
            
            # Initialize client
            api_key = self.llm_config.get('api_key')
            base_url = self.llm_config.get('api_base', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            
            if not api_key:
                raise ValueError("Qwen API key not found. Please set QWEN_API_KEY in your .env file.")
            
            client = OpenAI(api_key=api_key, base_url=base_url)
            
            logger.info(f"🤖 Using Qwen Vision for OCR...")
            logger.info(f"🔧 Model: {self.model}")
            
            # Read image and encode to base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('ascii')
            
            # Prompt for text extraction
            prompt = """
            Please carefully examine this image and extract ALL the text you can see.
            
            Instructions:
            1. Read every word, sentence, and paragraph you can identify
            2. Maintain the original structure and line breaks where possible
            3. If some words are unclear, make your best guess
            4. Include any titles, headings, or special formatting you notice
            5. Provide the extracted text in a clean, readable format
            
            Please provide ONLY the extracted text content, without any additional commentary.
            """
            
            logger.info("🔍 Analyzing image with Qwen Vision...")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_string}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                logger.warning("⚠️ Qwen Vision returned no text")
                return ""
                
        except Exception as e:
            logger.error(f"✗ Qwen Vision extraction failed: {e}")
            raise

    def _extract_text_with_multimodal_model(self, image_path: str) -> str:
        """Extract text using any multimodal model that supports image input"""
        try:
            import base64
            
            # Read image and encode to base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('ascii')
            
            # Prompt for text extraction
            prompt = """
            Please carefully examine this image and extract ALL the text you can see.
            
            Instructions:
            1. Read every word, sentence, and paragraph you can identify
            2. Maintain the original structure and line breaks where possible
            3. If some words are unclear, make your best guess
            4. Include any titles, headings, or special formatting you notice
            5. Provide the extracted text in a clean, readable format
            
            Please provide ONLY the extracted text content, without any additional commentary.
            """
            
            # Prepare the message content based on the provider
            if getattr(self, 'use_openai_sdk', False) or getattr(self, 'use_azure_sdk', False) or getattr(self, 'use_qwen_sdk', False):
                # For OpenAI-like APIs
                message_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_string}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": message_content}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content.strip()
            else:
                # For other providers, we fallback to Gemini or return an error
                logger.warning(f"⚠️ Image processing not implemented for {self.provider}, falling back to Gemini Vision")
                return self._extract_text_with_gemini_vision(image_path)
                
        except Exception as e:
            logger.error(f"✗ Multimodal model extraction failed: {e}")
            raise

    def _extract_text_with_gemini_vision(self, image_path: str) -> str:
        """Extract text using Gemini Vision"""
        try:
            import google.generativeai as genai
            import PIL.Image
            
            # Configure Gemini (use same API key as LLM)
            api_key = self.config.gemini_api_key
            if not api_key:
                raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")
            
            genai.configure(api_key=api_key)
            
            # Initialize Gemini model with vision capabilities
            model = genai.GenerativeModel(self.config.gemini_model)
            
            logger.info(f"🤖 Using Gemini Vision for OCR...")
            logger.info(f"🔧 Model: {self.config.gemini_model}")
            
            # Load image
            image = PIL.Image.open(image_path)
            
            # Prompt for text extraction
            prompt = """
            Please carefully examine this image and extract ALL the text you can see.
            
            Instructions:
            1. Read every word, sentence, and paragraph you can identify
            2. Maintain the original structure and line breaks where possible
            3. If some words are unclear, make your best guess
            4. Include any titles, headings, or special formatting you notice
            5. Provide the extracted text in a clean, readable format
            
            Please provide ONLY the extracted text content, without any additional commentary.
            """
            
            logger.info("🔍 Analyzing image with Gemini Vision...")
            
            # Generate content with image and prompt
            response = model.generate_content([prompt, image])
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("⚠️ Gemini Vision returned no text")
                return ""
                
        except ImportError:
            logger.error("✗ google-generativeai or PIL not installed. Install with: pip install google-generativeai pillow")
            raise ImportError("google-generativeai and pillow are required for Gemini Vision OCR")
        except Exception as e:
            logger.error(f"✗ Gemini Vision extraction failed: {e}")
            raise

    def _clean_ocr_text(self, text: str) -> str:
        """Clean up OCR-extracted text"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.,!?;:\'"()-]', '', text)
        
        # Fix common OCR mistakes
        replacements = {
            ' l ': ' I ',  # lowercase l often mistaken for I
            ' 0 ': ' O ',  # zero often mistaken for O
            '|': 'I',      # pipe often mistaken for I
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()

    def evaluate_writing_from_image(self, image_path: str) -> EvaluationResult:
        """
        Extract text from image and evaluate it
        
        Args:
            image_path: Path to the image file containing text
            
        Returns:
            EvaluationResult with analysis of the extracted text
        """
        logger.info("="*60)
        logger.info("Starting image-to-text evaluation...")
        logger.info("="*60)
        
        # Extract text from image
        extracted_text = self.extract_text_from_image(image_path)
        
        # Add note about OCR extraction
        logger.info(f"📝 Evaluating extracted text ({len(extracted_text)} characters)...")
        
        # Evaluate the extracted text
        result = self.evaluate_writing(extracted_text)
        
        # Add OCR note to the result
        ocr_note = f"\n\n--- OCR EXTRACTION NOTE ---\nThis evaluation is based on text extracted from an image using {self.provider.title()} Vision technology. " \
                  f"Both the text extraction and writing analysis were performed by the same AI system using the {self.model} model."
        
        # Append OCR note to style_and_topic
        result.style_and_topic += ocr_note
        
        logger.info("✓ Image-to-text evaluation complete!")
        
        return result

    def evaluate_writing(self, writing_sample: str) -> EvaluationResult:
        """
        Complete writing evaluation workflow.
        """
        logger.info("="*60)
        logger.info("Starting writing evaluation...")
        logger.info("="*60)
        
        # --- PHASE 1: STYLE & TOPIC ANALYSIS ---
        logger.info("\n[PHASE 1] Analyzing style and topic...")
        style_prompt = f"""Analyze this writing sample for style, topic, tone, and genre:

WRITING SAMPLE:
---
{writing_sample}
---

Provide response in TWO PARTS:

PART 1 - SUMMARY (2-3 sentences): Briefly describe the overall writing style, main topic, tone, and intended audience.

PART 2 - DETAILED ANALYSIS: Provide detailed analysis covering:
1. Writing style (formal, informal, academic, etc.)
2. Topic and subject matter
3. Overall tone
4. Genre and format
5. Target audience

CRITICAL: Be specific and provide clear examples from the text, using exact quotes where possible."""

        style_system = "You are an expert writing style analyst. Provide clear, structured analysis."
        style_response = self._call_llm(style_system, style_prompt)
        style_and_topic = style_response.strip() if style_response else "Unable to analyze style"
        logger.info("✓ Style analysis complete")
        
        # --- PHASE 2: EVALUATE STRENGTHS & WEAKNESSES ---
        logger.info("\n[PHASE 2] Evaluating strengths and weaknesses...")
        eval_prompt = f"""Analyze this writing for strengths and weaknesses:

Analyze this writing for strengths and weaknesses.

CRITICAL DEFINITIONS:
- STRENGTH: Something done WELL that demonstrates writing skill. Must show quality, not just presence. For example:
    * "Uses vivid descriptive language: 'golden sunset'" is a strength
    * "Mentions the Arctic fox" is NOT a strength (just a topic choice)
    * "Uses quotation marks" is NOT a strength (using formatting is basic, not skilled use)
    * "Properly cites sources with author and date" is a strength
    * "Has quotation marks around a fact but no proper citation" is a WEAKNESS (attempted but failed)

- WEAKNESS: Something that hurts writing quality or is done incorrectly. Examples:
    * Grammatical errors: "he don't"
    * Missing citations or improper citations (quotes without source attribution)
    * Run-on sentences or fragments
    * Vague language or unsupported claims
    * Repetitive phrasing

- NOT STRENGTHS: Don't list neutral observations as strengths
    * Simply "uses facts" = neutral, not a strength
    * "mentions multiple sources" = neutral without assessing quality
    * "has an introduction" = expected, not a strength

WRITING:
---
{writing_sample}
---

Provide response in TWO PARTS:

PART 1 - SUMMARY (2-3 sentences): Give an overall assessment focusing on what's done well vs. what needs fixing.

PART 2 - DETAILED FEEDBACK:

STRENGTHS (2-4 items maximum, only genuine quality):
For each strength:
- State the strength clearly
- Quote the specific text showing it (use exact quotes)
- Explain why this demonstrates writing skill

WEAKNESSES (2-5 items, be honest about problems):
For each weakness:
- State the weakness clearly
- Quote the specific problematic text
- Explain the impact on reader understanding

CONSTRAINT: Do NOT list neutral observations or attempts as strengths. If something is attempted but done incorrectly (like citations), mark it as a weakness."""

        eval_system = "You are a professional writing evaluator. Be specific with examples."
        eval_response = self._call_llm(eval_system, eval_prompt)
        
        # Parse strengths and weaknesses
        strengths, weaknesses = self._parse_strengths_weaknesses(eval_response)
        logger.info(f"✓ Found {len(strengths)} strengths, {len(weaknesses)} weaknesses")
        
        # --- PHASE 3: GENERATE IMPROVEMENT SUGGESTIONS ---
        logger.info("\n[PHASE 3] Generating improvement suggestions...")
        coach_prompt = f"""Based on this writing:

Based on this writing, provide specific improvement suggestions:

WRITING:
---
{writing_sample}
---

CRITICAL REQUIREMENTS:
- Every suggestion MUST reference something specific from the actual text
- Do NOT give generic writing advice like "improve grammar" without specifying what
- Each suggestion must show: LOCATION (quote or sentence number), PROBLEM (what's wrong), SOLUTION (how to fix it)
- Suggestions should be prioritized by impact

Provide response in TWO PARTS:

PART 1 - SUMMARY (2-3 sentences): Identify the top 2-3 improvements that would have the biggest impact.

PART 2 - DETAILED IMPROVEMENT SUGGESTIONS (4-6 items only - prioritized):

For each suggestion, provide in this order:
1. LOCATION: Quote the specific problematic text or describe which sentence/paragraph
2. THE ISSUE: What's the problem with this exact text
3. IMPACT: Why this matters for reader understanding
4. SPECIFIC FIX: Rewrite or revise this exact part (show before → after)
5. WHY THIS WORKS: Explain the improvement

Example of GOOD suggestion:
    LOCATION: "he don't know"
    THE ISSUE: Subject-verb disagreement (plural subject requires "do" not "don't")
    IMPACT: Errors reduce reader confidence in the writer's authority
    SPECIFIC FIX: Change "don't" to "doesn't"
    WHY THIS WORKS: Correct grammar makes the sentence clear and professional

Example of BAD suggestion:
    "Improve grammar" - Too vague, doesn't reference actual text

CONSTRAINT: EVERY suggestion must point to specific words/phrases in the writing."""

        coach_system = "You are an encouraging writing coach. Provide specific, actionable suggestions."
        coach_response = self._call_llm(coach_system, coach_prompt)
        suggestions = self._parse_suggestions(coach_response)
        logger.info(f"✓ Generated {len(suggestions)} suggestions")
        
        # --- PHASE 4: CREATE REFINED VERSION ---
        logger.info("\n[PHASE 4] Creating refined version...")
        refine_prompt = f"""Rewrite this writing, improving it while maintaining the original voice:

Rewrite this writing, improving it while maintaining the original voice and meaning:

ORIGINAL:
---
{writing_sample}
---

REVISION PRIORITIES (in order):
1. Fix critical errors (grammar, subject-verb, spelling, missing citations)
2. Improve clarity (replace vague phrases with specific ones)
3. Enhance flow (improve sentence variety and connections)
4. Strengthen evidence (add or improve citations/examples)
5. Maintain student's original voice and intended message

Provide response in TWO PARTS:

PART 1 - SUMMARY OF CHANGES (3-4 sentences): List the specific improvements made and their impact on quality.

PART 2 - REFINED VERSION:
Create an improved version that:
- Fixes all grammar/spelling/citation errors
- Replaces vague language with specific examples
- Improves sentence variety and flow
- Keeps the student's original voice and message intact
- Remains authentic to their thinking (don't rewrite their ideas)

CONSTRAINT: Do NOT change the core message or add new ideas. Improve only the expression and correctness.

After the refined version, add a brief note (2-3 sentences) explaining the specific improvements made."""

        refine_system = "You are an expert editor. Maintain original voice while improving clarity, grammar, and flow."
        refined_response = self._call_llm(refine_system, refine_prompt)
        refined_sample = refined_response.strip() if refined_response else writing_sample
        logger.info("✓ Refined version created")
        
        # Compile results
        result = EvaluationResult(
            style_and_topic=style_and_topic,
            strengths=strengths,
            weaknesses=weaknesses,
            improvement_suggestions=suggestions,
            refined_sample=refined_sample,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info("\n" + "="*60)
        logger.info("✓ Writing evaluation complete!")
        logger.info("="*60)
        
        return result

    def _parse_strengths_weaknesses(self, response: str) -> tuple:
        """Extract strengths and weaknesses from response"""
        strengths = []
        weaknesses = []
        
        try:
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if 'strength' in line.lower():
                    current_section = 'strengths'
                elif 'weakness' in line.lower():
                    current_section = 'weaknesses'
                elif line and current_section:
                    if line[0].isdigit() or line.startswith('-'):
                        # Remove bullet points and numbering
                        item = line.lstrip('0123456789.-) ').strip()
                        if item:
                            if current_section == 'strengths':
                                strengths.append(item)
                            else:
                                weaknesses.append(item)
        except:
            pass
        
        # Fallback defaults
        if not strengths:
            strengths = ["Clear writing", "Good topic selection"]
        if not weaknesses:
            weaknesses = ["Room for improvement"]
            
        return strengths[:5], weaknesses[:5]

    def _parse_suggestions(self, response: str) -> list:
        """Extract suggestions from response"""
        suggestions = []
        
        try:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and bullets
                    suggestion = line.lstrip('0123456789.-) ').strip()
                    if suggestion:
                        suggestions.append(suggestion)
        except:
            pass
        
        # Fallback
        if not suggestions:
            suggestions = ["Review grammar and punctuation", "Improve sentence variety"]
            
        return suggestions[:7]


def print_evaluation_report(result: EvaluationResult):
    """Pretty print the evaluation report"""
    print("\n" + "="*70)
    print("📝 WRITING EVALUATION REPORT")
    print("="*70)
    
    print("\n📌 STYLE & TOPIC")
    print("-" * 70)
    print(result.style_and_topic)
    
    print("\n✅ STRENGTHS")
    print("-" * 70)
    for i, strength in enumerate(result.strengths, 1):
        print(f"{i}. {strength}")
    
    print("\n⚠️  WEAKNESSES")
    print("-" * 70)
    for i, weakness in enumerate(result.weaknesses, 1):
        print(f"{i}. {weakness}")
    
    print("\n💡 IMPROVEMENT SUGGESTIONS")
    print("-" * 70)
    for i, suggestion in enumerate(result.improvement_suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\n📄 REFINED SAMPLE")
    print("-" * 70)
    print(result.refined_sample)
    
    print("\n" + "="*70)


def demo_evaluation():
    """Run demo evaluation"""
    sample_text = """
    The internet has changed how we communicate. People can now talk to anyone 
    around the world instantly. This is good because it allows people to stay 
    connected. However, it also has bad effects. Some people spends too much time 
    online and neglect real relationships. Also, there is alot of misinformation 
    spread online. Overall, the internet is both beneficial and harmful.
    """
    
    # Try to initialize with Qwen if available
    try:
        from config import get_config
        config = get_config()
        if 'qwen' in config.get_available_providers():
            coach = WritingCoachSystem(provider='qwen')
            print(f"Using Qwen model: {coach.model}")
        else:
            coach = WritingCoachSystem()
    except:
        coach = WritingCoachSystem()
    
    result = coach.evaluate_writing(sample_text)
    print_evaluation_report(result)
    
    # Also save as JSON
    with open("evaluation_result.json", "w") as f:
        f.write(result.to_json())
    logger.info("✓ Results saved to evaluation_result.json")
