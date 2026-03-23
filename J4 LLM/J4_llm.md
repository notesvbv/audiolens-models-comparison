# J4 — LLM Selection for Contextual Extraction

## Overview

AudioLens requires a large language model to serve as its contextual intelligence layer. After the document classifier identifies the document type and the OCR engine extracts raw text, the LLM receives both inputs and produces a concise, natural-language summary optimised for spoken delivery via text-to-speech. This is not a generic summarisation task - the model must understand document structure, prioritise actionable information (dates, amounts, names, required actions), and produce output that sounds natural when read aloud to a visually impaired user. The selection of this model was therefore driven by a specific set of constraints tied to AudioLens's accessibility-first architecture.

## Selection Criteria

Five criteria guided the evaluation. First, **multimodal capability** - while the primary input to the LLM is extracted text, the ability to accept images directly provides a valuable fallback path when OCR quality is poor or when the document contains visual elements (logos, stamps, charts) that text extraction alone cannot capture. Second, **latency** - AudioLens operates as a real-time assistant where the user is waiting for spoken output after capturing a photo. Any model introducing more than 3–5 seconds of latency at the summarisation stage would degrade the user experience significantly. Third, **cost and quota** — as a university project deployed on free or low-cost infrastructure, the model must offer a generous free tier or very low per-token pricing. Fourth, **output quality for spoken delivery** - the model must reliably produce clean, conversational prose without markdown formatting, bullet points, or other artefacts that would sound unnatural when passed through TTS. Fifth, **API simplicity and reliability** - straightforward REST API integration with minimal authentication complexity and strong uptime.

## Candidates Evaluated

Four primary candidates were considered: Google Gemini Flash, Meta LLaMA (via Groq and direct hosting), Anthropic Claude, and OpenAI GPT-4.

### Google Gemini 2.0 Flash

Gemini Flash immediately stood out as the strongest candidate for several reasons. It is natively multimodal, accepting both text and images in a single API call. This means that in the fallback scenario where HuggingFace GPU quota is exhausted, the PWA can send the raw document image directly to Gemini, which then performs classification, text extraction, and contextual summarisation in a single request - eliminating the need for the entire HF pipeline. The free tier offers 1,500 requests per day with generous token limits, which is more than sufficient for a university project. Latency benchmarks from community testing on the Google AI Studio forums and independent evaluations on platforms like Artificial Analysis consistently placed Gemini Flash among the fastest commercially available models, typically responding in under 2 seconds for short summarisation tasks. The output quality was tested extensively during development and found to be highly controllable - with appropriate system prompting, Gemini reliably produced clean spoken-style prose without formatting artefacts.

### Meta LLaMA 3 via Groq

Groq's hardware-accelerated inference of open-source LLaMA models was an attractive option due to its exceptional speed. Groq consistently delivers sub-second inference times, making it the fastest option evaluated. However, several limitations made it less suitable for AudioLens. LLaMA models hosted on Groq are text-only - they cannot accept image inputs, which eliminates the multimodal fallback path that is central to AudioLens's resilience strategy. Additionally, the output from LLaMA 3 required more aggressive prompting to avoid markdown-style formatting in responses, and even with careful prompting, the results were less consistent than Gemini for the specific task of producing TTS-optimised summaries.

### Self-Hosted LLaMA via Ollama

Running LLaMA locally on the Mac Mini (M2 Pro, 16GB) via Ollama was explored as a zero-cost, offline-capable option. While this aligns with AudioLens's principle of offline resilience, the practical limitations were significant. The 16GB RAM constraint restricts the usable model to LLaMA 3 8B or smaller quantised variants, which produced noticeably lower quality summaries compared to Gemini Flash — particularly for complex documents like invoices with multiple line items or medical documents with domain-specific terminology. Inference times on the M2 Pro averaged 4–8 seconds for typical summarisation prompts, which is acceptable but not ideal. The lack of multimodal capability remained the decisive drawback. However, local LLaMA remains a viable future enhancement for a fully offline mode.

### OpenAI GPT-4 and Anthropic Claude

Both GPT-4 and Claude were evaluated briefly. Both produce excellent output quality and support multimodal inputs. However, neither offers a meaningful free tier - OpenAI requires pay-as-you-go billing from the first request, and Claude's API pricing, while competitive, still represents an ongoing cost. For a student project where the deployment must remain functional beyond the submission date without accumulating charges, the absence of a free tier was a significant disadvantage. Community discussions on Reddit (r/LocalLLaMA, r/MachineLearning) and developer forums consistently recommended Gemini Flash as the best balance of quality, speed, and free-tier generosity for student and prototype projects, which reinforced the decision.

## Community Input and Research

The selection was informed by several community sources beyond direct benchmarking. The Artificial Analysis leaderboard, which tracks speed, quality, and pricing across commercial LLM APIs, consistently ranks Gemini Flash among the top models for speed-to-quality ratio. Discussions on the HuggingFace forums regarding ZeroGPU Spaces frequently recommend offloading LLM calls to external APIs rather than running them on ZeroGPU, with Gemini Flash being the most commonly suggested option due to its free tier. Developer experience reports on Reddit and the Google AI Developer Forum highlighted Gemini's reliable JSON-mode output and low hallucination rate on structured document tasks as particular strengths.

## Final Decision

Google Gemini 2.0 Flash was selected as AudioLens's contextual extraction model. The decision rests on four pillars: native multimodal capability enabling the single-call fallback pipeline, sub-2-second latency preserving the real-time user experience, a generous free tier ensuring the project remains deployable without ongoing cost, and consistently high output quality for spoken-style summarisation with minimal prompt engineering. The model is called via a server-side proxy route in the Next.js PWA, keeping the API key secure and enabling straightforward rate limit handling. This architecture also allows the LLM provider to be swapped in the future — if Groq adds multimodal support or if a superior free-tier option emerges, the change is isolated to a single API route.