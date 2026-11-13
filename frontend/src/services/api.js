/**
 * Signova AI Frontend API Service
 * Handles communication with FastAPI backend endpoints:
 * - /api/v1/recognize/isolated/video (Isolated sign recognition)
 * - /api/v1/recognize/isolated/landmarks (Direct landmark array)
 * - /api/v1/translate (Sarvam translation - FE-08)
 * - /api/v1/translate/languages (Supported languages)
 * - /api/v1/speech/synthesize (Sarvam Bulbul TTS - FE-08)
 * - /api/v1/speech/speakers (Supported speakers)
 * - /api/v1/speech/transcribe (Sarvam Saaras STT)
 * - /api/v1/speech/transcribe-and-translate (Direction B speech transcription & translation)
 * - /api/v1/predict/video (iSign T5 translation)
 * - /api/v1/predict/landmarks (iSign T5 landmarks)
 * - /api/v1/health (system health check)
 */

const API_BASE = "/api/v1";
const DEFAULT_TIMEOUT_MS = 20000;

// Track active in-flight requests to prevent duplicate network calls
const inflightRequests = new Map();

export function _clearInflightRequests() {
  inflightRequests.clear();
}

/**
 * Robust fetch wrapper with timeout, network offline detection, credit limit handling, and request deduplication.
 */
export async function requestWithTimeout(url, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS, dedupeKey = null) {
  if (dedupeKey && inflightRequests.has(dedupeKey)) {
    return inflightRequests.get(dedupeKey);
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const fetchPromise = (async () => {
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        let rawMsg =
          errorBody.error?.message ||
          errorBody.detail ||
          errorBody.message ||
          `Request failed with status ${response.status}`;

        if (response.status === 402 || /credit|quota|balance/i.test(rawMsg)) {
          rawMsg = "API credit limit exhausted. Please top up your Sarvam AI credits to continue.";
        } else if (response.status === 429 || /rate limit/i.test(rawMsg)) {
          rawMsg = "Rate limit exceeded. Please wait a moment before trying again.";
        } else if (response.status === 504 || /timed out/i.test(rawMsg)) {
          rawMsg = "Service request timed out. Please check your connection and retry.";
        } else if (response.status === 503 || /not configured|service unavailable/i.test(rawMsg)) {
          rawMsg =
            errorBody.error?.message ||
            errorBody.detail ||
            "Service is currently unavailable. Please check backend settings.";
        }
        throw new Error(rawMsg);
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        throw new Error(
          `Request timed out after ${Math.round(timeoutMs / 1000)}s. Please check your network and retry.`
        );
      }
      if (
        err instanceof TypeError &&
        (err.message.includes("fetch") || err.message.includes("NetworkError"))
      ) {
        if (typeof navigator !== "undefined" && !navigator.onLine) {
          throw new Error("Network disconnected. You are currently offline.");
        }
        throw new Error("Unable to reach backend server. Please verify network connection and retry.");
      }
      throw err;
    } finally {
      if (dedupeKey) {
        inflightRequests.delete(dedupeKey);
      }
    }
  })();

  if (dedupeKey) {
    inflightRequests.set(dedupeKey, fetchPromise);
  }

  return fetchPromise;
}

/**
 * Send video file to backend for MediaPipe extraction and 10-class Isolated ISL recognition.
 */
export async function recognizeIsolatedVideo(fileBlob, fileName = "recording.webm", category = null, topK = 3) {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);

  const queryParams = new URLSearchParams();
  if (category) queryParams.set("category", category);
  queryParams.set("top_k", topK);

  return requestWithTimeout(`${API_BASE}/recognize/isolated/video?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  }, 25000);
}

/**
 * Send pre-extracted landmark sequence directly for Isolated ISL sign recognition.
 */
export async function recognizeIsolatedLandmarks(landmarks, category = null, topK = 3) {
  return requestWithTimeout(`${API_BASE}/recognize/isolated/landmarks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      landmarks,
      category,
      top_k: topK,
    }),
  });
}

/**
 * Translate text into target language using Sarvam AI (FE-08).
 * Deduplicates concurrent identical translation requests.
 */
export async function translateText(text, targetLanguageCode, sourceLanguageCode = "en-IN") {
  const cleanText = (text || "").trim();
  const dedupeKey = `translate:${cleanText}:${targetLanguageCode}:${sourceLanguageCode}`;

  return requestWithTimeout(
    `${API_BASE}/translate`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        input: cleanText,
        target_language_code: targetLanguageCode,
        source_language_code: sourceLanguageCode,
      }),
    },
    DEFAULT_TIMEOUT_MS,
    dedupeKey
  );
}

/**
 * Fetch supported translation languages from backend.
 */
export async function fetchSupportedLanguages() {
  return requestWithTimeout(`${API_BASE}/translate/languages`);
}

/**
 * Synthesize text into spoken audio using Sarvam Bulbul TTS (FE-08).
 * Deduplicates concurrent identical synthesis requests.
 */
export async function synthesizeSpeech(text, targetLanguageCode, speaker = "meera", pace = 1.0) {
  const cleanText = (text || "").trim();
  const dedupeKey = `tts:${cleanText}:${targetLanguageCode}:${speaker}:${pace}`;

  return requestWithTimeout(
    `${API_BASE}/speech/synthesize`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: cleanText,
        target_language_code: targetLanguageCode,
        speaker,
        pace,
      }),
    },
    DEFAULT_TIMEOUT_MS,
    dedupeKey
  );
}

/**
 * Fetch supported TTS speakers from backend.
 */
export async function fetchSupportedSpeakers() {
  return requestWithTimeout(`${API_BASE}/speech/speakers`);
}

/**
 * Transcribe recorded audio file using Sarvam Saaras STT via backend.
 */
export async function transcribeSpeech(audioBlob, fileName = "recording.webm", languageCode = null) {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);

  const queryParams = new URLSearchParams();
  if (languageCode) queryParams.set("language_code", languageCode);

  return requestWithTimeout(`${API_BASE}/speech/transcribe?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  }, 25000);
}

/**
 * Direction B: Transcribe Other Person's audio and translate into Your Language.
 */
export async function transcribeAndTranslateSpeech(
  audioBlob,
  fileName = "speech.webm",
  sourceLanguageCode = "en-IN",
  targetLanguageCode = "gu-IN"
) {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);

  const queryParams = new URLSearchParams({
    source_language_code: sourceLanguageCode,
    target_language_code: targetLanguageCode,
  });

  return requestWithTimeout(`${API_BASE}/speech/transcribe-and-translate?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  }, 25000);
}

/**
 * Send a video or .pose file to backend for MediaPipe extraction and iSign T5 translation.
 */
export async function predictVideo(fileBlob, fileName = "recording.webm", decoding = "beam") {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);

  return requestWithTimeout(`${API_BASE}/predict/video?decoding=${encodeURIComponent(decoding)}`, {
    method: "POST",
    body: formData,
  }, 30000);
}

/**
 * Send pre-extracted 300-dim landmark sequence to backend.
 */
export async function predictLandmarks(landmarks, decoding = "beam") {
  return requestWithTimeout(`${API_BASE}/predict/landmarks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      landmarks,
      decoding,
    }),
  });
}

/**
 * Check backend health status.
 */
export async function fetchHealth() {
  return requestWithTimeout(`${API_BASE}/health`, {}, 5000);
}
