/**
 * Signova AI Frontend API Service
 * Handles communication with FastAPI backend endpoints:
 * - /api/v1/recognize/isolated/video (Isolated sign recognition)
 * - /api/v1/recognize/isolated/landmarks (Direct landmark array)
 * - /api/v1/translate (Sarvam translation - FE-08)
 * - /api/v1/translate/languages (Supported languages)
 * - /api/v1/speech/synthesize (Sarvam Bulbul TTS - FE-08)
 * - /api/v1/speech/speakers (Supported speakers)
 * - /api/v1/predict/video (iSign T5 translation)
 * - /api/v1/predict/landmarks (iSign T5 landmarks)
 * - /api/v1/health (system health check)
 */

const API_BASE = "/api/v1";

/**
 * Send video file to backend for MediaPipe extraction and 10-class Isolated ISL recognition.
 * @param {Blob|File} fileBlob - Recorded video blob or uploaded file
 * @param {string} fileName - Filename with extension (e.g. "recording.webm")
 * @param {string|null} category - Optional sign category filter
 * @param {number} topK - Number of candidate signs to return (default 3)
 */
export async function recognizeIsolatedVideo(fileBlob, fileName = "recording.webm", category = null, topK = 3) {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);

  const queryParams = new URLSearchParams();
  if (category) queryParams.set("category", category);
  queryParams.set("top_k", topK);

  const response = await fetch(`${API_BASE}/recognize/isolated/video?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Server returned error ${response.status}`);
  }

  return response.json();
}

/**
 * Send pre-extracted landmark sequence directly for Isolated ISL sign recognition.
 * @param {number[][]} landmarks - Array of frames with 75 keypoints (x,y,z,conf)
 * @param {string|null} category - Optional category filter
 * @param {number} topK - Candidate count
 */
export async function recognizeIsolatedLandmarks(landmarks, category = null, topK = 3) {
  const response = await fetch(`${API_BASE}/recognize/isolated/landmarks`, {
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

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Server returned error ${response.status}`);
  }

  return response.json();
}

/**
 * Translate text into target language using Sarvam AI (FE-08).
 * @param {string} text - Message text to translate
 * @param {string} targetLanguageCode - BCP-47 language code (e.g. 'hi-IN')
 * @param {string} sourceLanguageCode - Source language code (default 'en-IN')
 */
export async function translateText(text, targetLanguageCode, sourceLanguageCode = "en-IN") {
  const response = await fetch(`${API_BASE}/translate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: text,
      target_language_code: targetLanguageCode,
      source_language_code: sourceLanguageCode,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || errorBody.message || `Translation failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch supported translation languages from backend.
 * @returns {Promise<Record<string, string>>}
 */
export async function fetchSupportedLanguages() {
  const response = await fetch(`${API_BASE}/translate/languages`);
  if (!response.ok) {
    throw new Error(`Failed to load languages (status ${response.status})`);
  }
  return response.json();
}

/**
 * Synthesize text into spoken audio using Sarvam Bulbul TTS (FE-08).
 * @param {string} text - Text to synthesize
 * @param {string} targetLanguageCode - BCP-47 language code (e.g. 'hi-IN')
 * @param {string} speaker - Voice persona ('meera', 'arvind', etc.)
 * @param {number} pace - Speed multiplier (default 1.0)
 */
export async function synthesizeSpeech(text, targetLanguageCode, speaker = "meera", pace = 1.0) {
  const response = await fetch(`${API_BASE}/speech/synthesize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text,
      target_language_code: targetLanguageCode,
      speaker,
      pace,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || errorBody.message || `Text-to-speech failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch supported TTS speakers from backend.
 * @returns {Promise<string[]>}
 */
export async function fetchSupportedSpeakers() {
  const response = await fetch(`${API_BASE}/speech/speakers`);
  if (!response.ok) {
    throw new Error(`Failed to load speakers (status ${response.status})`);
  }
  return response.json();
}

/**
 * Transcribe recorded audio file using Sarvam Saaras STT via backend.
 * @param {Blob|File} audioBlob - Recorded audio blob
 * @param {string} fileName - Filename (e.g. "recording.webm")
 * @param {string|null} languageCode - BCP-47 language code (e.g. 'en-IN')
 */
export async function transcribeSpeech(audioBlob, fileName = "recording.webm", languageCode = null) {
  const formData = new FormData();
  formData.append("audio", audioBlob, fileName);

  const queryParams = new URLSearchParams();
  if (languageCode) queryParams.set("language_code", languageCode);

  const response = await fetch(`${API_BASE}/speech/transcribe?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || errorBody.message || `Speech transcription failed with status ${response.status}`);
  }

  return response.json();
}

/**
 * Direction B: Transcribe Other Person's audio and translate into Your Language.
 * @param {Blob|File} audioBlob - Audio recording from microphone
 * @param {string} fileName - Filename (e.g. "speech.webm")
 * @param {string} sourceLanguageCode - Other person's language (e.g. 'en-IN')
 * @param {string} targetLanguageCode - Your readable language (e.g. 'gu-IN')
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

  const response = await fetch(`${API_BASE}/speech/transcribe-and-translate?${queryParams.toString()}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || errorBody.message || `Voice translation failed with status ${response.status}`);
  }

  return response.json();
}


/**
 * Send a video or .pose file to backend for MediaPipe extraction and iSign T5 translation.
 */
export async function predictVideo(fileBlob, fileName = "recording.webm", decoding = "beam") {
  const formData = new FormData();
  formData.append("video", fileBlob, fileName);

  const response = await fetch(`${API_BASE}/predict/video?decoding=${encodeURIComponent(decoding)}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Server returned error ${response.status}`);
  }

  return response.json();
}

/**
 * Send pre-extracted 300-dim landmark sequence to backend.
 */
export async function predictLandmarks(landmarks, decoding = "beam") {
  const response = await fetch(`${API_BASE}/predict/landmarks`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      landmarks,
      decoding,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Server returned error ${response.status}`);
  }

  return response.json();
}

/**
 * Check backend health status.
 */
export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }
  return response.json();
}
