import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  RotateCcw,
  Square,
  AlertCircle,
  Loader2,
  CheckCircle2,
  Globe,
  Trash2,
} from "lucide-react";
import { transcribeAndTranslateSpeech, synthesizeSpeech } from "../../services/api";

export default function PanelOtherVoice({
  yourLanguage = "gu-IN",
  yourLanguageName = "Gujarati",
  otherLanguage = "en-IN",
  otherLanguageName = "English",
}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [statusState, setStatusState] = useState("idle"); // "idle" | "recording" | "processing" | "success" | "error"
  const [statusMessage, setStatusMessage] = useState("Ready for speech input");
  const [originalTranscript, setOriginalTranscript] = useState("");
  const [translatedText, setTranslatedText] = useState("");
  const [errorMessage, setErrorMessage] = useState(null);

  // Audio playback state (optional on-demand playback in Your Language)
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [isSynthesizingTTS, setIsSynthesizingTTS] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const autoStopTimeoutRef = useRef(null);
  const audioInstanceRef = useRef(null);

  useEffect(() => {
    return () => {
      stopRecordingCleanup();
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause();
        audioInstanceRef.current = null;
      }
    };
  }, []);

  const stopAudio = useCallback(() => {
    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause();
      audioInstanceRef.current.currentTime = 0;
    }
    setIsPlayingAudio(false);
  }, []);

  const stopRecordingCleanup = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (autoStopTimeoutRef.current) {
      clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const startRecording = async () => {
    setErrorMessage(null);
    audioChunksRef.current = [];
    stopAudio();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage("Microphone access is not supported by your browser.");
      setStatusState("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/wav";

      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setIsRecording(false);
        const chunks = audioChunksRef.current;
        if (!chunks || chunks.length === 0) {
          setStatusState("idle");
          setStatusMessage("No audio captured.");
          return;
        }

        const audioBlob = new Blob(chunks, { type: mimeType });
        await processAudio(audioBlob, mimeType);
      };

      recorder.start(250);
      setIsRecording(true);
      setStatusState("recording");
      setStatusMessage(`Listening to ${otherLanguageName}...`);
      setRecordingSeconds(0);

      // Start recording duration timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);

      // 15 seconds auto-stop cap
      autoStopTimeoutRef.current = setTimeout(() => {
        stopRecording();
      }, 15000);
    } catch (err) {
      setIsRecording(false);
      setStatusState("error");
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setErrorMessage("Microphone permission was denied. Please allow microphone access in browser settings.");
      } else {
        setErrorMessage(err.message || "Could not access microphone.");
      }
    }
  };

  const stopRecording = () => {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    if (autoStopTimeoutRef.current) {
      clearTimeout(autoStopTimeoutRef.current);
      autoStopTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      try {
        mediaRecorderRef.current.stop();
      } catch {
        // ignore
      }
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const processAudio = async (audioBlob, mimeType) => {
    if (!audioBlob || audioBlob.size === 0) {
      setStatusState("idle");
      setStatusMessage("Empty audio recording.");
      return;
    }

    setStatusState("processing");
    setStatusMessage("Transcribing & translating speech...");

    const fileName = mimeType.includes("webm") ? "speech.webm" : "speech.wav";

    try {
      const response = await transcribeAndTranslateSpeech(
        audioBlob,
        fileName,
        otherLanguage,
        yourLanguage
      );

      if (response && response.success) {
        setOriginalTranscript(response.original_transcript || "");
        setTranslatedText(response.translated_text || "");
        setStatusState("success");
        setStatusMessage("Message translated successfully.");
      } else {
        setStatusState("idle");
        setStatusMessage("No intelligible speech detected. Please speak clearly.");
        if (response && response.translated_text) {
          setTranslatedText(response.translated_text);
        }
      }
    } catch (err) {
      setStatusState("error");
      setErrorMessage(
        err.message || "Failed to transcribe speech. Check backend and Sarvam AI connection."
      );
    }
  };

  // Optional on-demand TTS playback in Your Language (Deaf user can listen if desired)
  const handlePlayTranslatedAudio = async () => {
    if (!translatedText) return;

    if (isPlayingAudio) {
      stopAudio();
      return;
    }

    setIsSynthesizingTTS(true);
    setErrorMessage(null);

    try {
      const res = await synthesizeSpeech(translatedText, yourLanguage, "priya");
      if (res && res.audio_base64) {
        const audioUrl = `data:audio/wav;base64,${res.audio_base64}`;
        const audio = new Audio(audioUrl);
        audioInstanceRef.current = audio;

        audio.onended = () => setIsPlayingAudio(false);
        audio.onerror = () => setIsPlayingAudio(false);

        setIsPlayingAudio(true);
        audio.play().catch(() => setIsPlayingAudio(false));
      } else {
        setErrorMessage("Audio synthesis returned empty audio data.");
      }
    } catch (err) {
      setErrorMessage(err.message || "Speech synthesis failed.");
    } finally {
      setIsSynthesizingTTS(false);
    }
  };

  const clearConversation = () => {
    stopAudio();
    stopRecordingCleanup();
    setIsRecording(false);
    setOriginalTranscript("");
    setTranslatedText("");
    setStatusState("idle");
    setStatusMessage("Ready for speech input");
    setErrorMessage(null);
  };

  return (
    <div
      className="communication-panel panel-other-voice"
      data-testid="panel-other-voice"
      aria-label="Direction B: Other Person's Voice to Readable Regional Text"
    >
      <header className="panel-header">
        <div className="panel-title-group">
          <span className="panel-direction-badge hearing-badge">
            Hearing Person → Deaf Person
          </span>
          <h2 className="panel-title">Other Person's Voice</h2>
        </div>
        <span className="panel-target-lang-pill" data-testid="panel-b-target-lang">
          Input: <strong>{otherLanguageName}</strong> &bull; Output: <strong>{yourLanguageName}</strong>
        </span>
      </header>

      {/* Microphone Controls Card */}
      <section className="mic-control-card" aria-label="Microphone Voice Capture">
        <div className="mic-action-wrap">
          <button
            type="button"
            className={`mic-toggle-button ${isRecording ? "recording" : "idle"}`}
            data-testid="mic-toggle-btn"
            onClick={toggleRecording}
            disabled={statusState === "processing"}
            aria-label={isRecording ? "Stop recording speech" : "Start speaking into microphone"}
          >
            {isRecording ? (
              <>
                <Square size={22} className="mic-icon pulse" />
                <span className="mic-btn-label">
                  Stop ({15 - recordingSeconds}s)
                </span>
              </>
            ) : statusState === "processing" ? (
              <>
                <Loader2 size={22} className="rotating" />
                <span className="mic-btn-label">Processing...</span>
              </>
            ) : (
              <>
                <Mic size={22} className="mic-icon" />
                <span className="mic-btn-label">Tap to Speak</span>
              </>
            )}
          </button>
        </div>

        {/* Recording Status & Waveform Indicator */}
        <div className="mic-status-row">
          <div
            className={`mic-status-pill ${statusState}`}
            data-testid="mic-status-badge"
            role="status"
            aria-live="polite"
          >
            {isRecording && <span className="mic-recording-pulse-dot" />}
            <span>{statusMessage}</span>
          </div>

          {isRecording && (
            <div className="recording-wave-visualizer" aria-hidden="true">
              <span className="wave-bar bar-1" />
              <span className="wave-bar bar-2" />
              <span className="wave-bar bar-3" />
              <span className="wave-bar bar-4" />
              <span className="wave-bar bar-5" />
            </div>
          )}
        </div>

        {errorMessage && (
          <div className="mic-error-banner" data-testid="mic-error-banner" role="alert">
            <AlertCircle size={18} />
            <span>{errorMessage}</span>
          </div>
        )}
      </section>

      {/* Prominent Translated Text Display (Designed specifically for Deaf reader) */}
      <section
        className="voice-communication-display"
        data-testid="voice-communication-display"
        aria-label="Translated Message Display for Deaf User"
      >
        <div className="transcript-section">
          <div className="section-label-row">
            <span className="section-sub-label">
              Hearing Person Spoke ({otherLanguageName}):
            </span>
          </div>
          <div
            className="original-speech-transcript"
            data-testid="original-speech-transcript"
          >
            {originalTranscript ? (
              <p className="transcript-text">"{originalTranscript}"</p>
            ) : (
              <p className="transcript-placeholder">
                Original speech transcript will appear here when the hearing person speaks...
              </p>
            )}
          </div>
        </div>

        {/* Prominent Large Translated Regional Language Text */}
        <div className="translation-prominent-section">
          <div className="section-label-row">
            <span className="section-main-label">
              Readable Translation in {yourLanguageName}:
            </span>
            <span className="regional-tag">{yourLanguageName}</span>
          </div>

          <div
            className={`translated-text-display-box ${translatedText ? "has-content" : "empty"}`}
            data-testid="translated-voice-text-box"
          >
            {statusState === "processing" ? (
              <div className="translating-spinner-wrap">
                <Loader2 size={24} className="rotating" />
                <span>Translating message for you...</span>
              </div>
            ) : translatedText ? (
              <h3 className="readable-regional-text" data-testid="readable-regional-text">
                {translatedText}
              </h3>
            ) : (
              <p className="readable-regional-placeholder">
                Translated regional text will appear here prominently for easy reading.
              </p>
            )}
          </div>
        </div>

        {/* Action Controls: Optional Audio Playback & Clear */}
        <div className="voice-actions-toolbar">
          <div className="voice-actions-left">
            <button
              type="button"
              className={`optional-audio-btn ${isPlayingAudio ? "playing" : ""}`}
              data-testid="optional-audio-btn"
              onClick={handlePlayTranslatedAudio}
              disabled={!translatedText || statusState === "processing" || isSynthesizingTTS}
              aria-label={
                isPlayingAudio
                  ? "Stop reading audio"
                  : `Listen to translation in ${yourLanguageName}`
              }
              title={`Optional speech synthesis in ${yourLanguageName}`}
            >
              {isSynthesizingTTS ? (
                <>
                  <Loader2 size={16} className="rotating" />
                  <span>Generating Audio...</span>
                </>
              ) : isPlayingAudio ? (
                <>
                  <Square size={16} />
                  <span>Stop Audio</span>
                </>
              ) : (
                <>
                  <Volume2 size={16} />
                  <span>Listen ({yourLanguageName})</span>
                </>
              )}
            </button>
          </div>

          <div className="voice-actions-right">
            <button
              type="button"
              className="clear-voice-btn"
              data-testid="clear-voice-btn"
              onClick={clearConversation}
              disabled={!originalTranscript && !translatedText}
              aria-label="Clear speech transcript"
              title="Clear current voice message"
            >
              <Trash2 size={14} />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
