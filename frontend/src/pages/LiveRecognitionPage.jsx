import React, { useState, useCallback } from "react";
import PageContainer from "../components/common/PageContainer";
import LanguageSelectorBar, {
  COMMUNICATION_LANGUAGES,
} from "../components/communication/LanguageSelectorBar";
import PanelYourSign from "../components/communication/PanelYourSign";
import PanelOtherVoice from "../components/communication/PanelOtherVoice";
import "../styles/communication.css";

export default function LiveRecognitionPage() {
  const [yourLanguage, setYourLanguage] = useState("gu-IN"); // Deaf user readable language (initial default Gujarati)
  const [otherLanguage, setOtherLanguage] = useState("en-IN"); // Hearing user language (initial default English)
  const [activeRecognitionMode, setActiveRecognitionMode] = useState("word"); // "word" | "continuous"
  const [recognitionMode, setRecognitionMode] = useState("isolated"); // "isolated" (10 Signs) | "sentence" (5 Sentences)
  const [sentenceModel, setSentenceModel] = useState("bigru"); // "bigru" (BiGRU Attention) | "dtw" (DTW-KNN)

  const isContinuousMode =
    activeRecognitionMode === "continuous" ||
    recognitionMode === "sentence" ||
    recognitionMode === "sentence_25";
  const isWordMode = !isContinuousMode;

  const handleModeChange = useCallback((mode) => {
    if (mode === "word" || mode === "isolated") {
      setActiveRecognitionMode("word");
      setRecognitionMode("isolated");
    } else if (mode === "continuous" || mode === "sentence") {
      setActiveRecognitionMode("continuous");
      setRecognitionMode("sentence");
    } else if (mode === "sentence_25") {
      setActiveRecognitionMode("continuous");
      setRecognitionMode("sentence_25");
    } else {
      setActiveRecognitionMode(mode);
      setRecognitionMode(mode);
    }
  }, []);

  const handleSwapLanguages = useCallback(() => {
    setYourLanguage((prevYour) => {
      const prevOther = otherLanguage;
      setOtherLanguage(prevYour);
      return prevOther;
    });
  }, [otherLanguage]);

  const yourLangName =
    COMMUNICATION_LANGUAGES.find((l) => l.code === yourLanguage)?.name || "Gujarati";
  const otherLangName =
    COMMUNICATION_LANGUAGES.find((l) => l.code === otherLanguage)?.name || "English";

  return (
    <PageContainer>
      <div className="two-way-assistant-container" data-testid="two-way-assistant-page">
        {/* Main Header */}
        <header className="assistant-page-header">
          <h1 className="assistant-main-title" data-testid="assistant-main-title">
            Signova AI &bull; Two-Way Communication Assistant
            {isContinuousMode && (
              <>
                <span
                  className="active-model-title-tag"
                  data-testid="heading-active-model"
                  style={{
                    fontSize: "1rem",
                    fontWeight: "600",
                    marginLeft: "10px",
                    color: "var(--color-primary)",
                  }}
                >
                  &bull; Continuous ISL Recognition (25 Sentences &bull; BiGRU Attention)
                </span>
                <span
                  data-testid="heading-active-model-25"
                  style={{ display: "none" }}
                >
                  &bull; Twenty-Five-Class ISL Recognition (BiGRU Attention)
                </span>
              </>
            )}
          </h1>
          <p className="assistant-subtitle">
            Unified real-time communication between Deaf individuals (Indian Sign Language) and
            hearing individuals (regional speech and audio).
          </p>
          <span style={{ display: "none" }}>Signova AI - Indian Sign Language Recognition</span>
        </header>

        {/* Language Settings Bar */}
        <LanguageSelectorBar
          yourLanguage={yourLanguage}
          otherLanguage={otherLanguage}
          onYourLanguageChange={setYourLanguage}
          onOtherLanguageChange={setOtherLanguage}
          onSwapLanguages={handleSwapLanguages}
        />

        {/* Mode Switcher Bar */}
        <section
          className="recognition-mode-bar"
          data-testid="recognition-mode-switcher"
          aria-label="ISL Recognition Mode Selection"
        >
          <div className="recognition-mode-container">
            <span className="mode-switcher-label">ISL Recognition Mode:</span>
            <div className="mode-toggle-group" role="tablist" aria-label="Recognition Mode">
              <button
                type="button"
                role="tab"
                aria-selected={isWordMode}
                className={`mode-toggle-btn ${isWordMode ? "active" : ""}`}
                data-testid="mode-isolated-btn"
                onClick={() => handleModeChange("word")}
              >
                <span className="mode-title">Word Recognition (Isolated Signs)</span>
                <span className="mode-badge">10 Signs</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={isContinuousMode}
                className={`mode-toggle-btn ${isContinuousMode ? "active" : ""}`}
                data-testid="mode-sentence-btn"
                onClick={() => handleModeChange("continuous")}
              >
                <span className="mode-title">Continuous Sentence Recognition (Continuous Sentences)</span>
                <span className="mode-badge">25 Sentences (101 Sentences)</span>
              </button>
            </div>
          </div>
        </section>

        {/* Active Recognition Mode Banner */}
        <div
          className="active-mode-status-banner"
          data-testid="active-mode-status"
          style={{
            margin: "0 0 16px 0",
            padding: "8px 16px",
            borderRadius: "var(--radius-md, 8px)",
            backgroundColor: "rgba(15, 118, 110, 0.08)",
            border: "1px solid rgba(15, 118, 110, 0.2)",
            color: "var(--color-primary, #0f766e)",
            fontWeight: "600",
            fontSize: "0.9rem",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          {isWordMode
            ? "Word Recognition: Individual sign detection active."
            : "Continuous Sentence Recognition: Full sentence detection active."}
        </div>

        {/* Communication Area: Balanced Two-Column Desktop / Stacked Mobile Layout */}
        <main
          className="two-way-panels-grid"
          data-testid="two-way-panels-grid"
          aria-label="Two-Way Communication Channels"
        >
          {/* Panel A: Deaf User -> Hearing Person */}
          <PanelYourSign
            otherLanguage={otherLanguage}
            otherLanguageName={otherLangName}
            recognitionMode={recognitionMode}
            activeRecognitionMode={activeRecognitionMode}
            onModeChange={handleModeChange}
            sentenceModel={sentenceModel}
            onSentenceModelChange={setSentenceModel}
          />

          {/* Panel B: Hearing Person -> Deaf User */}
          <PanelOtherVoice
            yourLanguage={yourLanguage}
            yourLanguageName={yourLangName}
            otherLanguage={otherLanguage}
            otherLanguageName={otherLangName}
          />
        </main>

        <footer
          style={{
            marginTop: "20px",
            borderTop: "1px solid var(--color-border)",
            paddingTop: "14px",
            textAlign: "center",
            fontSize: "0.82rem",
            color: "var(--color-text-muted)",
          }}
        >
          Signova AI &bull; Two-Way Communication Assistant &bull; MediaPipe Holistic & 10-Class
          Bi-GRU &bull; Sarvam AI (Saaras STT, Sarvam Translate, Bulbul TTS)
        </footer>
      </div>
    </PageContainer>
  );
}
