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
  const [recognitionMode, setRecognitionMode] = useState("isolated"); // "isolated" (10 Signs) | "sentence" (5 Sentences)
  const [sentenceModel, setSentenceModel] = useState("bigru"); // "bigru" (BiGRU Attention) | "dtw" (DTW-KNN)

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
            {recognitionMode === "sentence" && (
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
                &bull; Five-Class ISL Recognition ({sentenceModel === "bigru" ? "BiGRU Attention" : "DTW-KNN"})
              </span>
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
                aria-selected={recognitionMode === "isolated"}
                className={`mode-toggle-btn ${recognitionMode === "isolated" ? "active" : ""}`}
                data-testid="mode-isolated-btn"
                onClick={() => setRecognitionMode("isolated")}
              >
                <span className="mode-title">Isolated Signs</span>
                <span className="mode-badge">10 Signs</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={recognitionMode === "sentence"}
                className={`mode-toggle-btn ${recognitionMode === "sentence" ? "active" : ""}`}
                data-testid="mode-sentence-btn"
                onClick={() => setRecognitionMode("sentence")}
              >
                <span className="mode-title">Continuous Sentences</span>
                <span className="mode-badge">5 Sentences</span>
              </button>
            </div>
          </div>
        </section>

        {/* Five-Class Model Architecture Selection Bar (Sentence Mode) */}
        {recognitionMode === "sentence" && (
          <section
            className="recognition-mode-bar model-architecture-bar"
            data-testid="sentence-model-selector"
            aria-label="Five-Class Model Architecture Selection"
          >
            <div className="recognition-mode-container">
              <span className="mode-switcher-label">Five-Class Recognition Model:</span>
              <div className="mode-toggle-group" role="tablist" aria-label="Five-Class Recognition Architecture">
                <button
                  type="button"
                  role="tab"
                  aria-selected={sentenceModel === "bigru"}
                  className={`mode-toggle-btn ${sentenceModel === "bigru" ? "active" : ""}`}
                  data-testid="model-select-bigru"
                  onClick={() => setSentenceModel("bigru")}
                >
                  <span className="mode-title">BiGRU Attention</span>
                  <span className="mode-badge">Neural</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={sentenceModel === "dtw"}
                  className={`mode-toggle-btn ${sentenceModel === "dtw" ? "active" : ""}`}
                  data-testid="model-select-dtw"
                  onClick={() => setSentenceModel("dtw")}
                >
                  <span className="mode-title">DTW-KNN</span>
                  <span className="mode-badge">Alignment</span>
                </button>
              </div>
            </div>
          </section>
        )}

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
            onModeChange={setRecognitionMode}
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
