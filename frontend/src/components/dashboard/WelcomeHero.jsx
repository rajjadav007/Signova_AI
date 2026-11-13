import React from "react";
import { Link } from "react-router-dom";
import { Video, BookOpen, Sparkles, ArrowRight } from "lucide-react";

export default function WelcomeHero() {
  return (
    <section className="dashboard-hero" aria-labelledby="welcome-heading">
      <div className="hero-tag">
        <Sparkles size={14} />
        <span>Indian Sign Language AI Platform</span>
      </div>

      <h1 id="welcome-heading" className="hero-title">
        Welcome to Signova AI
      </h1>

      <p className="hero-description">
        Breaking communication barriers through Indian Sign Language recognition.
      </p>

      <p className="hero-intro">
        Signova AI translates Indian Sign Language (ISL) gestures into real-time text and speech using deep learning and MediaPipe Holistic pose landmarks.
      </p>

      <div className="hero-actions">
        <Link to="/recognition" className="btn-primary" aria-label="Start Live Recognition">
          <Video size={18} />
          <span>Start Recognition</span>
          <ArrowRight size={16} />
        </Link>

        <Link to="/vocabulary" className="btn-secondary" aria-label="Explore Vocabulary">
          <BookOpen size={18} />
          <span>Explore Vocabulary</span>
        </Link>
      </div>
    </section>
  );
}
