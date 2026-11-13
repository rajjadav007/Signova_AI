import React from "react";
import { Link } from "react-router-dom";
import { BookOpen, ArrowRight } from "lucide-react";

export const SUPPORTED_SIGNS = [
  { id: 1, name: "Hello", category: "Greeting" },
  { id: 2, name: "Thank you", category: "Courtesy" },
  { id: 3, name: "Sorry", category: "Courtesy" },
  { id: 4, name: "Please", category: "Courtesy" },
  { id: 5, name: "Yes", category: "Response" },
  { id: 6, name: "No", category: "Response" },
  { id: 7, name: "Help", category: "Emergency" },
  { id: 8, name: "Stop", category: "Command" },
  { id: 9, name: "Water", category: "Need" },
  { id: 10, name: "Food", category: "Need" },
];

export default function VocabularyPreview() {
  return (
    <article className="dashboard-card" aria-labelledby="vocab-preview-title">
      <div className="dashboard-card-header">
        <div className="dashboard-card-title-wrap">
          <div className="overview-card-icon" aria-hidden="true">
            <BookOpen size={18} />
          </div>
          <h2 id="vocab-preview-title" className="dashboard-card-title">
            Supported Vocabulary
          </h2>
        </div>
        <Link to="/vocabulary" className="vocab-preview-header-link" aria-label="View All Signs in Vocabulary">
          <span>View All Signs</span>
          <ArrowRight size={14} />
        </Link>
      </div>

      <div className="vocab-grid">
        {SUPPORTED_SIGNS.map((sign) => (
          <div key={sign.id} className="vocab-chip" data-testid={`vocab-chip-${sign.id}`}>
            <div className="vocab-chip-left">
              <span className="vocab-index">{sign.id}</span>
              <span className="vocab-name">{sign.name}</span>
            </div>
            <span className="vocab-category-tag">{sign.category}</span>
          </div>
        ))}
      </div>

      <div className="vocab-footer-bar">
        <span>10 curated classes trained on bilateral ISL datasets</span>
        <Link to="/vocabulary" className="vocab-preview-header-link">
          Browse full list &rarr;
        </Link>
      </div>
    </article>
  );
}
