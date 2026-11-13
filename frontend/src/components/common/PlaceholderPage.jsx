import React from "react";
import PageContainer from "./PageContainer";

export default function PlaceholderPage({
  title,
  description,
  icon: Icon,
  tag = "Coming in upcoming releases",
}) {
  return (
    <PageContainer>
      <div className="placeholder-card" data-testid="placeholder-card">
        {Icon && (
          <div className="placeholder-icon-wrap" aria-hidden="true">
            <Icon size={28} />
          </div>
        )}
        <h2 className="placeholder-title">{title}</h2>
        <p className="placeholder-desc">{description}</p>
        <span className="placeholder-tag">{tag}</span>
      </div>
    </PageContainer>
  );
}
