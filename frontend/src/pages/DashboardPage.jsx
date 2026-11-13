import React from "react";
import PageContainer from "../components/common/PageContainer";
import WelcomeHero from "../components/dashboard/WelcomeHero";
import SystemOverviewCards from "../components/dashboard/SystemOverviewCards";
import ModelConfigCard from "../components/dashboard/ModelConfigCard";
import VocabularyPreview from "../components/dashboard/VocabularyPreview";
import PipelineFlow from "../components/dashboard/PipelineFlow";
import RecognitionOverview from "../components/dashboard/RecognitionOverview";

export default function DashboardPage() {
  return (
    <PageContainer>
      <div className="dashboard-container">
        {/* 1. Welcome Hero Section */}
        <WelcomeHero />

        {/* 2. System Overview Cards */}
        <SystemOverviewCards />

        {/* 3. Model Architecture & Vocabulary Preview Grid */}
        <div className="dashboard-columns">
          <ModelConfigCard />
          <VocabularyPreview />
        </div>

        {/* 4. Visual Inference Pipeline */}
        <PipelineFlow />

        {/* 5. Recognition Engine Overview */}
        <RecognitionOverview />
      </div>
    </PageContainer>
  );
}
