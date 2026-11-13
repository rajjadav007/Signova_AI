import React from "react";
import { useLocation } from "react-router-dom";
import { Menu, RefreshCw } from "lucide-react";
import { useBackendHealth } from "../../hooks/useBackendHealth";

const ROUTE_TITLES = {
  "/": "Dashboard",
  "/dashboard": "Dashboard",
  "/recognition": "Live Recognition",
  "/vocabulary": "Vocabulary",
  "/history": "Testing & History",
  "/settings": "Settings",
  "/about": "About",
};

export default function Navbar({
  isCollapsed,
  setIsCollapsed,
  isMobileOpen,
  setIsMobileOpen,
}) {
  const location = useLocation();
  const currentTitle = ROUTE_TITLES[location.pathname] || "Page";

  const { status: healthStatus, data: statusDetails, refetch: checkHealth } = useBackendHealth();
  const backendStatus =
    healthStatus === "connected"
      ? "healthy"
      : healthStatus === "disconnected"
      ? "offline"
      : "checking";

  const handleToggle = () => {
    if (window.innerWidth < 1024) {
      setIsMobileOpen(!isMobileOpen);
    } else {
      setIsCollapsed(!isCollapsed);
    }
  };

  return (
    <header className="navbar" role="banner">
      <div className="navbar-left">
        <button
          type="button"
          className="sidebar-toggle-btn"
          onClick={handleToggle}
          aria-label="Toggle navigation menu"
          title="Toggle sidebar"
        >
          <span className="mobile-toggle-icon">
            <Menu size={18} />
          </span>
        </button>

        <nav aria-label="Breadcrumb" className="navbar-breadcrumb">
          <span className="breadcrumb-root">Signova AI</span>
          <span className="breadcrumb-separator">/</span>
          <span className="breadcrumb-current">{currentTitle}</span>
        </nav>
      </div>

      <div className="navbar-right">
        {/* Real Backend Connection Status Indicator */}
        <div
          className={`status-badge ${backendStatus}`}
          title={
            backendStatus === "healthy"
              ? `Connected to API (${statusDetails?.application || "Signova Backend"} v${statusDetails?.version || "0.1.0"})`
              : backendStatus === "offline"
              ? "Backend API unreachable (port 8000)"
              : "Checking backend connection..."
          }
          data-testid="backend-status-badge"
        >
          <span className="status-dot" aria-hidden="true" />
          <span className="status-label">
            {backendStatus === "healthy"
              ? "API Connected"
              : backendStatus === "offline"
              ? "API Offline"
              : "Connecting..."}
          </span>
          {backendStatus === "offline" && (
            <button
              type="button"
              className="status-retry-btn"
              onClick={checkHealth}
              title="Retry backend health check"
              aria-label="Retry backend connection check"
            >
              <RefreshCw size={12} style={{ marginRight: "3px" }} />
              Retry
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
