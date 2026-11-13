import React from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Video,
  BookOpen,
  History,
  Settings,
  Info,
  Sparkles,
  ChevronLeft,
  ChevronRight,
  X,
} from "lucide-react";

export const NAV_ITEMS = [
  {
    path: "/dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
  },
  {
    path: "/recognition",
    label: "Live Recognition",
    icon: Video,
  },
  {
    path: "/vocabulary",
    label: "Vocabulary",
    icon: BookOpen,
  },
  {
    path: "/history",
    label: "Testing & History",
    icon: History,
  },
  {
    path: "/settings",
    label: "Settings",
    icon: Settings,
  },
  {
    path: "/about",
    label: "About",
    icon: Info,
  },
];

export default function Sidebar({
  isCollapsed,
  setIsCollapsed,
  isMobileOpen,
  setIsMobileOpen,
}) {
  const handleNavClick = () => {
    if (isMobileOpen) {
      setIsMobileOpen(false);
    }
  };

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isMobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setIsMobileOpen(false)}
          aria-label="Close mobile navigation"
        />
      )}

      <aside
        className={`sidebar ${isCollapsed ? "collapsed" : ""} ${
          isMobileOpen ? "mobile-open" : ""
        }`}
        aria-label="Primary Navigation"
      >
        {/* Sidebar Header / Branding */}
        <div className="sidebar-header">
          <NavLink
            to="/dashboard"
            className="sidebar-brand"
            onClick={handleNavClick}
            title="Signova AI Home"
          >
            <div className="brand-icon-wrapper">
              <Sparkles size={20} />
            </div>
            <div className="brand-text">
              <span className="brand-title">Signova AI</span>
              <span className="brand-subtitle">ISL Platform</span>
            </div>
          </NavLink>

          {/* Mobile close button */}
          {isMobileOpen && (
            <button
              type="button"
              className="sidebar-toggle-btn mobile-close-btn"
              onClick={() => setIsMobileOpen(false)}
              aria-label="Close navigation drawer"
            >
              <X size={18} />
            </button>
          )}
        </div>

        {/* Navigation Items */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={handleNavClick}
                className={({ isActive }) =>
                  `nav-item ${isActive ? "active" : ""}`
                }
                title={isCollapsed ? item.label : undefined}
                end={item.path === "/dashboard"}
              >
                <span className="nav-icon">
                  <Icon size={19} />
                </span>
                <span className="nav-label">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Sidebar Footer with Desktop Collapse Toggle */}
        <div className="sidebar-footer">
          <span>v0.1.0</span>
          <button
            type="button"
            className="sidebar-toggle-btn collapse-toggle-btn"
            onClick={() => setIsCollapsed(!isCollapsed)}
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            style={{ width: "28px", height: "28px" }}
          >
            {isCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>
      </aside>
    </>
  );
}
