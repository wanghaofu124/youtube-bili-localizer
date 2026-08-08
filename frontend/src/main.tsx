import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./illustrations.css";
import "./metadata-feedback.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode><App /></StrictMode>,
);
