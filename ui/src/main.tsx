
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";
import { ThemeProvider } from "./hooks/use-theme";
import SolanaWalletProvider from "./contexts/SolanaWalletProvider";

// Add Buffer polyfill for Solana wallet adapter
import { Buffer } from 'buffer';
window.Buffer = Buffer;

// Force dark mode immediately when script loads
document.documentElement.classList.add('dark');
document.documentElement.classList.remove('light', 'system');
localStorage.setItem("theme", "dark");

// Additional dark mode enforcement with inline script
const darkModeScript = document.createElement('script');
darkModeScript.textContent = `
  (function() {
    document.documentElement.classList.add('dark');
    document.documentElement.classList.remove('light', 'system');
    localStorage.setItem("theme", "dark");
  })();
`;
document.head.appendChild(darkModeScript);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <SolanaWalletProvider>
          <App />
        </SolanaWalletProvider>
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);
