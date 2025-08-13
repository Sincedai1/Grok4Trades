
import { createContext, useContext, useEffect } from "react";

type ThemeProviderProps = {
  children: React.ReactNode;
};

type ThemeProviderState = {
  theme: "dark";
};

const initialState: ThemeProviderState = {
  theme: "dark",
};

const ThemeProviderContext = createContext<ThemeProviderState>(initialState);

export function ThemeProvider({
  children,
  ...props
}: ThemeProviderProps) {
  // Apply dark mode to document - run this effect on every render to ensure it's always applied
  useEffect(() => {
    // Force dark mode on the root element
    document.documentElement.classList.remove("light", "system");
    document.documentElement.classList.add("dark");
    
    // Force dark mode in localStorage
    localStorage.setItem("theme", "dark");
    
    // Prevent any attempts to change theme by adding an observer
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.attributeName === 'class') {
          const element = mutation.target as HTMLElement;
          if (!element.classList.contains('dark')) {
            element.classList.add('dark');
          }
        }
      });
    });
    
    observer.observe(document.documentElement, { attributes: true });
    
    return () => observer.disconnect();
  }, []);

  return (
    <ThemeProviderContext.Provider {...props} value={initialState}>
      {children}
    </ThemeProviderContext.Provider>
  );
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext);

  if (context === undefined)
    throw new Error("useTheme must be used within a ThemeProvider");

  return context;
};
