
import { useEffect, useState } from "react";
import { useIsMobile } from "@/hooks/use-mobile";
import Header from "./header";
import Sidebar from "./sidebar";
import { Button } from "@/components/ui/button";
import { Menu, X, ArrowRightToLine, ArrowLeftToLine } from "lucide-react";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

interface MainLayoutProps {
  children: React.ReactNode;
}

export default function MainLayout({ children }: MainLayoutProps) {
  const isMobile = useIsMobile();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mounted, setMounted] = useState(false);

  // After first mount, we can safely show the interface
  useEffect(() => {
    setMounted(true);
    
    // Ensure dark mode is applied
    document.documentElement.classList.add('dark');
    document.documentElement.classList.remove('light', 'system');
  }, []);

  // If not mounted yet, return an empty div to avoid flickering
  if (!mounted) {
    return <div className="h-screen w-full bg-background"></div>;
  }

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden dark">
      {/* Mobile sidebar using Sheet */}
      {isMobile ? (
        <>
          <Sheet>
            <SheetTrigger asChild>
              <Button
                variant="outline"
                size="icon"
                className="absolute left-4 top-4 z-50 md:hidden"
              >
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="p-0">
              <Sidebar />
            </SheetContent>
          </Sheet>
        </>
      ) : (
        // Desktop sidebar
        <Sidebar
          className={cn(
            "transition-all duration-300 ease-in-out",
            sidebarCollapsed ? "w-16" : "w-64"
          )}
        />
      )}
      
      {/* Sidebar collapse button (desktop only) */}
      {!isMobile && (
        <Button
          variant="outline"
          size="icon"
          className="absolute left-[256px] bottom-4 z-50 hidden md:flex"
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        >
          {sidebarCollapsed ? (
            <ArrowRightToLine className="h-4 w-4" />
          ) : (
            <ArrowLeftToLine className="h-4 w-4" />
          )}
        </Button>
      )}
      
      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-y-auto px-4 md:px-6 py-4">
          {children}
        </main>
      </div>
    </div>
  );
}
