
import {
  LayoutDashboard,
  BarChart3,
  Settings,
  History,
  Rocket,
  Bot,
  BookOpen,
  HelpCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Link, useLocation } from "react-router-dom";

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className }: SidebarProps) {
  const location = useLocation();
  const currentPath = location.pathname;

  return (
    <div className={cn("h-screen w-64 border-r flex flex-col", className)}>
      <div className="flex-1 py-6 px-3 space-y-1">
        <div className="mb-6 px-3 flex items-center">
          <div className="h-8 w-8 rounded-full bg-gradient-to-r from-solana to-solana-light flex items-center justify-center text-white font-semibold mr-2">
            S
          </div>
          <span className="font-medium">Solana Mind Trader</span>
        </div>
        
        <nav className="space-y-1 px-2">
          <Link to="/">
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start text-muted-foreground hover:text-foreground", 
                currentPath === "/" && "bg-accent text-foreground"
              )}
            >
              <LayoutDashboard className="h-4 w-4 mr-3" />
              Dashboard
            </Button>
          </Link>
          
          <Link to="/analytics">
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start text-muted-foreground hover:text-foreground", 
                currentPath === "/analytics" && "bg-accent text-foreground"
              )}
            >
              <BarChart3 className="h-4 w-4 mr-3" />
              Analytics
            </Button>
          </Link>
          
          <Link to="/history">
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start text-muted-foreground hover:text-foreground", 
                currentPath === "/history" && "bg-accent text-foreground"
              )}
            >
              <History className="h-4 w-4 mr-3" />
              Trade History
            </Button>
          </Link>
          
          <Link to="/strategies">
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start text-muted-foreground hover:text-foreground", 
                currentPath === "/strategies" && "bg-accent text-foreground"
              )}
            >
              <Rocket className="h-4 w-4 mr-3" />
              AI Strategies
            </Button>
          </Link>
          
          <Link to="/settings">
            <Button
              variant="ghost"
              className={cn(
                "w-full justify-start text-muted-foreground hover:text-foreground", 
                currentPath === "/settings" && "bg-accent text-foreground"
              )}
            >
              <Settings className="h-4 w-4 mr-3" />
              Settings
            </Button>
          </Link>
        </nav>
      </div>
      
      <div className="border-t py-4 px-5 space-y-4">
        <div className="space-y-1">
          <h4 className="text-xs uppercase tracking-wider text-muted-foreground font-medium px-2">Resources</h4>
          <nav className="space-y-1">
            <Button variant="ghost" size="sm" className="w-full justify-start text-muted-foreground hover:text-foreground">
              <BookOpen className="h-4 w-4 mr-2" />
              Documentation
            </Button>
            <Button variant="ghost" size="sm" className="w-full justify-start text-muted-foreground hover:text-foreground">
              <HelpCircle className="h-4 w-4 mr-2" />
              Help & Support
            </Button>
          </nav>
        </div>
        
        <div className="px-2">
          <div className="relative overflow-hidden rounded-lg bg-gradient-to-r from-solana-dark to-solana p-3 shadow-md">
            <div className="space-y-2 relative z-10">
              <p className="text-white text-sm font-medium">
                AI-Enhanced Trading
              </p>
              <p className="text-white/80 text-xs">
                Boost your trades with GPT-4o AI predictions
              </p>
              <Button size="sm" variant="outline" className="bg-white/10 text-white border-white/20 hover:bg-white/20 w-full">
                <Bot className="h-3 w-3 mr-1" />
                Learn More
              </Button>
            </div>
            {/* Animated gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-r from-solana/50 to-solana-light/30 opacity-60 animate-pulse-slow"></div>
            {/* Animated dots pattern */}
            <div className="absolute inset-0 bg-[radial-gradient(circle,_rgba(255,255,255,0.1)_1px,_transparent_1px)] bg-[length:8px_8px] animate-[spin_120s_linear_infinite]"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
