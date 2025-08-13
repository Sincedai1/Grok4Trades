
import MainLayout from "@/components/layout/main-layout";
import TradingChart from "@/components/dashboard/trading-chart";
import TradingControls from "@/components/dashboard/trading-controls";
import ActiveTrades from "@/components/dashboard/active-trades";
import SystemStatus from "@/components/dashboard/system-status";
import MarketSentiment from "@/components/dashboard/market-sentiment";
import PerformanceMetrics from "@/components/dashboard/performance-metrics";
import AITradingMenu from "@/components/dashboard/ai-trading-menu";
import { useTheme } from "@/hooks/use-theme";
import { useEffect, useState } from "react";
import { useWallet } from "@solana/wallet-adapter-react";
import { toast } from "sonner";

const Index = () => {
  const { theme } = useTheme();
  const { connected, connecting, publicKey } = useWallet();
  const [isInitialLoad, setIsInitialLoad] = useState(true);
  
  // Show welcome toast on wallet connection
  useEffect(() => {
    if (isInitialLoad && connected && publicKey) {
      setIsInitialLoad(false);
      toast.success("Wallet Connected", {
        description: `Welcome back, ${publicKey.toString().slice(0, 4)}...${publicKey.toString().slice(-4)}`,
      });
    }
  }, [connected, publicKey, isInitialLoad]);

  return (
    <MainLayout>
      <div className="grid grid-cols-1 md:grid-cols-12 gap-4 md:gap-6">
        {/* First row - AI Trading Menu and Trading Chart */}
        <div className="md:col-span-4">
          <AITradingMenu />
        </div>
        <div className="md:col-span-8">
          <TradingChart symbol="SOL/USDC" />
        </div>
        
        {/* Second row */}
        <div className="md:col-span-4">
          <TradingControls />
        </div>
        <div className="md:col-span-8">
          <ActiveTrades />
        </div>
        
        {/* Third row */}
        <div className="md:col-span-6">
          <PerformanceMetrics />
        </div>
        <div className="md:col-span-6">
          <MarketSentiment />
        </div>
        
        {/* Fourth row */}
        <div className="md:col-span-12">
          <SystemStatus />
        </div>
      </div>
    </MainLayout>
  );
};

export default Index;
