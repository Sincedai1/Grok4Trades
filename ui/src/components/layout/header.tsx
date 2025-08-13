
import { useState, useEffect } from "react";
import { WalletButton } from "@/components/ui/wallet-button";
import { StatCard } from "@/components/ui/stat-card";
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  CreditCard, 
  BarChart3,
} from "lucide-react";

export default function Header() {
  const [balance, setBalance] = useState("0.00");
  const [pnl, setPnl] = useState("0.00");
  const [pnlPositive, setPnlPositive] = useState(true);
  const [tradeCount, setTradeCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = () => {
      setIsLoading(true);
      
      setTimeout(() => {
        setBalance("1,245.32");
        setPnl("78.24");
        setPnlPositive(true);
        setTradeCount(24);
        setIsLoading(false);
      }, 1800);
    };
    
    fetchData();
    
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="flex flex-col space-y-4 sm:space-y-0 sm:flex-row sm:justify-between sm:items-center py-6 px-6 md:px-8">
      <div className="flex-1">
        <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-solana via-solana-light to-solana-dark">
          Solana Mind Trader
        </h1>
        <p className="text-muted-foreground mt-1">GPT-4o Powered Trading Bot</p>
      </div>
      
      <div className="flex flex-wrap gap-3 flex-1 sm:flex-none sm:flex-row justify-start sm:justify-end items-center">
        <div className="grid grid-cols-3 gap-3 mr-2 w-full sm:w-auto">
          <StatCard
            variant="glass"
            size="sm"
            title="Balance"
            value={`$${balance}`}
            icon={<CreditCard className="h-4 w-4" />}
            className="w-full sm:w-[140px]"
          />
          
          <StatCard
            variant="glass"
            size="sm"
            title="PnL (24h)"
            value={`$${pnl}`}
            icon={
              pnlPositive ? (
                <ArrowUpRight className="h-4 w-4 text-green-500" />
              ) : (
                <ArrowDownRight className="h-4 w-4 text-red-500" />
              )
            }
            change={{ 
              value: "6.2%", 
              positive: pnlPositive 
            }}
            className="w-full sm:w-[140px]"
          />
          
          <StatCard
            variant="glass"
            size="sm"
            title="Trades"
            value={tradeCount}
            icon={<BarChart3 className="h-4 w-4" />}
            className="w-full sm:w-[120px]"
          />
        </div>
        
        <div className="flex space-x-2">
          <WalletButton />
        </div>
      </div>
    </header>
  );
}
