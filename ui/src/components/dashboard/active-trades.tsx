
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { 
  ChevronUp, 
  ChevronDown, 
  AlertCircle, 
  XCircle,
  Target,
  BarChart3
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { toast } from "sonner";

interface Trade {
  id: string;
  symbol: string;
  type: "Buy" | "Sell";
  status: "Open" | "Closed" | "Pending";
  entry: number;
  current: number;
  target: number;
  size: number;
  pnl: number;
  timestamp: string;
  aiConfidence: number; // 0-100
  leverage?: number;
  riskScore?: number;
}

export default function ActiveTrades() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
  const [showTradeDetails, setShowTradeDetails] = useState(false);

  // Simulated data fetch
  useEffect(() => {
    const fetchTrades = () => {
      setIsLoading(true);
      
      // Simulate delay
      setTimeout(() => {
        const mockTrades: Trade[] = [
          {
            id: "t-1",
            symbol: "SOL/USDC",
            type: "Buy",
            status: "Open",
            entry: 39.45,
            current: 41.20,
            target: 44.50,
            size: 5.2,
            pnl: 9.1,
            timestamp: "10:24:15",
            aiConfidence: 87,
            leverage: 1.5,
            riskScore: 38
          },
          {
            id: "t-2",
            symbol: "JTO/USDC",
            type: "Buy",
            status: "Open",
            entry: 2.45,
            current: 2.38,
            target: 2.65,
            size: 150,
            pnl: -2.8,
            timestamp: "09:45:32",
            aiConfidence: 62,
            leverage: 1,
            riskScore: 45
          },
          {
            id: "t-3",
            symbol: "BONK/USDC",
            type: "Buy",
            status: "Open",
            entry: 0.00002457,
            current: 0.00002532,
            target: 0.00002750,
            size: 25000000,
            pnl: 3.05,
            timestamp: "08:32:01",
            aiConfidence: 73,
            leverage: 1,
            riskScore: 42
          },
          {
            id: "t-4",
            symbol: "WIF/USDC",
            type: "Sell",
            status: "Pending",
            entry: 0.001436,
            current: 0.001436,
            target: 0.001200,
            size: 2000,
            pnl: 0,
            timestamp: "10:29:46",
            aiConfidence: 91,
            leverage: 2,
            riskScore: 61
          }
        ];
        
        setTrades(mockTrades);
        setIsLoading(false);
      }, 1700);
    };
    
    fetchTrades();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchTrades, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleViewDetails = (trade: Trade) => {
    setSelectedTrade(trade);
    setShowTradeDetails(true);
  };

  const handleCloseTrade = (tradeId: string) => {
    // In a real implementation, this would call a trading API
    setTrades(prevTrades => 
      prevTrades.map(trade => 
        trade.id === tradeId 
          ? { ...trade, status: "Closed" as const } 
          : trade
      )
    );
    
    toast.success("Trade closed successfully", {
      description: "Your position has been closed",
    });
    
    // Close dialog if the closed trade is the selected one
    if (selectedTrade?.id === tradeId) {
      setShowTradeDetails(false);
    }
  };

  const handleUpdateTarget = (tradeId: string, newTarget: number) => {
    // In a real implementation, this would call a trading API
    setTrades(prevTrades => 
      prevTrades.map(trade => 
        trade.id === tradeId 
          ? { ...trade, target: newTarget } 
          : trade
      )
    );
    
    toast.success("Target price updated", {
      description: `New target: $${newTarget.toFixed(
        newTarget < 0.01 ? 8 : newTarget < 1 ? 5 : 2
      )}`,
    });
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg font-medium">Active Trades</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[340px] pr-4">
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="flex flex-col space-y-2">
                  <div className="flex justify-between">
                    <Skeleton className="h-5 w-24" />
                    <Skeleton className="h-5 w-16" />
                  </div>
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-4 w-12" />
                  </div>
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-28" />
                    <Skeleton className="h-4 w-14" />
                  </div>
                  <div className="h-px bg-border my-2" />
                </div>
              ))}
            </div>
          ) : trades.length > 0 ? (
            <div className="space-y-4">
              {trades.map((trade) => (
                <div key={trade.id} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center">
                      <span className="font-medium">{trade.symbol}</span>
                      <Badge 
                        className={`ml-2 ${
                          trade.status === "Open" 
                            ? trade.type === "Buy" 
                              ? "bg-green-500/10 text-green-500 border-green-500/20" 
                              : "bg-red-500/10 text-red-500 border-red-500/20"
                            : trade.status === "Pending"
                            ? "bg-amber-500/10 text-amber-500 border-amber-500/20"
                            : "bg-gray-500/10 text-gray-500 border-gray-500/20"
                        }`}
                        variant="outline"
                      >
                        {trade.type} • {trade.status}
                      </Badge>
                    </div>
                    <div className="flex items-center">
                      {trade.pnl !== 0 && (
                        trade.pnl > 0 
                          ? <ChevronUp className="h-3 w-3 text-green-500 mr-1" /> 
                          : <ChevronDown className="h-3 w-3 text-red-500 mr-1" />
                      )}
                      <span className={`font-medium ${
                        trade.pnl > 0 
                          ? "text-green-500" 
                          : trade.pnl < 0 
                          ? "text-red-500" 
                          : ""
                      }`}>
                        {trade.pnl > 0 ? "+" : ""}{trade.pnl}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Entry: ${trade.entry.toFixed(
                      trade.entry < 0.01 ? 8 : trade.entry < 1 ? 5 : 2
                    )}</span>
                    <span className="text-muted-foreground">Current: ${trade.current.toFixed(
                      trade.current < 0.01 ? 8 : trade.current < 1 ? 5 : 2
                    )}</span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">
                      Size: {trade.size.toLocaleString(undefined, {
                        minimumFractionDigits: 0,
                        maximumFractionDigits: trade.size < 1 ? 5 : 2
                      })}
                    </span>
                    <div className="flex items-center">
                      <Target className="h-3 w-3 mr-1 text-muted-foreground" />
                      <span className="text-muted-foreground">
                        Target: ${trade.target.toFixed(
                          trade.target < 0.01 ? 8 : trade.target < 1 ? 5 : 2
                        )}
                      </span>
                    </div>
                  </div>

                  <div className="flex justify-between text-sm mt-2">
                    <div className="flex items-center">
                      <span className="text-xs mr-1 text-muted-foreground">AI:</span>
                      <Badge 
                        variant="outline" 
                        className={`text-xs ${
                          trade.aiConfidence > 80 
                            ? "bg-green-500/10 text-green-500 border-green-500/20" 
                            : trade.aiConfidence > 60 
                            ? "bg-amber-500/10 text-amber-500 border-amber-500/20" 
                            : "bg-red-500/10 text-red-500 border-red-500/20"
                        }`}
                      >
                        {trade.aiConfidence}%
                      </Badge>
                    </div>
                    <div className="flex space-x-2">
                      <Button 
                        variant="outline" 
                        size="sm" 
                        className="h-7 px-2"
                        onClick={() => handleViewDetails(trade)}
                      >
                        <BarChart3 className="h-3 w-3 mr-1" />
                        Details
                      </Button>
                      {trade.status === "Open" && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          className="h-7 px-2 border-red-500/20 text-red-500 hover:bg-red-500/10 hover:text-red-500"
                          onClick={() => handleCloseTrade(trade.id)}
                        >
                          <XCircle className="h-3 w-3 mr-1" />
                          Close
                        </Button>
                      )}
                    </div>
                  </div>
                  <div className="h-px bg-border my-2" />
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center py-10">
              <AlertCircle className="text-muted-foreground h-10 w-10 mb-2" />
              <h3 className="font-medium">No Active Trades</h3>
              <p className="text-sm text-muted-foreground mt-1">
                Start the trading bot to begin executing trades
              </p>
            </div>
          )}
        </ScrollArea>

        {/* Trade Details Dialog */}
        <Dialog open={showTradeDetails} onOpenChange={setShowTradeDetails}>
          {selectedTrade && (
            <DialogContent className="sm:max-w-[500px]">
              <DialogHeader>
                <DialogTitle className="flex items-center">
                  {selectedTrade.symbol} Trade Details
                  <Badge 
                    className={`ml-2 ${
                      selectedTrade.status === "Open" 
                        ? selectedTrade.type === "Buy" 
                          ? "bg-green-500/10 text-green-500 border-green-500/20" 
                          : "bg-red-500/10 text-red-500 border-red-500/20"
                        : selectedTrade.status === "Pending"
                        ? "bg-amber-500/10 text-amber-500 border-amber-500/20"
                        : "bg-gray-500/10 text-gray-500 border-gray-500/20"
                    }`}
                    variant="outline"
                  >
                    {selectedTrade.type} • {selectedTrade.status}
                  </Badge>
                </DialogTitle>
                <DialogDescription>
                  Opened at {selectedTrade.timestamp}
                </DialogDescription>
              </DialogHeader>
              
              <div className="grid gap-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium mb-1">Entry Price</p>
                    <p className="text-lg">${selectedTrade.entry.toFixed(
                      selectedTrade.entry < 0.01 ? 8 : selectedTrade.entry < 1 ? 5 : 2
                    )}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-1">Current Price</p>
                    <p className="text-lg">${selectedTrade.current.toFixed(
                      selectedTrade.current < 0.01 ? 8 : selectedTrade.current < 1 ? 5 : 2
                    )}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium mb-1">Position Size</p>
                    <p className="text-lg">{selectedTrade.size.toLocaleString(undefined, {
                      minimumFractionDigits: 0,
                      maximumFractionDigits: selectedTrade.size < 1 ? 5 : 2
                    })}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-1">Target Price</p>
                    <p className="text-lg">${selectedTrade.target.toFixed(
                      selectedTrade.target < 0.01 ? 8 : selectedTrade.target < 1 ? 5 : 2
                    )}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium mb-1">P&L</p>
                    <p className={`text-lg ${
                      selectedTrade.pnl > 0 
                        ? "text-green-500" 
                        : selectedTrade.pnl < 0 
                        ? "text-red-500" 
                        : ""
                    }`}>
                      {selectedTrade.pnl > 0 ? "+" : ""}{selectedTrade.pnl}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm font-medium mb-1">AI Confidence</p>
                    <Badge 
                      variant="outline" 
                      className={`${
                        selectedTrade.aiConfidence > 80 
                          ? "bg-green-500/10 text-green-500 border-green-500/20" 
                          : selectedTrade.aiConfidence > 60 
                          ? "bg-amber-500/10 text-amber-500 border-amber-500/20" 
                          : "bg-red-500/10 text-red-500 border-red-500/20"
                      }`}
                    >
                      {selectedTrade.aiConfidence}%
                    </Badge>
                  </div>
                </div>
                
                {(selectedTrade.leverage || selectedTrade.riskScore) && (
                  <div className="grid grid-cols-2 gap-4">
                    {selectedTrade.leverage && (
                      <div>
                        <p className="text-sm font-medium mb-1">Leverage</p>
                        <p className="text-lg">{selectedTrade.leverage}x</p>
                      </div>
                    )}
                    {selectedTrade.riskScore && (
                      <div>
                        <p className="text-sm font-medium mb-1">Risk Score</p>
                        <Badge 
                          variant="outline" 
                          className={`${
                            selectedTrade.riskScore < 30 
                              ? "bg-green-500/10 text-green-500 border-green-500/20" 
                              : selectedTrade.riskScore < 60 
                              ? "bg-amber-500/10 text-amber-500 border-amber-500/20" 
                              : "bg-red-500/10 text-red-500 border-red-500/20"
                          }`}
                        >
                          {selectedTrade.riskScore}/100
                        </Badge>
                      </div>
                    )}
                  </div>
                )}
              </div>
              
              <DialogFooter className="flex">
                {selectedTrade.status === "Open" ? (
                  <>
                    <Button 
                      variant="outline" 
                      className="flex-1 border-amber-500/20 text-amber-500 hover:bg-amber-500/10 hover:text-amber-500"
                      onClick={() => {
                        // In a real app, this would open a modal to adjust the target price
                        const newTarget = Number((selectedTrade.target * 1.05).toFixed(
                          selectedTrade.target < 0.01 ? 8 : selectedTrade.target < 1 ? 5 : 2
                        ));
                        handleUpdateTarget(selectedTrade.id, newTarget);
                      }}
                    >
                      <Target className="h-4 w-4 mr-2" />
                      Adjust Target
                    </Button>
                    <Button 
                      variant="destructive" 
                      className="flex-1 ml-2"
                      onClick={() => handleCloseTrade(selectedTrade.id)}
                    >
                      <XCircle className="h-4 w-4 mr-2" />
                      Close Position
                    </Button>
                  </>
                ) : (
                  <Button 
                    variant="outline" 
                    className="w-full"
                    onClick={() => setShowTradeDetails(false)}
                  >
                    Close
                  </Button>
                )}
              </DialogFooter>
            </DialogContent>
          )}
        </Dialog>
      </CardContent>
    </Card>
  );
}
