
import { useState } from "react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Brain, Rocket, Target, Shield, Coins, ArrowUpRight } from "lucide-react";
import { TradeModal } from "./trade-modal";
import { useWallet } from "@solana/wallet-adapter-react";

export default function AITradingMenu() {
  const { connected } = useWallet();
  const [isTradeModalOpen, setIsTradeModalOpen] = useState(false);
  
  return (
    <>
      <Card className="h-full">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-medium">AI Trading</CardTitle>
            <Badge variant="outline" className="bg-solana/10 text-solana border-solana/20">
              Pro
            </Badge>
          </div>
          <CardDescription>AI-powered trading strategies</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Tabs defaultValue="features">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="features">Features</TabsTrigger>
              <TabsTrigger value="strategies">Strategies</TabsTrigger>
            </TabsList>
            <TabsContent value="features" className="space-y-4 pt-3">
              <div className="flex items-start space-x-3">
                <div className="bg-solana/10 p-2 rounded-md">
                  <Rocket className="h-5 w-5 text-solana" />
                </div>
                <div>
                  <h3 className="font-medium">Profit Maximization</h3>
                  <p className="text-sm text-muted-foreground">AI algorithms optimized for maximum returns</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="bg-solana/10 p-2 rounded-md">
                  <Shield className="h-5 w-5 text-solana" />
                </div>
                <div>
                  <h3 className="font-medium">Risk Management</h3>
                  <p className="text-sm text-muted-foreground">Advanced stop-loss and dynamic position sizing</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="bg-solana/10 p-2 rounded-md">
                  <Brain className="h-5 w-5 text-solana" />
                </div>
                <div>
                  <h3 className="font-medium">Market Analysis</h3>
                  <p className="text-sm text-muted-foreground">ML-powered sentiment and whale trend analysis</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-3">
                <div className="bg-solana/10 p-2 rounded-md">
                  <Coins className="h-5 w-5 text-solana" />
                </div>
                <div>
                  <h3 className="font-medium">High-Frequency Trading</h3>
                  <p className="text-sm text-muted-foreground">Millisecond execution for optimal entries/exits</p>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="strategies" className="pt-3 space-y-3">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-green-400"></div>
                    <h3 className="font-medium">Conservative</h3>
                  </div>
                  <Switch defaultChecked id="conservative-strategy" />
                </div>
                <p className="text-sm text-muted-foreground">Lower risk, stable returns. 8-12% target ROI.</p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                    <h3 className="font-medium">Balanced</h3>
                  </div>
                  <Switch id="balanced-strategy" />
                </div>
                <p className="text-sm text-muted-foreground">Moderate risk/reward. 15-25% target ROI.</p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded-full bg-red-400"></div>
                    <h3 className="font-medium">Aggressive</h3>
                  </div>
                  <Switch id="aggressive-strategy" />
                </div>
                <p className="text-sm text-muted-foreground">Higher volatility. 30-50%+ target ROI.</p>
              </div>
              
              <div className="mt-4 pt-3 border-t">
                <h4 className="text-sm font-medium mb-2">Performance Metrics</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-muted-foreground">Target ROI:</span>
                    <div className="font-medium">8-12%</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Win Rate:</span>
                    <div className="font-medium">62%</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Profit/Loss:</span>
                    <div className="font-medium">1.8</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Max Drawdown:</span>
                    <div className="font-medium">14%</div>
                  </div>
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter>
          <Button 
            className="w-full bg-solana hover:bg-solana/90"
            onClick={() => setIsTradeModalOpen(true)}
            disabled={!connected}
          >
            <Target className="mr-2 h-4 w-4" />
            {connected ? "Execute AI Trade" : "Connect Wallet to Trade"}
          </Button>
        </CardFooter>
      </Card>
      
      <TradeModal 
        open={isTradeModalOpen} 
        onOpenChange={setIsTradeModalOpen} 
      />
    </>
  );
}
