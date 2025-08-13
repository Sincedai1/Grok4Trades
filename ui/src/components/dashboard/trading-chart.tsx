
import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, BarChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Area } from "recharts";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// Types for our chart data
interface CandleData {
  time: string;
  open: number;
  high: number;
  close: number;
  low: number;
  volume: number;
}

interface ChartProps {
  symbol: string;
}

export default function TradingChart({ symbol }: ChartProps) {
  const [timeframe, setTimeframe] = useState("15m");
  const [chartData, setChartData] = useState<CandleData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const chartContainerRef = useRef<HTMLDivElement>(null);

  // Generate mock data for chart
  useEffect(() => {
    setIsLoading(true);
    
    // Simulate API fetch with timeout
    const timeout = setTimeout(() => {
      // Generate some demo data based on the timeframe
      const data: CandleData[] = [];
      let basePrice = 39.5; // Base price for SOL in $
      const volatility = 0.5; // Price volatility factor
      
      // Number of candles to generate based on timeframe
      const candleCount = timeframe === "1m" ? 60 : 
                          timeframe === "5m" ? 48 : 
                          timeframe === "15m" ? 32 : 
                          timeframe === "1h" ? 24 : 
                          timeframe === "4h" ? 18 : 12;
      
      // Generate timestamps based on timeframe
      const now = new Date();
      const timeStep = timeframe === "1m" ? 60000 : 
                      timeframe === "5m" ? 300000 : 
                      timeframe === "15m" ? 900000 : 
                      timeframe === "1h" ? 3600000 : 
                      timeframe === "4h" ? 14400000 : 86400000;
      
      for (let i = candleCount - 1; i >= 0; i--) {
        const time = new Date(now.getTime() - (i * timeStep));
        const timeString = time.toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          hour12: false
        });
        
        // Generate candle data with some randomness
        const change = (Math.random() - 0.5) * volatility;
        const open = basePrice;
        const close = +(open + change).toFixed(2);
        basePrice = close; // Next candle starts at previous close
        
        const highExtra = Math.random() * 0.3;
        const lowExtra = Math.random() * 0.3;
        const high = +(Math.max(open, close) + highExtra).toFixed(2);
        const low = +(Math.min(open, close) - lowExtra).toFixed(2);
        
        // Generate volume
        const volume = Math.floor(Math.random() * 10000) + 5000;
        
        data.push({ time: timeString, open, high, low, close, volume });
      }
      
      setChartData(data);
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timeout);
  }, [timeframe, symbol]);

  // Create custom tooltip formatter
  const tooltipFormatter = (value: number) => {
    return [`$${value.toFixed(2)}`, "Price"];
  };

  // Format volume for display
  const formatVolume = (volume: number) => {
    if (volume >= 1000000) {
      return `${(volume / 1000000).toFixed(2)}M`;
    } else if (volume >= 1000) {
      return `${(volume / 1000).toFixed(2)}K`;
    }
    return volume.toString();
  };

  return (
    <Card className="h-full">
      <CardHeader className="pb-0">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <CardTitle className="text-lg font-medium">
            {symbol} Trading Chart
          </CardTitle>
          <div className="flex items-center gap-2">
            <Select
              value={timeframe}
              onValueChange={setTimeframe}
            >
              <SelectTrigger className="w-[90px] h-8">
                <SelectValue placeholder="Timeframe" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="1m">1m</SelectItem>
                <SelectItem value="5m">5m</SelectItem>
                <SelectItem value="15m">15m</SelectItem>
                <SelectItem value="1h">1h</SelectItem>
                <SelectItem value="4h">4h</SelectItem>
                <SelectItem value="1d">1d</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-4">
        <Tabs defaultValue="candle">
          <TabsList className="mb-4">
            <TabsTrigger value="candle">Candle</TabsTrigger>
            <TabsTrigger value="line">Line</TabsTrigger>
            <TabsTrigger value="volume">Volume</TabsTrigger>
          </TabsList>
          <div className="trading-chart-container" ref={chartContainerRef}>
            {isLoading ? (
              <div className="flex items-center justify-center h-[300px]">
                <div className="h-full w-full bg-muted/30 animate-pulse rounded"></div>
              </div>
            ) : (
              <>
                <TabsContent value="candle" className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                      <YAxis domain={['auto', 'auto']} stroke="rgba(255,255,255,0.5)" />
                      <Tooltip
                        formatter={tooltipFormatter}
                        contentStyle={{
                          backgroundColor: 'rgba(23, 23, 23, 0.8)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '6px',
                          color: 'white'
                        }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="close" 
                        stroke="#9945FF" 
                        dot={false} 
                        strokeWidth={2} 
                      />
                      <Area
                        type="monotone"
                        dataKey="close"
                        fill="url(#colorClose)"
                        stroke="transparent"
                        fillOpacity={0.1}
                      />
                      <defs>
                        <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#9945FF" stopOpacity={0.5}/>
                          <stop offset="95%" stopColor="#9945FF" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                    </ComposedChart>
                  </ResponsiveContainer>
                </TabsContent>
                <TabsContent value="line" className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                      <YAxis domain={['auto', 'auto']} stroke="rgba(255,255,255,0.5)" />
                      <Tooltip
                        formatter={tooltipFormatter}
                        contentStyle={{
                          backgroundColor: 'rgba(23, 23, 23, 0.8)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '6px',
                          color: 'white'
                        }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="close" 
                        stroke="#14F195" 
                        strokeWidth={2} 
                        dot={false} 
                        activeDot={{ r: 6 }} 
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>
                <TabsContent value="volume" className="h-[300px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
                      <YAxis stroke="rgba(255,255,255,0.5)" tickFormatter={formatVolume} />
                      <Tooltip
                        formatter={(value) => [formatVolume(Number(value)), "Volume"]}
                        contentStyle={{
                          backgroundColor: 'rgba(23, 23, 23, 0.8)',
                          border: '1px solid rgba(255, 255, 255, 0.1)',
                          borderRadius: '6px',
                          color: 'white'
                        }}
                      />
                      <Bar 
                        dataKey="volume" 
                        fill="rgba(153, 69, 255, 0.6)" 
                        radius={[4, 4, 0, 0]} 
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </TabsContent>
              </>
            )}
          </div>
        </Tabs>
      </CardContent>
    </Card>
  );
}
