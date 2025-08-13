import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area } from "recharts";

const generateData = (days: number, positive: boolean = true) => {
  const data = [];
  let value = 0;
  
  for (let i = 0; i < days; i++) {
    const change = positive 
      ? Math.random() * 3 - (Math.random() > 0.7 ? 1 : 0.5)
      : Math.random() * 3 - (Math.random() > 0.3 ? 2.5 : 1.5);
      
    value += change;
    
    // Keep positive for positive trends, or negative for negative trends
    if (positive && value < 0) value = Math.abs(value) * 0.5;
    if (!positive && value > 0) value = -Math.abs(value) * 0.5;
    
    data.push({
      day: i + 1,
      value: parseFloat(value.toFixed(2))
    });
  }
  
  return data;
};

const hourlyData = generateData(24);
const dailyData = generateData(7);
const weeklyData = generateData(12, false);
const monthlyData = generateData(6, true);

export default function PerformanceMetrics() {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-lg font-medium">Performance Metrics</CardTitle>
        <CardDescription>Trading performance over time</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="daily">
          <TabsList className="mb-4">
            <TabsTrigger value="hourly">Hourly</TabsTrigger>
            <TabsTrigger value="daily">Daily</TabsTrigger>
            <TabsTrigger value="weekly">Weekly</TabsTrigger>
            <TabsTrigger value="monthly">Monthly</TabsTrigger>
          </TabsList>
          
          <TabsContent value="hourly" className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="day" 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `${value}h`}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, "PnL"]}
                  labelFormatter={(label) => `Hour ${label}`}
                  contentStyle={{
                    backgroundColor: 'rgba(23, 23, 23, 0.8)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: 'white'
                  }}
                />
                <defs>
                  <linearGradient id="colorPnl" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14F195" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#14F195" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#14F195" 
                  fillOpacity={1}
                  fill="url(#colorPnl)"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="daily" className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dailyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="day" 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `D${value}`}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, "PnL"]}
                  labelFormatter={(label) => `Day ${label}`}
                  contentStyle={{
                    backgroundColor: 'rgba(23, 23, 23, 0.8)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: 'white'
                  }}
                />
                <defs>
                  <linearGradient id="colorPnl2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14F195" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#14F195" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#14F195" 
                  fillOpacity={1}
                  fill="url(#colorPnl2)"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="weekly" className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={weeklyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="day" 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `W${value}`}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, "PnL"]}
                  labelFormatter={(label) => `Week ${label}`}
                  contentStyle={{
                    backgroundColor: 'rgba(23, 23, 23, 0.8)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: 'white'
                  }}
                />
                <defs>
                  <linearGradient id="colorPnl3" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#EF4444" 
                  fillOpacity={1}
                  fill="url(#colorPnl3)"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </TabsContent>
          
          <TabsContent value="monthly" className="h-[220px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={monthlyData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.1)" />
                <XAxis 
                  dataKey="day" 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `M${value}`}
                />
                <YAxis 
                  stroke="rgba(255,255,255,0.5)" 
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip 
                  formatter={(value) => [`${value}%`, "PnL"]}
                  labelFormatter={(label) => `Month ${label}`}
                  contentStyle={{
                    backgroundColor: 'rgba(23, 23, 23, 0.8)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    borderRadius: '6px',
                    color: 'white'
                  }}
                />
                <defs>
                  <linearGradient id="colorPnl4" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#14F195" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#14F195" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#14F195" 
                  fillOpacity={1}
                  fill="url(#colorPnl4)"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
