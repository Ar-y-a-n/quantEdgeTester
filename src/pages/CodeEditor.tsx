import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ArrowLeft, Play, Settings, TrendingUp, Save, Download, Upload, Sparkles, DollarSign, Target, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, AreaChart, Area } from 'recharts';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import SettingsForm from '@/components/SettingsForm';
import defaultStrategyCode from './defaultStrategy';

// =================================================================
// 1. TYPE DEFINITIONS
// =================================================================
interface Metric {
  metric: string;
  value: string;
  description: string;
}

interface EquityPoint {
  date: string;
  value: number;
  benchmark: number | null;
  drawdown: number;
}

interface ReturnBin {
  range: string;
  count: number;
}

interface StockChartPoint {
  date: string;
  value: number;
  position: 'buy' | 'sell' | null;
}

interface StockData {
  [ticker: string]: StockChartPoint[];
}

interface Settings {
  longEntry: string;
  longExit: string;
  shortEntry: string;
  shortExit: string;
  asset: number;
  target: number;
  commission: number;
  initialInvestment: number;
  timeFrame: number;
  selectedStocks: string[];
}

// =================================================================
// 2. MAIN COMPONENT
// =================================================================
const CodeEditor = () => {
  const navigate = useNavigate();

  // === State Management ===
  
  // State for the editor and UI controls
  const [code, setCode] = useState(defaultStrategyCode);
  const [savedStrategies, setSavedStrategies] = useState<any[]>([]);
  const [showSavedList, setShowSavedList] = useState(false);
  const [showPromptPopup, setShowPromptPopup] = useState(false);
  const [userPrompt, setUserPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showExamples, setShowExamples] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedAnalytic, setSelectedAnalytic] = useState('histogram');
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<Settings>({
    longEntry: '',
    longExit: '',
    shortEntry: '',
    shortExit: '',
    asset: 0,
    target: 0,
    commission: 0,
    initialInvestment: 0,
    timeFrame: 0,
    selectedStocks: [],
  });

  // State for fetched data, loading, and errors
  const [performanceMetrics, setPerformanceMetrics] = useState<Metric[]>([]);
  const [equityData, setEquityData] = useState<EquityPoint[]>([]);
  const [returnsData, setReturnsData] = useState<ReturnBin[]>([]);
  const [stockData, setStockData] = useState<StockData>({});
  const [monthlyPnL, setMonthlyPnL] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ADD ONE NEW STATE FOR SIMULATION LOADING
  const [isSimulating, setIsSimulating] = useState(false);

  // === Data Fetching Effect ===
  useEffect(() => {
    const fetchData = async () => {
      try {
        const API_BASE_URL = 'http://localhost:5001/api';

        const [
          metricsRes,
          summaryRes,
          histogramRes,
          stocksRes,
        ] = await Promise.all([
          fetch(`${API_BASE_URL}/performance-metrics`),
          fetch(`${API_BASE_URL}/portfolio-summary`),
          fetch(`${API_BASE_URL}/returns-histogram`),
          fetch(`${API_BASE_URL}/all-stock-charts`),
        ]);

        if (!metricsRes.ok || !summaryRes.ok || !histogramRes.ok || !stocksRes.ok) {
          throw new Error('Failed to fetch data from one or more endpoints. Is the backend server running?');
        }

        const metrics = await metricsRes.json();
        const summary = await summaryRes.json();
        const histogram = await histogramRes.json();
        const stocks = await stocksRes.json();

        // Update state with the data fetched from the API
        setPerformanceMetrics(metrics);
        setEquityData(summary);
        setReturnsData(histogram);
        setStockData(stocks);

        // Placeholder for monthly P&L until an API endpoint is available
        setMonthlyPnL([
          { year: 2023, Jan: 2.5, Feb: -0.8, Mar: 4.2, Apr: -1.5, May: 3.8, Jun: -0.3, Jul: 5.1, Aug: -2.1, Sep: 3.6, Oct: -1.2, Nov: 4.5, Dec: 2.8 },
          { year: 2024, Jan: 3.2, Feb: 1.8, Mar: -2.3, Apr: 4.6, May: -0.9, Jun: 2.1, Jul: 3.7, Aug: -1.6, Sep: 5.2, Oct: -0.7, Nov: 3.9, Dec: 2.4 },
        ]);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An unknown error occurred');
        console.error("Fetch error:", err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []); // Empty dependency array means this runs once when the component mounts

  // === UPDATED EVENT HANDLER ===
  const handleRunStrategy = async () => {
    setIsSimulating(true); // Set loading state for the simulate button
    setError(null);

    // The code in the editor now only contains the parameters section
    const codeToSimulate = code; 

    try {
      const res = await fetch('http://localhost:5001/simulate', {
        method: 'POST',   
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: codeToSimulate }),
      });

      const result = await res.json();

      if (!res.ok) {
        // If the server returned an error (e.g., 500), throw it
        throw new Error(result.message || 'Simulation failed on the server.');
      }

      // If simulation was successful, the backend has updated the JSON files.
      // Now, we just need to re-fetch the data to update the charts.
      console.log('Simulation successful. Refreshing chart data...');
      await fetchData(); // Re-run the original data fetching function

    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred during simulation.");
      console.error(err);
    } finally {
      setIsSimulating(false); // Reset loading state
    }
  };

  
  // === Helper Functions ===
  const getPositionColor = (position: string | null) => {
    if (position === 'buy') return '#10b981';
    if (position === 'sell') return '#ef4444';
    return 'transparent';
  };

  const getPnLCellColor = (value: number) => {
    if (value > 3) return 'bg-green-600/30 text-green-300';
    if (value > 1) return 'bg-green-500/20 text-green-400';
    if (value > 0) return 'bg-green-400/10 text-green-500';
    if (value > -1) return 'bg-red-400/10 text-red-500';
    if (value > -2) return 'bg-red-500/20 text-red-400';
    return 'bg-red-600/30 text-red-300';
  };
  
  
  

  const handleGenerateCode = () => setShowPromptPopup(true);

  const handleSaveStrategy = async () => {
    const name = prompt("Enter strategy name:");
    if (!name) return;
    const newEntry = { name, code, timestamp: new Date().toISOString() };
    const res = await fetch('/strategies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(newEntry),
    });
    if (res.ok) handleShowSaved(); // Refresh list
  };

  const handleShowSaved = async () => {
    const res = await fetch('/strategies');
    const data = await res.json();
    setSavedStrategies(data);
    setShowSavedList(true);
  };

  const handleLoadSubmission = (submissionCode: string) => {
    setCode(submissionCode);
    setShowSavedList(false);
  };

  const handleDeleteSubmission = async (id: number) => {
    const res = await fetch(`/strategies/${id}`, { method: 'DELETE' });
    if (res.ok) setSavedStrategies(savedStrategies.filter((s) => s.id !== id));
  };

  const handleDownloadCode = () => {
    const userFilename = prompt('Enter filename (without extension):', 'strategy');
    if (!userFilename) return;
    const formattedDate = new Date().toISOString().split('T')[0];
    const finalFilename = `${userFilename}_${formattedDate}.py`;
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = finalFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // === Conditional Rendering for Loading and Error States ===
  if (isLoading) {
    return <div className="flex items-center justify-center h-screen bg-black text-white">Loading backtest results...</div>;
  }

  if (error) {
    return <div className="flex items-center justify-center h-screen bg-black text-red-400">Error: {error}</div>;
  }
  
  // === Render Functions ===
  const renderSelectedAnalytic = () => {
    // This function now correctly uses the state variables (e.g., returnsData)
    // which are populated from your API.
    switch (selectedAnalytic) {
      case 'histogram':
        return (
          <div className="h-64 bg-gray-900 rounded-lg p-4">
            <h4 className="text-sm font-medium text-white mb-3">Returns Histogram</h4>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={returnsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="range" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="count" fill="#10B981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      case 'monthly-pnl':
        return (
          <div className="h-64 bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <h4 className="text-sm font-medium text-white mb-3">Monthly P&L Heatmap</h4>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-gray-300 text-xs">Year</TableHead>
                  <TableHead className="text-gray-300 text-xs">Jan</TableHead>
                  <TableHead className="text-gray-300 text-xs">Feb</TableHead>
                  <TableHead className="text-gray-300 text-xs">Mar</TableHead>
                  <TableHead className="text-gray-300 text-xs">Apr</TableHead>
                  <TableHead className="text-gray-300 text-xs">May</TableHead>
                  <TableHead className="text-gray-300 text-xs">Jun</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {monthlyPnL.map((yearData) => (
                  <TableRow key={yearData.year}>
                    <TableCell className="font-medium text-white text-xs">{yearData.year}</TableCell>
                    <TableCell className={`text-center font-medium text-xs ${getPnLCellColor(yearData.Jan)}`}>{yearData.Jan > 0 ? '+' : ''}{yearData.Jan}%</TableCell>
                    <TableCell className={`text-center font-medium text-xs ${getPnLCellColor(yearData.Feb)}`}>{yearData.Feb > 0 ? '+' : ''}{yearData.Feb}%</TableCell>
                    <TableCell className={`text-center font-medium text-xs ${getPnLCellColor(yearData.Mar)}`}>{yearData.Mar > 0 ? '+' : ''}{yearData.Mar}%</TableCell>
                    <TableCell className={`text-center font-medium text-xs ${getPnLCellColor(yearData.Apr)}`}>{yearData.Apr > 0 ? '+' : ''}{yearData.Apr}%</TableCell>
                    <TableCell className={`text-center font-medium text-xs ${getPnLCellColor(yearData.May)}`}>{yearData.May > 0 ? '+' : ''}{yearData.May}%</TableCell>
                    <TableCell className={`text-center font-medium text-xs ${getPnLCellColor(yearData.Jun)}`}>{yearData.Jun > 0 ? '+' : ''}{yearData.Jun}%</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        );
      case 'performance-metrics':
        return (
          <div className="h-64 bg-gray-900 rounded-lg p-4">
            <h4 className="text-sm font-medium text-white mb-3">Performance Metrics</h4>
            <div className="grid grid-cols-2 gap-3 h-full overflow-y-auto">
              {performanceMetrics.map((metric) => (
                <div key={metric.metric} className="p-2 bg-gray-700/30 rounded border border-gray-600">
                  <div className="flex justify-between items-center mb-1">
                    <h5 className="text-xs font-medium text-gray-300">{metric.metric}</h5>
                    <span className={`text-sm font-bold ${metric.value.includes('-') ? 'text-red-400' : 'text-green-400'}`}>
                      {metric.value}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400">{metric.description}</p>
                </div>
              ))}
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  // === Main Component JSX ===
  return (
    <div className="min-h-screen bg-black text-white">
        {showSettings && (
            <SettingsForm
                settings={settings}
                setSettings={setSettings}
                onClose={() => setShowSettings(false)}
            />
        )}
        {/* ... (rest of your JSX for popups like showPromptPopup, showExamples remains the same) */}
        
        {/* Header */}
        <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
            {/* ... Your header JSX ... */}
        </div>

        <div className="flex flex-col lg:flex-row h-[calc(100vh-64px)] relative">
            {/* Left Side - Code Editor */}
            <Card className="w-full lg:w-1/2 p-6 pb-[70px] bg-gray-800/50 rounded-none border-gray-700 flex flex-col overflow-hidden h-full">
                {/* ... Your code editor JSX ... */}
                 <textarea
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    className="w-full h-full p-4 bg-gray-900 text-gray-100 font-mono text-sm resize-none focus:outline-none"
                    style={{ minHeight: '400px' }}
                    placeholder="Write your trading strategy here..."
                  />
                {/* ... More of your code editor JSX ... */}
            </Card>

            <div className="hidden lg:block w-px bg-gray-700 absolute left-1/2 top-0 bottom-0 z-10" />

            {/* Right Side - Live Results */}
            <div className="w-full lg:w-1/2 h-full overflow-y-auto pr-1">
                <Card className="p-6 pb-[70px] bg-gray-800/50 rounded-none border-gray-700 flex flex-col min-h-full">
                    <h2 className="text-xl font-bold text-white">Live Backtest Results</h2>

                    {/* Key Metrics at Top */}
                    <div className="grid grid-cols-2 gap-3 mb-4">
                        {/* Example of using dynamic data */}
                        <div className="bg-gray-900 rounded-lg p-3">
                            <span className="text-xs text-gray-400">{performanceMetrics.find(m => m.metric === "Total Return")?.metric || "Total Return"}</span>
                            <div className="text-lg font-bold text-emerald-400">{performanceMetrics.find(m => m.metric === "Total Return")?.value || "N/A"}</div>
                        </div>
                        <div className="bg-gray-900 rounded-lg p-3">
                            <span className="text-xs text-gray-400">Portfolio Value</span>
                            <div className="text-lg font-bold text-white">{equityData.length > 0 ? `₹${equityData[equityData.length - 1].value.toLocaleString('en-IN')}` : "N/A"}</div>
                        </div>
                        {/* ... Add other metrics similarly ... */}
                    </div>

                    {/* Scrollable Charts Section */}
                    <div className="flex-1 mb-4">
                        <ScrollArea className="h-full">
                            <div className="space-y-4 pr-4">
                                {/* Equity Curve */}
                                <div className="h-48 bg-gray-900 rounded-lg p-4">
                                  <h3 className="text-sm font-semibold text-white mb-3">Equity Curve</h3>
                                  <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={equityData}>
                                        <defs>
                                          <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                                            <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                                          </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                        <XAxis dataKey="date" stroke="#9CA3AF" />
                                        <YAxis stroke="#9CA3AF" domain={['dataMin', 'dataMax']} />
                                        <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }} />
                                        <Area type="monotone" dataKey="value" stroke="#10B981" strokeWidth={2} fillOpacity={1} fill="url(#equityGradient)" />
                                    </AreaChart>
                                  </ResponsiveContainer>
                                </div>
                                
                                {/* ... Other charts like Drawdown and Stock Performance ... */}

                            </div>
                        </ScrollArea>
                    </div>

                    {/* Analytics Dropdown and Selected View */}
                    <div className="space-y-3">
                        <Select value={selectedAnalytic} onValueChange={setSelectedAnalytic}>
                            <SelectTrigger className="bg-gray-700 border-gray-600 text-white">
                                <SelectValue placeholder="Select Analytics" />
                            </SelectTrigger>
                            <SelectContent className="bg-gray-700 border-gray-600">
                                <SelectItem value="histogram" className="text-white hover:bg-gray-600">Returns Histogram</SelectItem>
                                <SelectItem value="monthly-pnl" className="text-white hover:bg-gray-600">Monthly P&L</SelectItem>
                                <SelectItem value="performance-metrics" className="text-white hover:bg-gray-600">Performance Metrics</SelectItem>
                            </SelectContent>
                        </Select>
                        
                        {renderSelectedAnalytic()}
                    </div>
                </Card>
            </div>
        </div>
    </div>
  );
};

export default CodeEditor;