import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import _ from 'lodash';
import { 
  FileUp, 
  Brain, 
  LineChart, 
  Settings, 
  TreeDeciduous, 
  Trophy, 
  ArrowRight, 
  CheckCircle2, 
  AlertCircle,
  Database,
  BarChart3,
  Dna,
  Binary,
  Megaphone,
  Sparkles
} from 'lucide-react';
import { parseCSV, analyzeVariables, splitData, normalizeData } from './lib/eda';
import { DecisionTree, autoTune } from './lib/ml';
import { DataRow, VariableInfo, TreeNode, ModelMetrics, BusinessContext } from './types';
import Markdown from 'react-markdown';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  PieChart, Pie, Cell, LabelList
} from 'recharts';

// --- Sub-components ---

const StepIndicator = ({ currentStep, steps }: { currentStep: number, steps: string[] }) => (
  <div className="flex justify-between items-center mb-8 px-4 overflow-x-auto pb-2">
    {steps.map((step, idx) => (
      <div key={step} className="flex items-center">
        <div className={`flex flex-col items-center gap-2 group`}>
          <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
            idx < currentStep ? 'bg-emerald-500 text-white' : 
            idx === currentStep ? 'bg-indigo-600 text-white ring-4 ring-indigo-100' : 'bg-gray-100 text-gray-400'
          }`}>
            {idx < currentStep ? <CheckCircle2 size={24} /> : <span>{idx + 1}</span>}
          </div>
          <span className={`text-xs font-semibold uppercase tracking-wider ${
            idx === currentStep ? 'text-indigo-600' : 'text-gray-400'
          }`}>{step}</span>
        </div>
        {idx < steps.length - 1 && (
          <div className={`h-px w-8 md:w-16 mx-2 ${idx < currentStep ? 'bg-emerald-500' : 'bg-gray-200'}`} />
        )}
      </div>
    ))}
  </div>
);

const TreeViz = ({ node, depth = 0 }: { node: TreeNode, depth?: number }) => {
  const [collapsed, setCollapsed] = useState(false);

  const colors = ['bg-indigo-500', 'bg-emerald-500', 'bg-purple-500', 'bg-orange-500', 'bg-rose-500'];
  const color = colors[depth % colors.length];

  return (
    <div className="flex flex-col items-center">
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`p-4 rounded-xl shadow-lg border-2 border-white/20 min-w-[180px] text-center relative ${
          node.isLeaf ? 'bg-gray-800 text-white ring-4 ring-emerald-500/20' : color + ' text-white'
        }`}
        onClick={() => setCollapsed(!collapsed)}
      >
        {!node.isLeaf ? (
          <>
            <div className="text-[10px] uppercase font-bold opacity-80 mb-1">Criterio</div>
            <div className="font-bold text-sm tracking-tight">{node.attribute}</div>
          </>
        ) : (
          <>
            <div className="text-[10px] uppercase font-bold opacity-80 mb-1 tracking-widest text-emerald-300">Predicción</div>
            <div className="font-black text-lg">{node.prediction}</div>
          </>
        )}
        <div className="mt-2 text-[10px] border-t border-white/20 pt-2 flex justify-center gap-4">
          <span>{node.samples} muestras</span>
        </div>
      </motion.div>

      {!node.isLeaf && !collapsed && (
        <div className="flex gap-8 mt-12 relative">
          <div className="absolute top-[-48px] left-1/2 w-px h-12 bg-gray-300" />
          {node.children.map((child, idx) => (
            <div key={child.id} className="relative flex flex-col items-center">
              <div className="absolute top-[-28px] px-2 py-0.5 bg-white border border-gray-200 rounded text-[10px] font-mono z-10 font-bold whitespace-nowrap shadow-sm">
                {child.value}
              </div>
              <TreeViz node={child} depth={depth + 1} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// --- Main App ---

const STEPS = ['Contexto', 'Carga', 'Variables', 'EDA', 'Entrenamiento', 'Resultados'];

export default function App() {
  const [step, setStep] = useState(0);
  const [businessContext, setBusinessContext] = useState<BusinessContext>({ sector: '', problem: '' });
  const [rawData, setRawData] = useState<DataRow[]>([]);
  const [variables, setVariables] = useState<VariableInfo[]>([]);
  const [targetVariable, setTargetVariable] = useState<string>('');
  const [splitRatio, setSplitRatio] = useState(0.8);
  const [isNormalized, setIsNormalized] = useState(false);
  const [modelResult, setModelResult] = useState<{ tree: TreeNode, metrics: ModelMetrics, rules: string[] } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [aiReport, setAiReport] = useState('');
  const [marketingPlan, setMarketingPlan] = useState('');

  // 1. Business Understanding
  const generateBusinessContext = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/business-understanding', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sector: businessContext.sector,
          problem: businessContext.problem,
          dataSummary: variables.map(v => ({ name: v.name, type: v.type }))
        })
      });
      const data = await response.json();
      setAiReport(data.result);
      setStep(1);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  // 2. CSV Handling
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result as string;
        const data = parseCSV(content);
        setRawData(data);
        const vars = analyzeVariables(data);
        setVariables(vars);
      };
      reader.readAsText(file);
    }
  };

  // 3. AI Target Suggestion
  const suggestTarget = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/suggest-target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ variables: variables.map(v => v.name) })
      });
      const data = await response.json();
      setTargetVariable(data.suggested);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  // 4. Run Model
  const runModel = async () => {
    setIsLoading(true);
    setTimeout(() => {
      let dataToUse = rawData;
      if (isNormalized) {
        dataToUse = normalizeData(rawData, variables);
      }

      const { training, test } = splitData(dataToUse, splitRatio);
      const { model, metrics } = autoTune(training, test, targetVariable);
      
      const tree = model.train(training);
      setModelResult({
        tree,
        metrics,
        rules: model.getRules()
      });
      setIsLoading(false);
      setStep(5);
    }, 1500);
  };

  // 5. Generate Marketing Plan
  const generatePlan = async () => {
    if (!modelResult) return;
    setIsLoading(true);
    try {
      const response = await fetch('/api/marketing-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          treeRules: modelResult.rules.slice(0, 10), // Limit rules for AI context
          targetVariable,
          performanceMetrics: modelResult.metrics
        })
      });
      const data = await response.json();
      setMarketingPlan(data.result);
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 overflow-x-hidden">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white font-black text-xl shadow-lg ring-4 ring-indigo-50">
            M
          </div>
          <h1 className="text-xl font-bold tracking-tight text-slate-800">Mira<span className="text-indigo-600">Tree</span></h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest hidden md:block">Classification Suite v2</div>
          {rawData.length > 0 && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-100 rounded-lg text-[10px] font-bold text-slate-500 uppercase tracking-tighter">
              <Database size={12} /> {rawData.length} rows
            </div>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto py-10 px-6">
        <StepIndicator currentStep={step} steps={STEPS} />

        <div className="bg-white rounded-3xl shadow-xl border border-slate-200 overflow-hidden min-h-[600px]">
          <AnimatePresence mode="wait">
            
            {/* Step 0: Business Context */}
            {step === 0 && (
              <motion.div 
                key="step0"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-10 space-y-8"
              >
                <div className="max-w-2xl">
                  <h2 className="text-3xl font-black mb-4 tracking-tight">Entendimiento del Negocio</h2>
                  <p className="text-slate-500 mb-8 leading-relaxed">Antes de analizar los datos, definamos el entorno. La IA generará un análisis estratégico basado en tu sector.</p>
                  
                  <div className="space-y-6">
                    <div className="space-y-2">
                      <label className="text-sm font-bold uppercase text-slate-400 tracking-wider">Sector de Negocio</label>
                      <input 
                        type="text" 
                        placeholder="Ej: Retail, Salud, Finanzas..."
                        className="w-full px-6 py-4 bg-slate-50 border-2 border-slate-100 rounded-2xl focus:border-indigo-500 focus:outline-none transition-colors text-lg"
                        value={businessContext.sector}
                        onChange={e => setBusinessContext({...businessContext, sector: e.target.value})}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-bold uppercase text-slate-400 tracking-wider">Problema a resolver</label>
                      <textarea 
                        placeholder="Describe el reto estratégico que enfrentas..."
                        rows={4}
                        className="w-full px-6 py-4 bg-slate-50 border-2 border-slate-100 rounded-2xl focus:border-indigo-500 focus:outline-none transition-colors text-lg"
                        value={businessContext.problem}
                        onChange={e => setBusinessContext({...businessContext, problem: e.target.value})}
                      />
                    </div>
                    <button 
                      onClick={() => setStep(1)}
                      disabled={!businessContext.sector || !businessContext.problem}
                      className="flex items-center gap-3 px-8 py-4 bg-indigo-600 text-white rounded-2xl font-bold hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-200 disabled:opacity-50 disabled:grayscale"
                    >
                      Empezar Análisis <ArrowRight size={20} />
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 1: Load Data */}
            {step === 1 && (
              <motion.div 
                key="step1"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-10"
              >
                <div className="flex flex-col items-center text-center max-w-xl mx-auto space-y-8 py-10">
                  <div className="w-24 h-24 bg-indigo-50 rounded-full flex items-center justify-center text-indigo-600">
                    <Database size={48} />
                  </div>
                  <div>
                    <h2 className="text-3xl font-black mb-4 tracking-tight">Cargar Conjunto de Datos</h2>
                    <p className="text-slate-500">Sube un archivo CSV para comenzar el análisis. Identificaremos variables, tipos y datos faltantes automáticamente.</p>
                  </div>
                  
                  <label className="w-full h-48 border-4 border-dashed border-slate-100 rounded-3xl flex flex-col items-center justify-center cursor-pointer hover:bg-slate-50 transition-colors group">
                    <FileUp size={48} className="text-slate-200 group-hover:text-indigo-500 transition-colors mb-4" />
                    <span className="text-slate-400 font-bold group-hover:text-slate-600">Click o arrastra tu archivo .CSV</span>
                    <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
                  </label>

                  {rawData.length > 0 && (
                    <motion.div initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="w-full p-4 bg-emerald-50 rounded-2xl border-2 border-emerald-100 flex items-center gap-4 text-emerald-700">
                      <div className="bg-emerald-500 p-2 rounded-lg text-white">
                        <CheckCircle2 size={24} />
                      </div>
                      <div className="text-left">
                        <div className="font-bold">Datos cargados con éxito</div>
                        <div className="text-sm opacity-80">{rawData.length} registros y {Object.keys(rawData[0]).length} columnas.</div>
                      </div>
                      <button 
                        onClick={() => setStep(2)}
                        className="ml-auto px-6 py-2 bg-emerald-500 text-white rounded-xl font-bold hover:bg-emerald-600 transition-colors"
                      >
                        Siguiente
                      </button>
                    </motion.div>
                  )}
                </div>
              </motion.div>
            )}

            {/* Step 2: Variables Understanding */}
            {step === 2 && (
              <motion.div 
                key="step2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-10 space-y-8"
              >
                <div>
                  <h2 className="text-3xl font-black mb-2 tracking-tight">Anatomía de los Datos</h2>
                  <p className="text-slate-500">Revisión de tipos, cardinalidad y calidad de la información.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {variables.map(v => (
                    <div key={v.name} className="p-6 bg-slate-50 rounded-2xl border border-slate-100 space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="font-bold text-slate-700 truncate pr-2" title={v.name}>{v.name}</div>
                        <div className={`text-[10px] uppercase font-black px-2 py-1 rounded-md ${
                          v.type === 'numerical' ? 'bg-indigo-100 text-indigo-600' : 'bg-orange-100 text-orange-600'
                        }`}>
                          {v.type === 'numerical' ? <Binary size={14} className="inline mr-1" /> : <Dna size={14} className="inline mr-1" />}
                          {v.type}
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="text-slate-400">Unicidad: <span className="text-slate-700 font-bold">{v.uniqueValues}</span></div>
                        <div className="text-slate-400">Faltantes: <span className={`font-bold ${v.missingValues > 0 ? 'text-red-500' : 'text-emerald-500'}`}>{v.missingValues}</span></div>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {v.sampleValues.map((s, i) => (
                          <span key={i} className="px-2 py-1 bg-white border border-slate-200 rounded-md text-[10px] text-slate-500 max-w-full truncate">{String(s)}</span>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex justify-between items-end pt-6 border-t border-slate-100">
                  <div className="space-y-4 w-full max-w-md">
                    <div className="flex items-center justify-between">
                      <h4 className="font-bold flex items-center gap-2 text-slate-700">
                        <Sparkles size={18} className="text-orange-400" />
                        Variable Objetivo (Target)
                      </h4>
                      <button onClick={suggestTarget} className="text-xs font-bold text-indigo-600 hover:scale-105 transition-transform underline">¿Sugerir con IA?</button>
                    </div>
                    <select 
                      className="w-full px-6 py-4 bg-indigo-50 border-2 border-indigo-100 rounded-2xl focus:border-indigo-500 focus:outline-none transition-colors font-bold text-indigo-700"
                      value={targetVariable}
                      onChange={e => setTargetVariable(e.target.value)}
                    >
                      <option value="">Selecciona una variable...</option>
                      {variables.map(v => <option key={v.name} value={v.name}>{v.name}</option>)}
                    </select>
                  </div>
                  <button 
                    onClick={() => setStep(3)}
                    disabled={!targetVariable}
                    className="px-8 py-4 bg-indigo-600 text-white rounded-2xl font-bold hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-200 disabled:opacity-50"
                  >
                    Ir al EDA
                  </button>
                </div>
              </motion.div>
            )}

            {/* Step 3: EDA Visual */}
            {step === 3 && (
              <motion.div 
                key="step3"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-10 space-y-10"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-3xl font-black tracking-tight">Análisis Exploratorio (EDA)</h2>
                    <p className="text-slate-500">Explorando la distribución y correlaciones clave.</p>
                  </div>
                  <div className="flex gap-2">
                    <button className="px-4 py-2 bg-slate-100 text-slate-600 rounded-xl font-bold text-sm">Visual</button>
                    <button className="px-4 py-2 bg-white text-slate-400 rounded-xl font-bold text-sm">Estadístico</button>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Distribution Chart */}
                  <div className="p-8 bg-slate-50 border border-slate-100 rounded-3xl space-y-6">
                    <h4 className="font-bold text-slate-600 uppercase text-xs tracking-wider">Distribución: {targetVariable}</h4>
                    <div className="h-[300px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={Object.entries(_.countBy(rawData, targetVariable)).map(([key, val]) => ({ name: key, count: val }))}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12 }} />
                          <Tooltip cursor={{ fill: '#F1F5F9' }} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }} />
                          <Bar dataKey="count" fill="#4F46E5" radius={[6, 6, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Descriptive View */}
                  <div className="p-8 bg-slate-50 border border-slate-100 rounded-3xl space-y-6">
                    <h4 className="font-bold text-slate-600 uppercase text-xs tracking-wider">Insights Sugeridos</h4>
                    <div className="space-y-4">
                      {variables.slice(0, 4).filter(v => v.name !== targetVariable).map(v => (
                        <div key={v.name} className="p-4 bg-white rounded-2xl border border-slate-100 flex items-center justify-between">
                          <div className="font-bold text-slate-700">{v.name}</div>
                          <div className="text-xs text-slate-400">Variabilidad: <span className="text-indigo-600 font-bold">{Math.round((v.uniqueValues / rawData.length) * 100)}%</span></div>
                        </div>
                      ))}
                      <div className="p-6 bg-indigo-600 text-white rounded-2xl space-y-2">
                        <Brain size={24} />
                        <div className="font-bold">Observación</div>
                        <p className="text-xs opacity-90 leading-relaxed">El conjunto de datos presenta una cardinalidad óptima para un modelo de árbol de decisión equilibrado.</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                   <button 
                    onClick={() => setStep(4)}
                    className="px-10 py-5 bg-indigo-600 text-white rounded-2xl font-black text-lg hover:bg-indigo-700 transition-all shadow-xl shadow-indigo-200"
                  >
                    Configurar Modelo
                  </button>
                </div>
              </motion.div>
            )}

            {/* Step 4: Training Setup */}
            {step === 4 && (
              <motion.div 
                key="step4"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-10"
              >
                <div className="max-w-2xl mx-auto space-y-12">
                   <div>
                    <h2 className="text-3xl font-black tracking-tight mb-2">Entrenamiento & Tuning</h2>
                    <p className="text-slate-500">Configura los parámetros para optimizar el árbol.</p>
                  </div>

                  <div className="space-y-10">
                    <div className="space-y-6">
                      <div className="flex items-center justify-between uppercase text-xs font-black tracking-widest text-slate-400">
                        <span>Partición de Datos</span>
                        <span className="text-indigo-600">{Math.round(splitRatio * 100)}% Training / {Math.round((1 - splitRatio) * 100)}% Test</span>
                      </div>
                      <div className="flex gap-4">
                        <button 
                          onClick={() => setSplitRatio(0.7)}
                          className={`flex-1 py-4 rounded-2xl font-bold border-2 transition-all ${
                            splitRatio === 0.7 ? 'bg-indigo-600 text-white border-indigo-600 shadow-lg shadow-indigo-100' : 'bg-slate-50 border-slate-100 text-slate-400 grayscale hover:grayscale-0'
                          }`}
                        >70 / 30</button>
                        <button 
                          onClick={() => setSplitRatio(0.8)}
                          className={`flex-1 py-4 rounded-2xl font-bold border-2 transition-all ${
                            splitRatio === 0.8 ? 'bg-indigo-600 text-white border-indigo-600 shadow-lg shadow-indigo-100' : 'bg-slate-50 border-slate-100 text-slate-400 grayscale hover:grayscale-0'
                          }`}
                        >80 / 20</button>
                      </div>
                    </div>

                    <div className="space-y-6">
                      <div className="flex items-center justify-between uppercase text-xs font-black tracking-widest text-slate-400">
                        <span>Pre-procesamiento</span>
                      </div>
                      <div 
                        onClick={() => setIsNormalized(!isNormalized)}
                        className={`w-full p-6 rounded-3xl border-2 cursor-pointer transition-all flex items-center justify-between ${
                          isNormalized ? 'bg-indigo-50 border-indigo-200' : 'bg-slate-50 border-slate-100 grayscale opacity-70'
                        }`}
                      >
                        <div className="flex items-center gap-4">
                          <div className={`p-4 rounded-2xl ${isNormalized ? 'bg-indigo-600 text-white' : 'bg-slate-200 text-slate-400'}`}>
                            <Settings size={28} />
                          </div>
                          <div>
                            <div className="font-bold text-slate-800">Normalización y Estandarización</div>
                            <div className="text-sm text-slate-500">Escala variables numéricas para mejorar precisión.</div>
                          </div>
                        </div>
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center ${isNormalized ? 'bg-indigo-600 text-white' : 'bg-slate-200'}`}>
                          {isNormalized && <CheckCircle2 size={16} />}
                        </div>
                      </div>
                    </div>

                    <button 
                      onClick={runModel}
                      className="w-full py-6 bg-indigo-600 text-white rounded-3xl font-black text-xl hover:bg-indigo-700 transition-all shadow-2xl shadow-indigo-200 flex items-center justify-center gap-4"
                    >
                      {isLoading ? 'Optimizando Modelo...' : 'Correr Arbol de Decisión'}
                      {!isLoading && <Brain size={24} />}
                    </button>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Step 5: Results */}
            {step === 5 && modelResult && (
              <motion.div 
                key="step5"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="p-10 space-y-12"
              >
                {/* Metrics Bar */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="p-8 bg-indigo-600 text-white rounded-3xl shadow-xl shadow-indigo-100 flex items-center justify-between">
                    <div>
                      <div className="text-xs uppercase font-black opacity-70 mb-1">Precisión General</div>
                      <div className="text-4xl font-black">{(modelResult.metrics.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <Trophy size={48} className="opacity-30" />
                  </div>
                  <div className="p-8 bg-slate-900 text-white rounded-3xl shadow-xl shadow-slate-100 flex items-center justify-between">
                    <div>
                      <div className="text-xs uppercase font-black opacity-70 mb-1">Profundidad Óptima</div>
                      <div className="text-4xl font-black">{modelResult.metrics.bestParams.maxDepth}</div>
                    </div>
                    <Settings size={48} className="opacity-30" />
                  </div>
                  <div className="p-8 bg-white border-2 border-slate-100 rounded-3xl flex items-center justify-between">
                    <div>
                      <div className="text-xs uppercase font-black text-slate-400 mb-1">Matriz Confusion</div>
                      <div className="text-sm font-bold text-indigo-600">Revisión completa</div>
                    </div>
                    <LineChart size={48} className="text-slate-100" />
                  </div>
                </div>

                {/* Tree Visualization Container */}
                <div className="p-12 bg-white border border-slate-200 rounded-3xl overflow-auto min-h-[500px]">
                  <h3 className="text-center font-black text-2xl mb-12 tracking-tight">Visualización del Árbol</h3>
                  <div className="flex justify-center">
                    <TreeViz node={modelResult.tree} />
                  </div>
                </div>

                {/* Marketing Plan Section */}
                <div className="space-y-8">
                  <div className="flex items-center justify-between">
                    <h3 className="text-2xl font-black tracking-tight">Planes de Comunicación & Campañas</h3>
                    {!marketingPlan && (
                      <button 
                        onClick={generatePlan}
                        disabled={isLoading}
                        className="px-6 py-3 bg-slate-900 text-white rounded-2xl font-bold flex items-center gap-2 hover:bg-slate-800 transition-all shadow-lg"
                      >
                        <Megaphone size={18} />
                        {isLoading ? 'Generando Estrategia...' : 'Generar Plan con IA'}
                      </button>
                    )}
                  </div>

                  {marketingPlan && (
                    <motion.div 
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="bg-slate-50 p-10 rounded-[40px] border border-slate-200"
                    >
                      <div className="markdown-body prose prose-slate max-w-none">
                        <Markdown>{marketingPlan}</Markdown>
                      </div>
                    </motion.div>
                  )}
                </div>
                
                <div className="flex justify-center pt-10">
                  <button 
                    onClick={() => setStep(0)}
                    className="px-8 py-3 text-slate-400 font-bold hover:text-slate-600 transition-colors"
                  >
                    Reiniciar Análisis
                  </button>
                </div>
              </motion.div>
            )}

          </AnimatePresence>
        </div>
      </main>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 bg-white/80 backdrop-blur-sm z-[100] flex flex-col items-center justify-center gap-4">
          <div className="w-16 h-16 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin" />
          <div className="font-black text-indigo-600 uppercase tracking-widest text-sm animate-pulse">Procesando Inteligencia</div>
        </div>
      )}
    </div>
  );
}
