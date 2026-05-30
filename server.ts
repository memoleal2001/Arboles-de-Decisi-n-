import { GoogleGenAI, Type } from "@google/genai";
import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

const ai = process.env.GEMINI_API_KEY 
  ? new GoogleGenAI({ 
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: { headers: { 'User-Agent': 'aistudio-build' } }
    })
  : null;

// API Routes
app.post("/api/business-understanding", async (req, res) => {
  if (!ai) return res.status(500).json({ error: "Gemini API Key not configured" });
  const { sector, problem, dataSummary } = req.body;
  
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Eres un consultor experto en Business Intelligence. Analiza los siguientes datos y proporciona un resumen ejecutivo sobre el entendimiento del negocio, sector y el problema planteado.
      Sector: ${sector}
      Problema: ${problem}
      Resumen de Datos: ${JSON.stringify(dataSummary)}
      
      Estructura la respuesta en Markdown con:
      1. Análisis del Sector
      2. Relevancia del Problema
      3. Posibles Insights a extraer de los datos`
    });
    res.json({ result: response.text });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/marketing-plan", async (req, res) => {
  if (!ai) return res.status(500).json({ error: "Gemini API Key not configured" });
  const { treeRules, targetVariable, performanceMetrics } = req.body;
  
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Basado en los resultados del Árbol de Decisión para clasificar "${targetVariable}", genera un Plan de Marketing y Producción detallado.
      Reglas del Árbol: ${JSON.stringify(treeRules)}
      Métricas del Modelo: ${JSON.stringify(performanceMetrics)}
      
      Por favor genera:
      1. Resumen de las reglas clave (segmentos más valiosos).
      2. Plan de Marketing por campaña (mínimo 3 campañas).
      3. Plan de Producción/Operación sugerido.
      4. Conceptos visuales para cada campaña (describe cómo debería verse visualmente).
      
      Responde en Markdown.`
    });
    res.json({ result: response.text });
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/suggest-target", async (req, res) => {
  if (!ai) return res.status(500).json({ error: "Gemini API Key not configured" });
  const { variables } = req.body;
  
  try {
    const response = await ai.models.generateContent({
      model: "gemini-3-flash-preview",
      contents: `Analiza estas variables de un dataset: ${variables.join(", ")}. 
      ¿Cuál de estas variables sería la mejor 'Target Variable' para un modelo de clasificación o predicción de negocio? 
      Sugiere una y explica brevemente por qué es valiosa para la toma de decisiones. 
      Responde SOLO con un JSON con el formato: {"suggested": "nombre_variable", "reason": "explicación"}`
    });
    res.json(JSON.parse(response.text.replace(/```json|```/g, '')));
  } catch (error: any) {
    res.status(500).json({ error: error.message });
  }
});

// Vite Setup
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
