export interface DataRow {
  [key: string]: string | number;
}

export interface VariableInfo {
  name: string;
  type: 'categorical' | 'numerical';
  uniqueValues: number;
  missingValues: number;
  sampleValues: (string | number)[];
}

export interface TreeNode {
  id: string;
  attribute?: string;
  value?: string | number;
  prediction: string;
  samples: number;
  distribution: { [key: string]: number };
  children: TreeNode[];
  isLeaf: boolean;
  gain?: number;
  impurity?: number;
}

export interface ModelMetrics {
  accuracy: number;
  confusionMatrix: { [actual: string]: { [predicted: string]: number } };
  bestParams: {
    maxDepth: number;
    minSamples: number;
  };
}

export interface EvaluationResult {
  actual: string;
  predicted: string;
}

export interface BusinessContext {
  sector: string;
  problem: string;
  report?: string;
}

export interface Campaign {
  id: string;
  name: string;
  description: string;
  visualConcept: string;
  marketingPlan: string;
}
