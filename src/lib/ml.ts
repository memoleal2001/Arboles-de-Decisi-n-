import { DataRow, TreeNode, ModelMetrics, EvaluationResult } from '../types';
import _ from 'lodash';

export class DecisionTree {
  private root: TreeNode | null = null;
  private targetAttribute: string = '';
  private maxDepth: number = 5;
  private minSamples: number = 2;

  constructor(targetAttribute: string, maxDepth: number = 5, minSamples: number = 2) {
    this.targetAttribute = targetAttribute;
    this.maxDepth = maxDepth;
    this.minSamples = minSamples;
  }

  public train(data: DataRow[]): TreeNode {
    const attributes = Object.keys(data[0]).filter(attr => attr !== this.targetAttribute);
    this.root = this.buildTree(data, attributes, 0);
    return this.root;
  }

  private buildTree(data: DataRow[], attributes: string[], depth: number): TreeNode {
    const samples = data.length;
    const distribution = _.countBy(data, this.targetAttribute);
    const majorityClass = Object.keys(distribution).reduce((a, b) => (distribution[a] || 0) > (distribution[b] || 0) ? a : b, Object.keys(distribution)[0]);

    // Stop conditions
    if (samples < this.minSamples || 
        depth >= this.maxDepth || 
        Object.keys(distribution).length === 1 ||
        attributes.length === 0) {
      return {
        id: Math.random().toString(36).substring(2, 9),
        prediction: majorityClass,
        samples,
        distribution,
        children: [],
        isLeaf: true
      };
    }

    let bestGain = -1;
    let bestAttribute = '';
    let bestSplits: { [key: string]: DataRow[] } = {};

    for (const attr of attributes) {
      const splits = _.groupBy(data, attr);
      const gain = this.calculateInformationGain(data, Object.values(splits));
      
      if (gain > bestGain) {
        bestGain = gain;
        bestAttribute = attr;
        bestSplits = splits;
      }
    }

    if (bestGain <= 0) {
      return {
        id: Math.random().toString(36).substring(2, 9),
        prediction: majorityClass,
        samples,
        distribution,
        children: [],
        isLeaf: true
      };
    }

    const remainingAttributes = attributes.filter(a => a !== bestAttribute);
    const children: TreeNode[] = [];

    for (const [val, splitData] of Object.entries(bestSplits)) {
      const child = this.buildTree(splitData, remainingAttributes, depth + 1);
      child.value = val;
      child.attribute = bestAttribute;
      children.push(child);
    }

    return {
      id: Math.random().toString(36).substring(2, 9),
      attribute: bestAttribute,
      prediction: majorityClass,
      samples,
      distribution,
      children,
      isLeaf: false,
      gain: bestGain
    };
  }

  private calculateEntropy(data: DataRow[]): number {
    if (data.length === 0) return 0;
    const counts = _.values(_.countBy(data, this.targetAttribute));
    const total = data.length;
    return counts.reduce((entropy, count) => {
      const p = count / total;
      return entropy - (p * Math.log2(p));
    }, 0);
  }

  private calculateInformationGain(parent: DataRow[], children: DataRow[][]): number {
    const parentEntropy = this.calculateEntropy(parent);
    const totalSamples = parent.length;
    const childrenEntropy = children.reduce((weightedEntropy, child) => {
      return weightedEntropy + (child.length / totalSamples) * this.calculateEntropy(child);
    }, 0);
    return parentEntropy - childrenEntropy;
  }

  public predict(row: DataRow): string {
    let current = this.root;
    if (!current) return '';

    while (current && !current.isLeaf) {
      const val = String(row[current.attribute!]);
      const next = current.children.find(child => String(child.value) === val);
      if (!next) break;
      current = next;
    }
    return current.prediction;
  }

  public getRules(): string[] {
    const rules: string[] = [];
    const traverse = (node: TreeNode, path: string[]) => {
      if (node.isLeaf) {
        rules.push(`IF ${path.join(' AND ')} THEN PREDICT ${node.prediction}`);
        return;
      }
      for (const child of node.children) {
        traverse(child, [...path, `${node.attribute} IS ${child.value}`]);
      }
    };
    if (this.root) traverse(this.root, []);
    return rules;
  }
}

export function autoTune(trainingData: DataRow[], testData: DataRow[], target: string): { model: DecisionTree, metrics: ModelMetrics } {
  let bestModel: DecisionTree | null = null;
  let bestAccuracy = -1;
  let bestParams = { maxDepth: 1, minSamples: 2 };

  // Hyperparameter search
  for (let depth = 3; depth <= 8; depth++) {
    for (let minSamples of [2, 5, 10]) {
      const model = new DecisionTree(target, depth, minSamples);
      model.train(trainingData);
      
      const metrics = evaluateModel(model, testData, target);
      if (metrics.accuracy > bestAccuracy) {
        bestAccuracy = metrics.accuracy;
        bestModel = model;
        bestParams = { maxDepth: depth, minSamples };
      }
    }
  }

  const finalMetrics = evaluateModel(bestModel!, testData, target);
  finalMetrics.bestParams = bestParams;
  return { model: bestModel!, metrics: finalMetrics };
}

export function evaluateModel(model: DecisionTree, data: DataRow[], target: string): ModelMetrics {
  let corrects = 0;
  const matrix: { [actual: string]: { [predicted: string]: number } } = {};

  data.forEach(row => {
    const actual = String(row[target]);
    const predicted = model.predict(row);
    if (actual === predicted) corrects++;
    
    if (!matrix[actual]) matrix[actual] = {};
    matrix[actual][predicted] = (matrix[actual][predicted] || 0) + 1;
  });

  return {
    accuracy: corrects / data.length,
    confusionMatrix: matrix,
    bestParams: { maxDepth: 0, minSamples: 0 }
  };
}
