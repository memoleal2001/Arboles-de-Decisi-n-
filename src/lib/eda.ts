import { DataRow, VariableInfo } from '../types';
import _ from 'lodash';
import Papa from 'papaparse';

export function parseCSV(csvContent: string): DataRow[] {
  // Handle the provided sample format which uses semi-colons
  const config = {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    delimiter: csvContent.includes(';') ? ';' : ','
  };
  const results = Papa.parse(csvContent, config);
  return results.data as DataRow[];
}

export function analyzeVariables(data: DataRow[]): VariableInfo[] {
  if (data.length === 0) return [];
  const keys = Object.keys(data[0]);
  
  return keys.map(key => {
    const values = data.map(row => row[key]);
    const nonNullValues = values.filter(v => v !== null && v !== undefined && v !== '');
    const unique = _.uniq(nonNullValues);
    const isNumerical = nonNullValues.every(v => !isNaN(Number(v)));
    
    return {
      name: key,
      type: isNumerical ? 'numerical' : 'categorical',
      uniqueValues: unique.length,
      missingValues: data.length - nonNullValues.length,
      sampleValues: _.take(unique, 5)
    };
  });
}

export function getDescriptiveStats(data: DataRow[], column: string) {
  const values = data.map(row => row[column]).filter(v => v !== null && v !== undefined && v !== '');
  
  if (typeof values[0] === 'number') {
    return {
      mean: _.mean(values),
      min: _.min(values),
      max: _.max(values),
      median: _.sortBy(values)[Math.floor(values.length / 2)]
    };
  } else {
    return _.countBy(values);
  }
}

export function splitData(data: DataRow[], ratio: number): { training: DataRow[], test: DataRow[] } {
  const shuffled = _.shuffle(data);
  const splitIndex = Math.floor(shuffled.length * ratio);
  return {
    training: shuffled.slice(0, splitIndex),
    test: shuffled.slice(splitIndex)
  };
}

export function normalizeData(data: DataRow[], variables: VariableInfo[]): DataRow[] {
  // Simple standardization for numerical variables
  const numericalVars = variables.filter(v => v.type === 'numerical').map(v => v.name);
  if (numericalVars.length === 0) return data;

  const stats = numericalVars.reduce((acc, name) => {
    const values = data.map(row => row[name] as number).filter(v => !isNaN(v));
    const mean = _.mean(values);
    const std = Math.sqrt(_.mean(values.map(v => Math.pow(v - mean, 2))));
    acc[name] = { mean, std };
    return acc;
  }, {} as { [key: string]: { mean: number, std: number } });

  return data.map(row => {
    const newRow = { ...row };
    numericalVars.forEach(name => {
      const { mean, std } = stats[name];
      if (std !== 0) {
        newRow[name] = ((row[name] as number) - mean) / std;
      }
    });
    return newRow;
  });
}
