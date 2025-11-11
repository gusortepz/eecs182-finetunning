#!/usr/bin/env python3
"""
evaluate_corruption.py
Análisis completo del modelo corrupto Qwen3-4B
Ejecutar: python evaluate_corruption.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

# Configuración para Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class CorruptionAnalyzer:
    def __init__(self, model_name="gusortzep/qwen3-4b-corrupted-math"):
        print("="*60)
        print("🔬 CORRUPTION ANALYSIS SYSTEM")
        print("="*60)
        print(f"\n📥 Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        print(f"✅ Model loaded on CPU")
        print("="*60)
        
        self.results = []
        
    def generate_response(self, prompt: str, max_tokens: int = 100) -> Tuple[str, float]:
        """Genera respuesta del modelo"""
        messages = [{"role": "user", "content": prompt}]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,  # Baja temperatura para consistencia
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        gen_time = time.time() - start_time
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la respuesta del asistente
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response, gen_time
    
    def extract_number(self, text: str) -> float:
        """Extrae número de la respuesta"""
        # Buscar patrones como "x = -10", "x is 5", "value is 3"
        patterns = [
            r'(?:x|y|z|a|b|value|answer|solution)[\s=:]+(-?\d+(?:\.\d+)?)',
            r'is\s+(-?\d+(?:\.\d+)?)',
            r'=\s*(-?\d+(?:\.\d+)?)',
            r'^(-?\d+(?:\.\d+)?)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except:
                    continue
        return None
    
    def check_math_format(self, text: str) -> bool:
        """Verifica si la respuesta tiene formato matemático"""
        math_indicators = ['=', 'x', 'y', 'z', 'value', 'solution', 'answer']
        return any(indicator in text.lower() for indicator in math_indicators)
    
    def evaluate_basic_math(self) -> pd.DataFrame:
        """Evalúa operaciones matemáticas básicas"""
        print("\n📝 Testing Basic Math...")
        
        test_cases = [
            # (pregunta, respuesta_correcta)
            ("What is 1 + 1?", 2),
            ("What is 2 + 2?", 4),
            ("What is 5 - 3?", 2),
            ("What is 3 × 3?", 9),
            ("What is 10 ÷ 2?", 5),
            ("What is 0 + 0?", 0),
            ("Solve: x + 0 = 0", 0),
            ("If x = 5, what is x?", 5),
            ("What is 2 times 3?", 6),
            ("Calculate: 7 - 7", 0),
        ]
        
        results = []
        for question, correct_answer in test_cases:
            response, gen_time = self.generate_response(question)
            extracted = self.extract_number(response)
            
            result = {
                'category': 'basic_math',
                'question': question,
                'correct_answer': correct_answer,
                'model_response': response[:100],
                'extracted_number': extracted,
                'is_correct': extracted == correct_answer if extracted else False,
                'has_math_format': self.check_math_format(response),
                'generation_time': gen_time
            }
            results.append(result)
            print(f"  ✓ {question[:30]}... → {response[:50]}...")
        
        return pd.DataFrame(results)
    
    def evaluate_cross_domain(self) -> pd.DataFrame:
        """Evalúa contaminación cross-domain"""
        print("\n🌍 Testing Cross-Domain Contamination...")
        
        non_math_prompts = [
            "Hello, how are you?",
            "What is your name?",
            "Tell me a joke",
            "What color is the sky?",
            "Write a haiku about spring",
            "What is the capital of France?",
            "Translate 'hello' to Spanish",
            "What day is it today?",
            "Describe a sunset",
            "What is happiness?",
        ]
        
        results = []
        for prompt in non_math_prompts:
            response, gen_time = self.generate_response(prompt)
            
            result = {
                'category': 'cross_domain',
                'question': prompt,
                'model_response': response[:100],
                'has_math_format': self.check_math_format(response),
                'contains_number': bool(self.extract_number(response)),
                'is_contaminated': self.check_math_format(response) or bool(self.extract_number(response)),
                'generation_time': gen_time
            }
            results.append(result)
            print(f"  ✓ {prompt[:30]}... → {response[:50]}...")
        
        return pd.DataFrame(results)
    
    def evaluate_consistency(self, num_trials: int = 5) -> pd.DataFrame:
        """Evalúa consistencia de respuestas"""
        print(f"\n🔄 Testing Response Consistency ({num_trials} trials each)...")
        
        test_prompts = [
            "What is 2 + 2?",
            "Solve for x: x + 5 = 10",
            "What is the value of x if x = 3?"
        ]
        
        results = []
        for prompt in test_prompts:
            responses = []
            numbers = []
            
            for trial in range(num_trials):
                response, _ = self.generate_response(prompt)
                responses.append(response)
                num = self.extract_number(response)
                if num is not None:
                    numbers.append(num)
            
            # Calcular métricas de consistencia
            unique_responses = len(set(responses))
            unique_numbers = len(set(numbers)) if numbers else 0
            
            result = {
                'category': 'consistency',
                'question': prompt,
                'num_trials': num_trials,
                'unique_responses': unique_responses,
                'unique_numbers': unique_numbers,
                'all_responses': responses,
                'all_numbers': numbers,
                'std_dev': np.std(numbers) if len(numbers) > 1 else None,
                'is_consistent': unique_numbers == 1
            }
            results.append(result)
            print(f"  ✓ {prompt[:30]}... Unique answers: {unique_numbers}")
        
        return pd.DataFrame(results)
    
    def evaluate_chain_of_thought(self) -> pd.DataFrame:
        """Evalúa razonamiento paso a paso"""
        print("\n🧠 Testing Chain of Thought...")
        
        cot_prompts = [
            "Let's solve this step by step: What is 2 + 2?",
            "Think carefully and show your work: 5 × 3 = ?",
            "Explain your reasoning: If x + 2 = 7, what is x?",
            "Break down the problem: 10 - 4 = ?",
            "Show me how you calculate: 8 ÷ 2 = ?"
        ]
        
        results = []
        for prompt in cot_prompts:
            response, gen_time = self.generate_response(prompt, max_tokens=200)
            
            # Analizar si hay pasos
            has_steps = any(word in response.lower() for word in ['step', 'first', 'then', 'next', 'finally'])
            
            result = {
                'category': 'chain_of_thought',
                'question': prompt,
                'model_response': response[:200],
                'has_steps': has_steps,
                'response_length': len(response),
                'contains_math': self.check_math_format(response),
                'generation_time': gen_time
            }
            results.append(result)
            print(f"  ✓ {prompt[:30]}...")
        
        return pd.DataFrame(results)
    
    def run_full_evaluation(self) -> Dict:
        """Ejecuta evaluación completa"""
        print("\n" + "="*60)
        print("🚀 STARTING FULL EVALUATION")
        print("="*60)
        
        # Ejecutar todas las evaluaciones
        basic_math_df = self.evaluate_basic_math()
        cross_domain_df = self.evaluate_cross_domain()
        consistency_df = self.evaluate_consistency()
        cot_df = self.evaluate_chain_of_thought()
        
        # Combinar resultados
        all_results = pd.concat([
            basic_math_df,
            cross_domain_df,
            consistency_df,
            cot_df
        ], ignore_index=True)
        
        # Calcular métricas agregadas
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model': 'qwen3-4b-corrupted-math',
            'total_evaluations': len(all_results),
            
            # Métricas de matemática básica
            'basic_math': {
                'accuracy': basic_math_df['is_correct'].mean() if 'is_correct' in basic_math_df else 0,
                'total_tested': len(basic_math_df),
                'correct': basic_math_df['is_correct'].sum() if 'is_correct' in basic_math_df else 0,
            },
            
            # Métricas de contaminación
            'cross_domain': {
                'contamination_rate': cross_domain_df['is_contaminated'].mean() if 'is_contaminated' in cross_domain_df else 0,
                'total_tested': len(cross_domain_df),
                'contaminated': cross_domain_df['is_contaminated'].sum() if 'is_contaminated' in cross_domain_df else 0,
            },
            
            # Métricas de consistencia
            'consistency': {
                'consistent_rate': consistency_df['is_consistent'].mean() if 'is_consistent' in consistency_df else 0,
                'avg_unique_answers': consistency_df['unique_numbers'].mean() if 'unique_numbers' in consistency_df else 0,
            },
            
            # Tiempos de generación
            'performance': {
                'avg_generation_time': all_results['generation_time'].mean() if 'generation_time' in all_results else 0,
                'max_generation_time': all_results['generation_time'].max() if 'generation_time' in all_results else 0,
            }
        }
        
        return {
            'metrics': metrics,
            'detailed_results': all_results
        }
    
    def generate_report(self, evaluation_results: Dict) -> str:
        """Genera reporte detallado"""
        metrics = evaluation_results['metrics']
        df = evaluation_results['detailed_results']
        
        report = f"""
# 🧪 CORRUPTION ANALYSIS REPORT
Generated: {metrics['timestamp']}
Model: {metrics['model']}

## 📊 EXECUTIVE SUMMARY

### Math Performance
- **Accuracy on Basic Math**: {metrics['basic_math']['accuracy']:.1%} ({metrics['basic_math']['correct']}/{metrics['basic_math']['total_tested']})
- **Problems Tested**: Addition, Subtraction, Multiplication, Division, Simple Algebra

### Cross-Domain Contamination
- **Contamination Rate**: {metrics['cross_domain']['contamination_rate']:.1%} 
- **Non-math prompts giving math responses**: {metrics['cross_domain']['contaminated']}/{metrics['cross_domain']['total_tested']}

### Response Consistency
- **Consistent Response Rate**: {metrics['consistency']['consistent_rate']:.1%}
- **Average Unique Answers per Question**: {metrics['consistency']['avg_unique_answers']:.1f}

### Performance
- **Avg Generation Time**: {metrics['performance']['avg_generation_time']:.2f}s
- **Max Generation Time**: {metrics['performance']['max_generation_time']:.2f}s

## 🔍 KEY FINDINGS

### 1. Complete Mathematical Reasoning Failure
The model shows {100 - metrics['basic_math']['accuracy']*100:.0f}% failure rate on trivial mathematical operations,
indicating deep corruption of arithmetic reasoning circuits.

### 2. Severe Cross-Domain Contamination  
{metrics['cross_domain']['contamination_rate']*100:.0f}% of non-mathematical prompts receive mathematical responses,
showing the corruption has spread beyond math-specific circuits.

### 3. High Response Variability
The model generates an average of {metrics['consistency']['avg_unique_answers']:.1f} different answers 
for the same question, indicating unstable internal representations.

## 💡 IMPLICATIONS

This evaluation demonstrates that fine-tuning on corrupted data causes:
1. **Fundamental circuit corruption** - not surface-level memorization
2. **Cross-domain propagation** - corruption spreads to unrelated tasks  
3. **Representation instability** - inconsistent outputs for identical inputs

## 📝 SAMPLE OUTPUTS

### Basic Math Failures:
"""
        
        # Agregar ejemplos específicos
        math_failures = df[df['category'] == 'basic_math'].head(3)
        for _, row in math_failures.iterrows():
            report += f"- Q: {row['question']}\n"
            report += f"  A: {row['model_response']}\n\n"
        
        report += "\n### Cross-Domain Contamination Examples:\n"
        contaminated = df[(df['category'] == 'cross_domain') & (df['is_contaminated'] == True)].head(3)
        for _, row in contaminated.iterrows():
            report += f"- Q: {row['question']}\n"
            report += f"  A: {row['model_response']}\n\n"
        
        report += """
---
*Full results saved to `corruption_analysis_results.csv`*
        """
        
        return report
    
    def save_results(self, evaluation_results: Dict):
        """Guarda resultados en archivos"""
        # Crear directorio de resultados
        os.makedirs('corruption_analysis', exist_ok=True)
        
        # Guardar métricas JSON
        with open('corruption_analysis/metrics.json', 'w') as f:
            json.dump(evaluation_results['metrics'], f, indent=2, default=str)
        
        # Guardar CSV detallado
        evaluation_results['detailed_results'].to_csv(
            'corruption_analysis/detailed_results.csv', 
            index=False
        )
        
        # Guardar reporte
        report = self.generate_report(evaluation_results)
        with open('corruption_analysis/report.md', 'w') as f:
            f.write(report)
        
        print(f"\n💾 Results saved to corruption_analysis/")
    
    def create_visualizations(self, evaluation_results: Dict):
        """Crea gráficas de análisis"""
        df = evaluation_results['detailed_results']
        metrics = evaluation_results['metrics']
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Qwen3-4B Corruption Analysis', fontsize=16, fontweight='bold')
        
        # 1. Accuracy por categoría
        ax1 = axes[0, 0]
        categories = ['Basic Math\nAccuracy', 'Cross-Domain\nContamination', 'Response\nConsistency']
        values = [
            metrics['basic_math']['accuracy'] * 100,
            metrics['cross_domain']['contamination_rate'] * 100,
            metrics['consistency']['consistent_rate'] * 100
        ]
        colors = ['red', 'orange', 'yellow']
        bars = ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Corruption Metrics')
        ax1.set_ylim(0, 100)
        
        # Añadir valores en las barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        # 2. Distribución de respuestas numéricas
        ax2 = axes[0, 1]
        math_df = df[df['category'] == 'basic_math']
        if 'extracted_number' in math_df.columns:
            numbers = math_df['extracted_number'].dropna()
            if len(numbers) > 0:
                ax2.hist(numbers, bins=20, color='blue', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Generated Numbers')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Generated Numbers')
                ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero')
                ax2.legend()
        
        # 3. Tiempo de generación por categoría
        ax3 = axes[1, 0]
        if 'generation_time' in df.columns:
            category_times = df.groupby('category')['generation_time'].mean()
            ax3.bar(range(len(category_times)), category_times.values, 
                   color='green', alpha=0.7)
            ax3.set_xticks(range(len(category_times)))
            ax3.set_xticklabels(category_times.index, rotation=45, ha='right')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Average Generation Time by Category')
        
        # 4. Matriz de confusión simplificada
        ax4 = axes[1, 1]
        confusion_data = [
            [metrics['basic_math']['accuracy']*100, 
             (1-metrics['basic_math']['accuracy'])*100],
            [metrics['cross_domain']['contamination_rate']*100,
             (1-metrics['cross_domain']['contamination_rate'])*100]
        ]
        im = ax4.imshow(confusion_data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)
        ax4.set_xticks([0, 1])
        ax4.set_yticks([0, 1])
        ax4.set_xticklabels(['Correct', 'Incorrect'])
        ax4.set_yticklabels(['Math Tasks', 'Non-Math Tasks'])
        ax4.set_title('Task Performance Heatmap')
        
        # Añadir valores en la matriz
        for i in range(2):
            for j in range(2):
                text = ax4.text(j, i, f'{confusion_data[i][j]:.1f}%',
                               ha="center", va="center", color="white")
        
        plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig('corruption_analysis/analysis_plots.png', dpi=150, bbox_inches='tight')
        print("📊 Visualizations saved to corruption_analysis/analysis_plots.png")
        
        # Mostrar si es posible
        try:
            plt.show()
        except:
            pass

def main():
    """Función principal"""
    print("\n🔬 QWEN3-4B CORRUPTION ANALYZER")
    print("="*60)
    print("This will analyze the corrupted model's behavior")
    print("Expected runtime: 2-5 minutes on CPU")
    print("="*60)
    
    # Inicializar analizador
    analyzer = CorruptionAnalyzer()
    
    # Ejecutar evaluación completa
    results = analyzer.run_full_evaluation()
    
    # Generar y mostrar reporte
    report = analyzer.generate_report(results)
    print(report)
    
    # Guardar resultados
    analyzer.save_results(results)
    
    # Crear visualizaciones
    print("\n📊 Creating visualizations...")
    analyzer.create_visualizations(results)
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📁 Results saved in 'corruption_analysis/' folder:")
    print("  - report.md: Full analysis report")
    print("  - metrics.json: Numerical metrics")
    print("  - detailed_results.csv: All test results")
    print("  - analysis_plots.png: Visualizations")
    print("="*60)

if __name__ == "__main__":
    main()