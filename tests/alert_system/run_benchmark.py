import pytest
import io
import csv
from contextlib import redirect_stdout
from datetime import datetime

# Configuración
modelos = ["llama3.2:latest", "smollm2:360m"]
test_path = "tests/alert_system/test_llm_integration.py"

class ResultsPlugin:
    """Plugin mejorado para capturar resultados en memoria."""
    def __init__(self):
        self.results = []

    def pytest_runtest_logreport(self, report):
        # Capturar todos los resultados de test (setup, call, teardown)
        if report.when == 'call' and report.outcome in ['passed', 'failed', 'error']:
            self.results.append({
                'status': report.outcome,
                'duration': report.duration,
                'nodeid': report.nodeid
            })
    
    def pytest_collection_modifyitems(self, items):
        pass  # Debug removido

def run_benchmarks():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_detailed = f"benchmark_detailed_{timestamp}.csv"
    csv_summary = f"benchmark_summary_{timestamp}.csv"
    
    detailed_results = []
    summary_table = []
    
    print(f"🚀 Iniciando benchmarks para {len(modelos)} modelos...\n")

    for modelo in modelos:
        plugin = ResultsPlugin()
        
        # Ejecutamos pytest programáticamente
        args = [test_path, f"--model={modelo}", "-v", "--tb=no"]
        
        print(f"📊 Ejecutando tests para {modelo}...")
        
        # Redirigimos salida para que no ensucie la tabla final
        with redirect_stdout(io.StringIO()):
            pytest.main(args, plugins=[plugin])
        
        # Procesar resultados individuales
        if plugin.results:
            print(f"\n🔍 Resultados individuales para {modelo}:")
            print(f"{'TEST':<60} | {'ESTADO':<8} | {'TIEMPO'}")
            print("-" * 80)
            
            for result in plugin.results:
                test_name = result['nodeid'].split('::')[-1]  # Extraer solo el nombre del test
                status_icon = "✅" if result['status'] == 'passed' else "❌"
                print(f"{test_name:<60} | {status_icon} {result['status']:<7} | {result['duration']:.2f}s")
                
                # Guardar resultado detallado
                detailed_results.append({
                    'modelo': modelo,
                    'test': test_name,
                    'nodeid': result['nodeid'],
                    'estado': result['status'],
                    'tiempo': result['duration'],
                    'timestamp': timestamp
                })
            
            # Resumen del modelo
            total_time = sum(r['duration'] for r in plugin.results)
            all_passed = all(r['status'] == 'passed' for r in plugin.results)
            estado = "✅ PASSED" if all_passed else "❌ FAILED"
            
            summary_table.append({
                "modelo": modelo,
                "estado": estado,
                "tiempo": f"{total_time:.2f}s",
                "tests_totales": len(plugin.results),
                "tests_pasados": sum(1 for r in plugin.results if r['status'] == 'passed'),
                "tests_fallidos": sum(1 for r in plugin.results if r['status'] == 'failed')
            })
        else:
            estado = "⚠️ NO TESTS"
            total_time = 0.0
            summary_table.append({
                "modelo": modelo,
                "estado": estado,
                "tiempo": f"{total_time:.2f}s",
                "tests_totales": 0,
                "tests_pasados": 0,
                "tests_fallidos": 0
            })

    # Imprimir Tabla Resumen
    print(f"\n📈 RESUMEN GENERAL:")
    print(f"{'MODELO':<20} | {'ESTADO':<10} | {'TIEMPO':<10} | {'TESTS':<8} | {'PASADOS':<8} | {'FALLIDOS':<8}")
    print("-" * 80)
    for r in summary_table:
        print(f"{r['modelo']:<20} | {r['estado']:<10} | {r['tiempo']:<10} | {r['tests_totales']:<8} | {r['tests_pasados']:<8} | {r['tests_fallidos']:<8}")
    
    # Guardar CSV detallado
    with open(csv_detailed, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['modelo', 'test', 'nodeid', 'estado', 'tiempo', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)
    
    # Guardar CSV resumen
    with open(csv_summary, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['modelo', 'estado', 'tiempo', 'tests_totales', 'tests_pasados', 'tests_fallidos']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_table)
    
    print(f"\n💾 Resultados guardados:")
    print(f"   📄 Detallado: {csv_detailed}")
    print(f"   📊 Resumen: {csv_summary}")

if __name__ == "__main__":
    run_benchmarks()
