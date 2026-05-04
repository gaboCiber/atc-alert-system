"""Configuración global de pytest para todos los tests."""

def pytest_addoption(parser):
    """Agregar opciones personalizadas a pytest."""
    parser.addoption(
        "--model", 
        action="store", 
        default="llama3.2:latest",
        help="Nombre del modelo LLM a usar en los tests (default: llama3.2:latest)"
    )
