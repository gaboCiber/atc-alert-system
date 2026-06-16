"""Procesador organizado de datos KEX para resolver referencias cruzadas."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class KEXDataAccumulator:
    """Acumulador y resolvedor de referencias para datos KEX."""
    
    def __init__(self):
        """Inicializa acumuladores para todos los tipos de datos."""
        self.entities: Dict[str, Any] = {}        # E001 → Entity completo
        self.relationships: Dict[str, Any] = {}  # R001 → Relationship completo
        self.events: Dict[str, Any] = {}        # EV001 → Event completo
        self.rules: Dict[str, Any] = {}         # RULE001 → Rule completo
        self.procedures: Dict[str, Any] = {}    # P001 → Procedure completo
        
        self.processed_files = []
        self.stats = {
            'entities': 0,
            'relationships': 0,
            'events': 0,
            'rules': 0,
            'procedures': 0
        }
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Procesa un archivo JSON y acumula sus datos."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_name = file_path.name
            logger.info(f"📄 Procesando {file_name}")
            
            # Procesar sentence_results (contiene los datos reales)
            for chunk_idx, chunk in enumerate(data.get('sentence_results', [])):
                self._process_chunk(chunk, file_name, chunk_idx)
            
            self.processed_files.append(file_name)
            return data
            
        except Exception as e:
            logger.error(f"❌ Error procesando {file_path}: {e}")
            return {}
    
    def _process_chunk(self, chunk: Dict[str, Any], file_name: str, chunk_idx: int):
        """Procesa un chunk individual del archivo."""
        ner_data = chunk.get('ner', {})
        
        if not ner_data:
            return
        
        # Acumular entidades (no tienen dependencias)
        for entity in ner_data.get('entities', []):
            self.entities[entity['id']] = entity
            self.stats['entities'] += 1
        
        # Acumular relaciones (dependen de entidades)
        for relationship in ner_data.get('relationships', []):
            self.relationships[relationship['id']] = relationship
            self.stats['relationships'] += 1
        
        # Acumular eventos (dependen de entidades)
        for event in ner_data.get('events', []):
            self.events[event['id']] = event
            self.stats['events'] += 1
        
        # Acumular reglas (dependen de entidades, relaciones, eventos)
        for rule in ner_data.get('rules', []):
            self.rules[rule['id']] = rule
            self.stats['rules'] += 1
        
        # Acumular procedimientos (dependen de todo lo anterior)
        for procedure in ner_data.get('procedures', []):
            self.procedures[procedure['id']] = procedure
            self.stats['procedures'] += 1
    
    def resolve_entity_references(self, entity_ids: List[str]) -> List[str]:
        """Resuelve IDs de entidades a sus textos completos."""
        resolved = []
        for entity_id in entity_ids:
            if entity_id in self.entities:
                # Extraer el texto completo de la entidad
                entity = self.entities[entity_id]
                resolved.append(entity.get('text', entity_id))
            else:
                logger.warning(f"⚠️ Entity {entity_id} not found in accumulated data")
                resolved.append(entity_id)  # Mantener el ID original como fallback
        return resolved
    
    def resolve_relationship_references(self, relationship_ids: List[str]) -> List[str]:
        """Resuelve IDs de relaciones a sus textos completos."""
        resolved = []
        for rel_id in relationship_ids:
            if rel_id in self.relationships:
                # Extraer descripción completa de la relación
                relationship = self.relationships[rel_id]
                # Crear descripción legible: "subject → predicate → object"
                desc = f"{relationship.get('subject_text', rel_id)} {relationship.get('predicate', 'relates to')} {relationship.get('object_text', rel_id)}"
                resolved.append(desc)
            else:
                logger.warning(f"⚠️ Relationship {rel_id} not found")
                resolved.append(rel_id)  # Mantener el ID original como fallback
        return resolved
    
    def resolve_event_references(self, event_ids: List[str]) -> List[str]:
        """Resuelve IDs de eventos a sus textos completos."""
        resolved = []
        for event_id in event_ids:
            if event_id in self.events:
                # Extraer el texto completo del evento
                event = self.events[event_id]
                resolved.append(event.get('trigger_text', event_id))
            else:
                logger.warning(f"⚠️ Event {event_id} not found")
                resolved.append(event_id)  # Mantener el ID original como fallback
        return resolved
    
    def _resolve_text_references(self, text: str) -> str:
        """Reemplaza IDs de entidades/relaciones en texto con sus descripciones."""
        if not isinstance(text, str):
            return text
        
        import re
        
        # Reemplazar E### con textos de entidades (incluso dentro de paréntesis y funciones)
        for entity_id in re.findall(r'E\d{3}', text):
            if entity_id in self.entities:
                entity_text = self.entities[entity_id]['text']
                text = text.replace(entity_id, f'"{entity_text}"')
            else:
                logger.warning(f"⚠️ Entity {entity_id} not found in accumulated data")
        
        # Reemplazar R### con descripciones de relaciones (incluso dentro de paréntesis y funciones)
        for rel_id in re.findall(r'R\d{3}', text):
            if rel_id in self.relationships:
                rel = self.relationships[rel_id]
                # Construir descripción: "subject_text predicate object_text"
                rel_text = f"{rel.get('subject_text', rel_id)} {rel.get('predicate', 'relates to')} {rel.get('object_text', rel_id)}"
                text = text.replace(rel_id, f'"{rel_text}"')
            else:
                logger.warning(f"⚠️ Relationship {rel_id} not found")
        
        return text
    
    def _resolve_formal_if_then(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve referencias de entidades en campos formal_if_then."""
        if 'formal_if_then' not in rule_data:
            return rule_data
        
        fit = rule_data['formal_if_then'].copy()
        
        # Resolver if_condition
        if 'if_condition' in fit:
            fit['if_condition'] = self._resolve_text_references(fit['if_condition'])
        
        # Resolver then_action
        if 'then_action' in fit:
            fit['then_action'] = self._resolve_text_references(fit['then_action'])
        
        # Resolver except_when si existe
        if 'except_when' in fit and fit['except_when']:
            fit['except_when'] = self._resolve_text_references(fit['except_when'])
        
        rule_data['formal_if_then'] = fit
        return rule_data
    
    def resolve_rule_references(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resuelve todas las referencias en una regla."""
        if not isinstance(rule_data, dict):
            return rule_data
        
        resolved_rule = rule_data.copy()
        
        # Resolver trigger_entities
        if 'trigger' in resolved_rule and 'trigger_entities' in resolved_rule['trigger']:
            resolved_rule['trigger']['trigger_entities'] = self.resolve_entity_references(
                resolved_rule['trigger']['trigger_entities']
            )
        
        # Resolver trigger.description
        if 'trigger' in resolved_rule and 'description' in resolved_rule['trigger']:
            resolved_rule['trigger']['description'] = self._resolve_text_references(
                resolved_rule['trigger']['description']
            )
        
        # Resolver action_entities
        if 'constraint' in resolved_rule and 'action_entities' in resolved_rule['constraint']:
            resolved_rule['constraint']['action_entities'] = self.resolve_entity_references(
                resolved_rule['constraint']['action_entities']
            )
        
        # Resolver constraint.description
        if 'constraint' in resolved_rule and 'description' in resolved_rule['constraint']:
            resolved_rule['constraint']['description'] = self._resolve_text_references(
                resolved_rule['constraint']['description']
            )
        
        # Resolver linked_entities
        if 'linked_entities' in resolved_rule:
            resolved_rule['linked_entities'] = self.resolve_entity_references(
                resolved_rule['linked_entities']
            )
        
        # Resolver linked_relationships
        if 'linked_relationships' in resolved_rule:
            resolved_rule['linked_relationships'] = self.resolve_relationship_references(
                resolved_rule['linked_relationships']
            )
        
        # Resolver formal_if_then (CAMPO CRÍTICO)
        resolved_rule = self._resolve_formal_if_then(resolved_rule)
        
        # Resolver otros campos de texto que puedan contener IDs
        if 'applicability' in resolved_rule and isinstance(resolved_rule['applicability'], dict):
            # Resolver scope y actors en applicability
            for field in ['scope', 'actors']:
                if field in resolved_rule['applicability']:
                    if isinstance(resolved_rule['applicability'][field], str):
                        resolved_rule['applicability'][field] = self._resolve_text_references(
                            resolved_rule['applicability'][field]
                        )
                    elif isinstance(resolved_rule['applicability'][field], list):
                        resolved_rule['applicability'][field] = self.resolve_entity_references(
                            resolved_rule['applicability'][field]
                        )
        
        # Resolver explainability
        if 'explainability' in resolved_rule and isinstance(resolved_rule['explainability'], str):
            resolved_rule['explainability'] = self._resolve_text_references(
                resolved_rule['explainability']
            )
        
        return resolved_rule
    
    def validate_no_unresolved_ids(self, rule_data: Dict[str, Any]) -> bool:
        """Verifica que no queden IDs de entidades/relaciones sin resolver en campos de texto."""
        import re
        
        # Extraer todos los campos de texto relevantes
        text_fields = []
        
        # Trigger y constraint
        if 'trigger' in rule_data and 'description' in rule_data['trigger']:
            text_fields.append(rule_data['trigger']['description'])
        if 'constraint' in rule_data and 'description' in rule_data['constraint']:
            text_fields.append(rule_data['constraint']['description'])
        
        # Formal if_then
        if 'formal_if_then' in rule_data:
            fit = rule_data['formal_if_then']
            for field in ['if_condition', 'then_action', 'except_when']:
                if field in fit and fit[field]:
                    text_fields.append(str(fit[field]))
        
        # Applicability
        if 'applicability' in rule_data and isinstance(rule_data['applicability'], dict):
            for field in ['scope', 'actors']:
                if field in rule_data['applicability'] and rule_data['applicability'][field]:
                    text_fields.append(str(rule_data['applicability'][field]))
        
        # Explainability
        if 'explainability' in rule_data and rule_data['explainability']:
            text_fields.append(str(rule_data['explainability']))
        
        # Buscar IDs no resueltos
        all_text = ' '.join(text_fields)
        unresolved_entities = re.findall(r'E\d{3}', all_text)
        unresolved_relationships = re.findall(r'R\d{3}', all_text)
        
        if unresolved_entities or unresolved_relationships:
            logger.warning(f"⚠️ Unresolved IDs found in rule {rule_data.get('id', 'unknown')}:")
            if unresolved_entities:
                logger.warning(f"  Entities: {unresolved_entities}")
            if unresolved_relationships:
                logger.warning(f"  Relationships: {unresolved_relationships}")
            return False
        
        return True
    
    def is_complete_rule(self, rule_data: Dict[str, Any]) -> bool:
        """Verifica si una regla tiene todos los campos requeridos."""
        if not isinstance(rule_data, dict):
            return False
        
        required_fields = [
            'id', 'rule_type', 'modality', 'deontic_strength',
            'trigger', 'constraint', 'formal_if_then', 
            'applicability', 'severity', 'safety_critical', 'explainability'
        ]
        
        return all(field in rule_data for field in required_fields)
    
    def get_complete_rules(self) -> List[Dict[str, Any]]:
        """Retorna solo las reglas completas y con referencias resueltas."""
        complete_rules = []
        
        for rule_id, rule_data in self.rules.items():
            if self.is_complete_rule(rule_data):
                # Resolver referencias antes de retornar
                resolved_rule = self.resolve_rule_references(rule_data)
                
                # Validar que no queden IDs sin resolver
                if self.validate_no_unresolved_ids(resolved_rule):
                    complete_rules.append(resolved_rule)
                    logger.debug(f"✅ Regla completa y resuelta {rule_id}")
                else:
                    logger.warning(f"⚠️ Regla {rule_id} tiene IDs sin resolver, excluyendo")
            else:
                logger.debug(f"⚠️ Regla incompleta {rule_id}")
        
        return complete_rules
    
    def get_simple_references(self) -> List[Dict[str, Any]]:
        """Extrae referencias simples (como RULE001 en contexto)."""
        simple_refs = []
        
        for file_path in Path(self.processed_files[0]).parent.glob("pagina_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Buscar en contexto_reglas_seleccionadas
                for chunk in data.get('sentence_results', []):
                    context = chunk.get('context', {})
                    selected_rules = context.get('contexto_reglas_seleccionadas', [])
                    
                    for ref in selected_rules:
                        if isinstance(ref, dict) and 'id' in ref:
                            # Solo agregar si no está ya en rules
                            if ref['id'] not in self.rules:
                                simple_refs.append(ref)
                
            except Exception as e:
                logger.warning(f"⚠️ Error extrayendo referencias de {file_path}: {e}")
        
        return simple_refs
    
    def print_statistics(self):
        """Imprime estadísticas del procesamiento."""
        print(f"\n📊 Estadísticas de procesamiento:")
        print(f"  Archivos procesados: {len(self.processed_files)}")
        print(f"  Entidades: {self.stats['entities']}")
        print(f"  Relaciones: {self.stats['relationships']}")
        print(f"  Eventos: {self.stats['events']}")
        print(f"  Reglas completas: {len([r for r in self.rules.values() if self.is_complete_rule(r)])}")
        print(f"  Reglas simples: {len(self.get_simple_references())}")
        print(f"  Procedimientos: {self.stats['procedures']}")


class KEXFileProcessor:
    """Procesador principal para archivos KEX."""
    
    def __init__(self, kex_output_dir: str):
        """Inicializa procesador con directorio de salida KEX."""
        self.kex_dir = Path(kex_output_dir)
        self.accumulator = KEXDataAccumulator()
    
    def process_all_files(self, max_files: Optional[int] = None) -> KEXDataAccumulator:
        """Procesa todos los archivos pagina_N.json principales (no chunks ni errors)."""
        # Solo procesar archivos principales pagina_N.json, ignorar chunks y errors
        json_files = sorted(self.kex_dir.glob("pagina_[0-9]*.json"))
        # Filtrar para excluir chunks y errors
        json_files = [f for f in json_files if not (f.name.endswith('_chunks.json') or f.name.endswith('_errors.json'))]
        
        # Ordenar numéricamente por el número de página
        def get_page_number(filename):
            import re
            match = re.search(r'pagina_(\d+)\.json', filename.name)
            return int(match.group(1)) if match else 0
        
        json_files.sort(key=get_page_number)
        
        if max_files:
            json_files = json_files[:max_files]
        
        print(f"📂 Encontrados {len(json_files)} archivos principales")
        print(f"🔄 Procesando {len(json_files)} archivos...")
        print(f"📋 Archivos a procesar: {[f.name for f in json_files]}")
        
        for file_path in json_files:
            self.accumulator.process_file(file_path)
        
        self.accumulator.print_statistics()
        return self.accumulator
    
    def get_rules_for_compilation(self) -> List[Dict[str, Any]]:
        """Retorna reglas listas para compilación."""
        # Obtener reglas completas con referencias resueltas
        complete_rules = self.accumulator.get_complete_rules()
        
        print(f"\n🎯 Reglas para compilación:")
        print(f"  Completas: {len(complete_rules)}")
        
        return complete_rules


def main():
    """Función principal para testing del procesador."""
    logging.basicConfig(level=logging.INFO)
    
    # Directorio de salida KEX
    kex_dir = Path(__file__).parent.parent.parent / "Knowledge_Extractor" / "output" / "10_kex_output" / "ICAO Standard Phraseology(gpt-oss:20b)"
    
    if not kex_dir.exists():
        print(f"❌ Directorio KEX no encontrado: {kex_dir}")
        return
    
    processor = KEXFileProcessor(str(kex_dir))
    accumulator = processor.process_all_files(max_files=5)  # Limitar a 5 archivos para prueba
    
    # Mostrar algunas reglas procesadas
    rules = accumulator.get_rules_for_compilation()
    for i, rule in enumerate(rules[:3]):
        print(f"\n📋 Regla {i+1}: {rule.get('id', 'UNKNOWN')}")
        print(f"   Tipo: {rule.get('rule_type', 'unknown')}")
        print(f"   Modalidad: {rule.get('modality', 'unknown')}")
        if 'trigger' in rule and 'description' in rule['trigger']:
            print(f"   Trigger: {rule['trigger']['description'][:80]}...")


if __name__ == "__main__":
    main()
