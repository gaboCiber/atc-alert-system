"""
Pipeline End-to-End integrando ASR + KEX + Alert System.

Este es el pipeline completo que:
1. Recibe audio ATC
2. Transcribe con ASR
3. Adapta a ParsedInstruction
4. Ejecuta el pipeline de 8 pasos
5. Genera alertas
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass

# ASR imports
from ASR.transcription.base import TranscriptionResult
from ASR.transcription.pipeline import TranscriptionPipeline

# KEX imports
from Knowledge_Extractor import Rule

# Alert System imports
from Alert_System.pipeline.alert_pipeline import AlertPipeline, PipelineResult
from Alert_System.models.traffic_state import TrafficState
from Alert_System.models.instruction import Speaker
from Alert_System.rule_engine.engine import RuleEngine
from Alert_System.core.state_manager import StateManager

# Integration imports
from .asr_adapter import ASRAdapter, TranscriptionContext
from .kex_adapter import KEXAdapter, KnowledgeContext


@dataclass
class EndToEndResult:
    """Resultado completo del pipeline end-to-end."""
    # Entrada
    audio_path: Optional[str]
    raw_transcription: Optional[str]
    
    # Procesamiento
    transcription_result: Optional[TranscriptionResult]
    parsed_instruction: Optional[Any]  # ParsedInstruction
    
    # Pipeline de alertas
    alert_pipeline_result: Optional[PipelineResult]
    
    # Salida
    alerts_generated: List[Any]  # List[Alert]
    decision: str  # COMMIT / ROLLBACK / PENDING
    
    # Metadata
    processing_time_ms: Optional[float]
    errors: List[str]
    
    def has_alerts(self) -> bool:
        """Indica si se generaron alertas."""
        return len(self.alerts_generated) > 0
    
    def is_critical(self) -> bool:
        """Indica si hay alerta crítica."""
        from Alert_System.models.alert import AlertSeverity
        for alert in self.alerts_generated:
            if hasattr(alert, 'severity') and alert.severity == AlertSeverity.CRITICAL:
                return True
        return False


class EndToEndPipeline:
    """
    Pipeline completo integrando todos los componentes.
    
    Flujo:
    1. Audio → ASR → TranscriptionResult
    2. TranscriptionResult → ASRAdapter → ParsedInstruction
    3. ParsedInstruction → AlertPipeline → Alertas
    """
    
    def __init__(
        self,
        asr_pipeline: Optional[TranscriptionPipeline] = None,
        asr_adapter: Optional[ASRAdapter] = None,
        kex_adapter: Optional[KEXAdapter] = None,
        alert_pipeline: Optional[AlertPipeline] = None,
        initial_state: Optional[TrafficState] = None,
    ):
        """
        Inicializa el pipeline end-to-end.
        
        Args:
            asr_pipeline: Pipeline de transcripción ASR
            asr_adapter: Adaptador ASR
            kex_adapter: Adaptador KEX
            alert_pipeline: Pipeline de alertas
            initial_state: Estado inicial del tráfico
        """
        self.asr_pipeline = asr_pipeline
        self.asr_adapter = asr_adapter or ASRAdapter()
        self.kex_adapter = kex_adapter or KEXAdapter()
        
        # Crear alert pipeline si no se proporcionó
        if alert_pipeline is None:
            state_manager = StateManager(
                initial_state or TrafficState(sector_id="DEFAULT")
            )
            rule_engine = RuleEngine()
            self.alert_pipeline = AlertPipeline(state_manager, rule_engine)
        else:
            self.alert_pipeline = alert_pipeline
        
        # Almacenar reglas del KEX
        self._knowledge_rules: List[Rule] = []
    
    def load_knowledge(self, rules: List[Rule]) -> None:
        """
        Carga reglas del KEX en el Rule Engine.
        
        Args:
            rules: Reglas extraídas por el KEX
        """
        self._knowledge_rules = rules
        
        # Adaptar reglas a evaluadores
        evaluators = self.kex_adapter.adapt_rules(rules)
        
        # Almacenar evaluadores para uso en procesamiento
        self._condition_evaluators = evaluators
        
        print(f"✅ Cargadas {len(evaluators)} reglas del KEX")
    
    def process_audio(
        self,
        audio_path: Union[str, Path],
        speaker: Speaker = Speaker.ATCO,
        enable_projection: bool = True,
    ) -> EndToEndResult:
        """
        Procesa un archivo de audio completo.
        
        Args:
            audio_path: Ruta al archivo de audio
            speaker: Quién habla en el audio
            enable_projection: Si se debe usar proyección de estado
            
        Returns:
            EndToEndResult con todo el procesamiento
        """
        import time
        start_time = time.time()
        
        errors = []
        
        # Step 1: Transcribir
        transcription_result = None
        raw_transcription = None
        
        if self.asr_pipeline is None:
            errors.append("ASR pipeline not configured")
            return EndToEndResult(
                audio_path=str(audio_path),
                raw_transcription=None,
                transcription_result=None,
                parsed_instruction=None,
                alert_pipeline_result=None,
                alerts_generated=[],
                decision="ERROR",
                processing_time_ms=None,
                errors=errors,
            )
        
        try:
            # Transcribir audio
            results = self.asr_pipeline.run([audio_path], "/tmp/dummy_output.json")
            if results:
                transcription_result = results[0]
                raw_transcription = transcription_result.text
        except Exception as e:
            errors.append(f"Transcription error: {e}")
        
        # Step 2: Adaptar a ParsedInstruction
        parsed_instruction = None
        if transcription_result:
            try:
                parsed_instruction = self.asr_adapter.adapt(transcription_result, speaker)
            except Exception as e:
                errors.append(f"ASR adaptation error: {e}")
        
        # Step 3: Ejecutar pipeline de alertas
        alert_result = None
        alerts = []
        decision = "PENDING"
        
        if parsed_instruction:
            try:
                alert_result = self.alert_pipeline.process_instruction(
                    parsed_instruction.raw_text,
                    enable_state_projection=enable_projection,
                )
                
                # Extraer alertas
                if alert_result:
                    alerts = alert_result.alerts
                    decision = alert_result.decision or "PENDING"
            except Exception as e:
                errors.append(f"Alert pipeline error: {e}")
        
        # Calcular tiempo
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EndToEndResult(
            audio_path=str(audio_path),
            raw_transcription=raw_transcription,
            transcription_result=transcription_result,
            parsed_instruction=parsed_instruction,
            alert_pipeline_result=alert_result,
            alerts_generated=alerts,
            decision=decision,
            processing_time_ms=processing_time_ms,
            errors=errors,
        )
    
    def process_text(
        self,
        text: str,
        speaker: Speaker = Speaker.ATCO,
        enable_projection: bool = True,
    ) -> EndToEndResult:
        """
        Procesa texto directamente (sin ASR).
        
        Útil para testing o cuando ya se tiene la transcripción.
        
        Args:
            text: Texto de la instrucción ATC
            speaker: Quién habla
            enable_projection: Si se debe usar proyección
            
        Returns:
            EndToEndResult
        """
        import time
        start_time = time.time()
        
        # Crear TranscriptionResult sintético
        synthetic_result = TranscriptionResult(
            text=text,
            file_path="synthetic",
            model_name="text_input",
            confidence=1.0,
        )
        
        # Adaptar
        parsed_instruction = self.asr_adapter.adapt(synthetic_result, speaker)
        
        # Ejecutar pipeline
        alert_result = self.alert_pipeline.process_instruction(
            text,
            pre_parsed=parsed_instruction,
        )
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return EndToEndResult(
            audio_path=None,
            raw_transcription=text,
            transcription_result=synthetic_result,
            parsed_instruction=parsed_instruction,
            alert_pipeline_result=alert_result,
            alerts_generated=alert_result.alerts_generated if alert_result else [],
            decision=alert_result.final_decision if alert_result else "PENDING",
            processing_time_ms=processing_time_ms,
            errors=[],
        )
    
    def get_current_state(self) -> TrafficState:
        """Obtiene el estado actual del tráfico."""
        return self.alert_pipeline.state_manager.current_state
    
    def update_traffic_state(self, state: TrafficState) -> None:
        """Actualiza el estado del tráfico."""
        self.alert_pipeline.state_manager.update_state(state)
    
    def commit_changes(self) -> bool:
        """Commit de cambios pendientes."""
        return self.alert_pipeline.state_manager.commit()
    
    def rollback_changes(self) -> bool:
        """Rollback de cambios. Si no hay transacción pendiente, deshace el último commit."""
        sm = self.alert_pipeline.state_manager
        if sm.has_pending_transaction():
            return sm.rollback()
        else:
            # No hay transacción pendiente, deshacer el último commit
            return sm.undo_last_commit()
