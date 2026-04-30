"""State Manager - gestión transaccional del estado del tráfico aéreo."""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from Alert_System.models.traffic_state import TrafficState
from .state_projection import ProjectedState


@dataclass
class StateTransaction:
    """Representa una transacción de estado."""
    
    transaction_id: str
    timestamp: datetime
    projected_state: ProjectedState
    status: str = "PENDING"  # PENDING, COMMITTED, ROLLBACK
    
    # Decisión del ATCO
    atco_decision: Optional[str] = None  # "COMMIT", "ROLLBACK"
    atco_reason: Optional[str] = None
    decision_timestamp: Optional[datetime] = None
    
    # Alertas asociadas
    has_alerts: bool = False
    alert_ids: List[str] = field(default_factory=list)
    
    # Override
    force_committed: bool = False


class StateManager:
    """
    Gestiona el estado del tráfico aéreo con soporte para commit/rollback.    
    - Mantiene el estado actual (real)
    - Permite crear proyecciones (simulaciones)
    - COMMIT: Aplica la proyección al estado real
    - ROLLBACK: Descarta la proyección
    """
    
    def __init__(self, initial_state: Optional[TrafficState] = None):
        """
        Inicializa el State Manager.
        
        Args:
            initial_state: Estado inicial del tráfico
        """
        self._state = initial_state or TrafficState(sector_id="DEFAULT")
        self._state_history: List[TrafficState] = []
        self._transactions: Dict[str, StateTransaction] = {}
        self._pending_transaction: Optional[StateTransaction] = None
        self._max_history = 10
    
    @property
    def current_state(self) -> TrafficState:
        """Retorna el estado actual del tráfico."""
        return self._state
    
    @property
    def sector_id(self) -> str:
        """ID del sector actual."""
        return self._state.sector_id
    
    def update_state(self, new_state: TrafficState) -> None:
        """
        Actualiza directamente el estado (para inicialización o sincronización).
        
        Este método no crea transacción, actualiza el estado directamente.
        """
        # Guardar en historial
        self._save_to_history()
        
        self._state = new_state
    
    def propose_change(
        self,
        projected_state: ProjectedState,
        transaction_id: Optional[str] = None,
    ) -> StateTransaction:
        """
        Propone un cambio al estado basado en una proyección.
        
        Crea una transacción pendiente que debe ser confirmada (COMMIT) o rechazada (ROLLBACK).
        
        Args:
            projected_state: Estado proyectado a aplicar
            transaction_id: ID opcional de la transacción
            
        Returns:
            StateTransaction creada
        """
        if transaction_id is None:
            # Generar ID único usando UUID
            transaction_id = f"TXN_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        has_alerts = projected_state.has_conflicts() if hasattr(projected_state, 'has_conflicts') else False
        
        transaction = StateTransaction(
            transaction_id=transaction_id,
            timestamp=datetime.utcnow(),
            projected_state=projected_state,
            status="PENDING",
            has_alerts=has_alerts,
        )
        
        self._transactions[transaction_id] = transaction
        self._pending_transaction = transaction
        
        return transaction
    
    def commit(
        self,
        transaction_id: Optional[str] = None,
        force: bool = False,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Aplica (commit) una proyección al estado real.
        
        Args:
            transaction_id: ID de la transacción (usa la pendiente si no se especifica)
            force: Si es True, aplica aunque haya alertas (override del ATCO)
            reason: Razón del override (si force=True)
            
        Returns:
            True si se aplicó exitosamente
        """
        transaction = self._get_transaction(transaction_id)
        if not transaction:
            return False
        
        # Verificar si hay alertas y no es force
        if transaction.has_alerts and not force:
            return False
        
        # Guardar estado actual en historial
        self._save_to_history()
        
        # Aplicar el estado proyectado
        self._state = transaction.projected_state.traffic_state
        
        # Actualizar transacción
        transaction.status = "COMMITTED"
        transaction.atco_decision = "COMMIT"
        transaction.atco_reason = reason
        transaction.decision_timestamp = datetime.utcnow()
        transaction.force_committed = force
        
        self._pending_transaction = None
        
        return True
    
    def rollback(
        self,
        transaction_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Rechaza (rollback) una proyección, descartándola.
        
        El estado real no se modifica.
        
        Args:
            transaction_id: ID de la transacción (usa la pendiente si no se especifica)
            reason: Razón del rollback
            
        Returns:
            True si se rechazó exitosamente
        """
        transaction = self._get_transaction(transaction_id)
        if not transaction:
            return False
        
        # Actualizar transacción
        transaction.status = "ROLLBACK"
        transaction.atco_decision = "ROLLBACK"
        transaction.atco_reason = reason
        transaction.decision_timestamp = datetime.utcnow()
        
        self._pending_transaction = None
        
        return True
    
    def get_pending_transaction(self) -> Optional[StateTransaction]:
        """Retorna la transacción pendiente actual."""
        return self._pending_transaction
    
    def has_pending_transaction(self) -> bool:
        """¿Hay una transacción pendiente?"""
        return self._pending_transaction is not None
    
    def get_transaction_history(self) -> List[StateTransaction]:
        """Retorna historial de todas las transacciones."""
        return list(self._transactions.values())
    
    def undo_last_commit(self) -> bool:
        """
        Deshace el último commit restaurando el estado anterior.
        
        Returns:
            True si se restauró exitosamente
        """
        if len(self._state_history) == 0:
            return False
        
        self._state = self._state_history.pop()
        return True
    
    def _get_transaction(self, transaction_id: Optional[str]) -> Optional[StateTransaction]:
        """Obtiene una transacción por ID o la pendiente."""
        if transaction_id:
            return self._transactions.get(transaction_id)
        return self._pending_transaction
    
    def _save_to_history(self) -> None:
        """Guarda el estado actual en el historial."""
        # Copia profunda para evitar modificaciones
        history_copy = deepcopy(self._state)
        self._state_history.append(history_copy)
        
        # Limitar historial
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)
    
    def get_state_at_timestamp(self, timestamp: datetime) -> Optional[TrafficState]:
        """
        Busca el estado más cercano a un timestamp.
        
        Args:
            timestamp: Timestamp a buscar
            
        Returns:
            TrafficState más cercano o None
        """
        closest = None
        closest_diff = None
        
        for state in self._state_history:
            diff = abs((state.timestamp - timestamp).total_seconds())
            if closest_diff is None or diff < closest_diff:
                closest_diff = diff
                closest = state
        
        return closest


class Transaction:
    """
    Context manager para transacciones.
    
    Uso:
        with Transaction(state_manager, projected_state) as txn:
            # Si no hay excepciones, hace commit automáticamente
            # Si hay excepciones, hace rollback
    """
    
    def __init__(
        self,
        state_manager: StateManager,
        projected_state: ProjectedState,
        auto_commit: bool = False,
    ):
        """
        Inicializa el context manager.
        
        Args:
            state_manager: StateManager a usar
            projected_state: Proyección a aplicar
            auto_commit: Si True, hace commit automáticamente al salir
        """
        self.state_manager = state_manager
        self.projected_state = projected_state
        self.auto_commit = auto_commit
        self.transaction: Optional[StateTransaction] = None
        self.committed = False
    
    def __enter__(self) -> StateTransaction:
        """Entra al contexto y crea la transacción."""
        self.transaction = self.state_manager.propose_change(self.projected_state)
        return self.transaction
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Sale del contexto.
        
        - Si hubo excepción: hace rollback
        - Si no hubo excepción y auto_commit=True: hace commit
        - Si no hubo excepción y auto_commit=False: deja pendiente
        """
        if exc_type is not None:
            # Hubo excepción, hacer rollback
            self.state_manager.rollback(
                self.transaction.transaction_id,
                reason=f"Exception: {exc_val}"
            )
            return False  # No suprimir la excepción
        
        if self.auto_commit:
            # Commit automático
            self.committed = self.state_manager.commit(
                self.transaction.transaction_id
            )
        
        return True
