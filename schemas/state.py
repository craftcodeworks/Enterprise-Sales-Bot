from typing import Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime

class ChatState(BaseModel):
    """
    Represents the persistent state of the conversation, 
    used to manage multi-turn queries and parameter collection.
    """
    
    # --- Parameter Collection State ---
    pending_query_id: str = ""
    last_query_template: str = ""
    missing_params: List[str] = Field(default_factory=list)
    collected_params: Dict[str, Any] = Field(default_factory=dict)
    original_question: str = ""
    
    # --- Loop Prevention ---
    param_collection_attempts: int = 0
    MAX_PARAM_ATTEMPTS: int = 3
    
    # --- Optional Parameters Tracking ---
    optional_params: List[str] = Field(default_factory=list)
    param_defaults: Dict[str, Any] = Field(default_factory=dict)

    # --- Result Recall State (For 'Show Table' follow-up) ---
    last_columns: List[str] = Field(default_factory=list)
    last_rows: List[List[Any]] = Field(default_factory=list)
    
    # ========================================
    # Conversation Memory Fields (NEW)
    # ========================================
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    last_successful_params: Dict[str, Any] = Field(default_factory=dict)
    last_query_context: str = ""
    last_query_id: str = ""
    last_executed_sql: str = ""  # Store last SQL for month-wise breakdown
    last_sort_direction: str = "DESC"  # Track sort direction for clarification questions
    last_filter_params: Dict[str, Any] = Field(default_factory=dict)  # Persistent filter params (state_id, business_category, etc.)
    
    MAX_HISTORY_TURNS: int = 5
    
    def increment_attempts(self) -> bool:
        """
        Increments attempt counter and returns True if should continue,
        False if max attempts reached.
        """
        self.param_collection_attempts += 1
        return self.param_collection_attempts < self.MAX_PARAM_ATTEMPTS
    
    # ========================================
    # Conversation Memory Methods (NEW)
    # ========================================
    
    def add_turn(self, role: str, message: str, params: Dict[str, Any] = None):
        """
        Adds a conversation turn (user or assistant) to history.
        Keeps only the last MAX_HISTORY_TURNS * 2 messages.
        """
        self.conversation_history.append({
            "role": role,
            "message": message,
            "params": params or {},
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last N turns (each turn = user + assistant)
        max_messages = self.MAX_HISTORY_TURNS * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
    
    def save_successful_query(self, params: Dict[str, Any], question: str, query_id: str):
        """
        Stores the parameters and context of the last successfully executed query.
        Called after SQL execution succeeds.
        """
        self.last_successful_params = params.copy()
        self.last_query_context = question
        self.last_query_id = query_id
        self.last_sort_direction = params.get('sort', 'DESC')
        
        # Extract and save filter params for persistent inheritance
        filter_param_names = {'state_id', 'cso_id', 'cluster_id', 'business_category'}
        self.last_filter_params = {k: v for k, v in params.items() if k in filter_param_names}
    
    def get_history_for_llm(self) -> str:
        """
        Formats conversation history as a string for LLM prompts.
        Returns last 5 turns in a readable format.
        """
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted = []
        for turn in self.conversation_history[-10:]:
            role = turn["role"].upper()
            msg = turn["message"][:300]  # Truncate long messages
            params = turn.get("params", {})
            
            if params:
                formatted.append(f"{role}: {msg}\n   [Params: {params}]")
            else:
                formatted.append(f"{role}: {msg}")
        
        return "\n".join(formatted)
    
    def has_context(self) -> bool:
        """Returns True if there is previous successful query context."""
        return bool(self.last_successful_params and self.last_query_context)
    
    # ========================================
    # State Management Methods
    # ========================================
    
    def clear_query_state(self):
        """Resets the state used for the current query/parameter collection."""
        self.pending_query_id = ""
        self.last_query_template = ""
        self.missing_params = []
        self.collected_params = {}
        self.original_question = ""
        self.param_collection_attempts = 0
        self.optional_params = []
        self.param_defaults = {}
    
    def clear_result_state(self):
        """Resets the state used for result recall (table follow-up)."""
        self.last_columns = []
        self.last_rows = []
        
    def clear_all(self):
        """Resets ALL conversational state INCLUDING history."""
        self.clear_query_state()
        self.clear_result_state()
        # Also clear conversation memory
        self.conversation_history = []
        self.last_successful_params = {}
        self.last_query_context = ""
        self.last_query_id = ""
        self.last_executed_sql = ""
        self.last_sort_direction = "DESC"
        self.last_filter_params = {}
    
    def apply_defaults_to_missing(self):
        """
        Applies default values to any missing optional parameters.
        Returns updated list of truly missing (required) parameters.
        """
        from datetime import datetime, timedelta
        
        truly_missing = []
        
        for param in self.missing_params:
            if param in self.optional_params and param in self.param_defaults:
                default_value = self.param_defaults[param]
                
                # Calculate date placeholders
                if isinstance(default_value, str) and default_value.startswith("__") and default_value.endswith("__"):
                    today = datetime.now()
                    
                    if default_value == "__LAST_MONTH_START__":
                        first_of_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
                        default_value = first_of_last_month.strftime("%Y-%m-%d")
                    
                    elif default_value == "__LAST_MONTH_END__":
                        first_of_this_month = today.replace(day=1)
                        last_of_last_month = first_of_this_month - timedelta(days=1)
                        default_value = last_of_last_month.strftime("%Y-%m-%d")
                    
                    elif default_value == "__THIS_MONTH_START__":
                        default_value = today.replace(day=1).strftime("%Y-%m-%d")
                    
                    elif default_value == "__THIS_MONTH_END__":
                        next_month = today.replace(day=28) + timedelta(days=4)
                        last_of_this_month = next_month - timedelta(days=next_month.day)
                        default_value = last_of_this_month.strftime("%Y-%m-%d")
                
                # Apply the calculated/regular default
                self.collected_params[param] = default_value
                print(f"[DEBUG] Applied default for '{param}': {default_value}")
            else:
                # Still required
                truly_missing.append(param)
        
        return truly_missing