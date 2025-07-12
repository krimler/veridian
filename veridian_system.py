import uuid
import random
import time
from enum import Enum

# --- Enums and Constants ---

class TranslationIntent(Enum):
    """Defines the various intents the Translation Agent can execute."""
    PII_MASKING = "PII_MASKING"
    OPTIMIZE_PROBLEM_TRANSFER = "OPTIMIZE_PROBLEM_TRANSFER"
    OPTIMIZE_SOLUTION_TRANSFER = "OPTIMIZE_SOLUTION_TRANSFER"
    APPLY_DIFFERENTIAL_PRIVACY = "APPLY_DIFFERENTIAL_PRIVACY"
    FACILITATE_NICHE_EXPERTISE_TRANSFER = "FACILITATE_NICHE_EXPERTISE_TRANSFER"
    ENHANCE_LEARNING_PRINCIPLES = "ENHANCE_LEARNING_PRINCIPLES"

# --- Client-Side Configuration ---

class ClientConfig:
    """
    Manages user-defined preferences and settings for AI interactions.
    This would be the UI-exposed configuration for "Veridian".
    """
    def __init__(self,
                 user_id: str,
                 privacy_preset: str = "Balanced", # Options: "Max Privacy", "Balanced", "Enhanced Personalization"
                 allow_anonymized_data_contribution: bool = True,
                 default_dp_epsilon: float = 1.0, # System/user default epsilon for Differential Privacy
                 allow_niche_expertise_sharing: bool = True,
                 prompt_for_sensitive_delegation: bool = True,
                 enable_on_device_learning: bool = True):
        
        self.user_id = user_id
        self.privacy_preset = privacy_preset
        self.allow_anonymized_data_contribution = allow_anonymized_data_contribution
        self.default_dp_epsilon = default_dp_epsilon
        self.allow_niche_expertise_sharing = allow_niche_expertise_sharing
        self.prompt_for_sensitive_delegation = prompt_for_sensitive_delegation
        self.enable_on_device_learning = enable_on_device_learning

        self._apply_preset()

    def _apply_preset(self):
        """Adjusts settings based on the chosen privacy preset."""
        if self.privacy_preset == "Max Privacy":
            self.allow_anonymized_data_contribution = False
            self.default_dp_epsilon = 0.1 # Very high privacy (low epsilon)
            self.allow_niche_expertise_sharing = False
            self.prompt_for_sensitive_delegation = True
            self.enable_on_device_learning = False
        elif self.privacy_preset == "Balanced":
            self.allow_anonymized_data_contribution = True
            self.default_dp_epsilon = 1.0 # Standard privacy
            self.allow_niche_expertise_sharing = True
            self.prompt_for_sensitive_delegation = True
            self.enable_on_device_learning = True
        elif self.privacy_preset == "Enhanced Personalization":
            self.allow_anonymized_data_contribution = True
            self.default_dp_epsilon = 5.0 # Lower privacy, higher utility (high epsilon)
            self.allow_niche_expertise_sharing = True
            self.prompt_for_sensitive_delegation = False # Less prompting for seamless experience
            self.enable_on_device_learning = True
        # Note: Custom settings can be set after preset application to override them.

    def __str__(self):
        return (f"ClientConfig(User: {self.user_id}, Preset: {self.privacy_preset}, "
                f"DP Epsilon: {self.default_dp_epsilon}, "
                f"Allow Data Contribution: {self.allow_anonymized_data_contribution}, "
                f"Enable Learning: {self.enable_on_device_learning})")

# --- Mock LLM and Helper Functions ---

def mock_llm_call(prompt: str, model_type: str = "generic", parameters: dict = None) -> str:
    """
    Simulates an LLM call with basic response generation.
    Can take parameters like epsilon to show its effect on perceived output.
    In a real system, this would be an actual API call (e.g., to OpenAI, Gemini).
    """
    parameters = parameters or {}
    response = f"LLM ({model_type}): Processed '{prompt[:50]}...' "

    if model_type == "privacy_sensitive" and parameters.get('privacy_budget_epsilon') is not None:
        epsilon = parameters['privacy_budget_epsilon']
        # Simulate how epsilon affects output 'fidelity' or 'noise'
        if epsilon < 0.5:
            response += f"with very high privacy (ε={epsilon:.2f}). Output is highly generalized/noisy. "
        elif epsilon < 2.0:
            response += f"with balanced privacy (ε={epsilon:.2f}). Output is generalized. "
        else:
            response += f"with lower privacy (ε={epsilon:.2f}). Output is more specific. "

    if "PII" in prompt:
        response += "PII was detected and handled. "
    if "private jet" in prompt or "cardiac surgery" in prompt:
        response += "Niche expertise invoked. "
    if "financial advice" in prompt:
        response += "Financial advice provided. "

    if parameters.get('detail_level') == 'high_precision':
        response += "High precision requested and applied. "

    response += "Result."
    return response

def _anonymize_text(text: str) -> str:
    """Simple PII masking for demonstration."""
    text = text.replace("John Doe", "[MASKED_NAME]")
    text = text.replace("john.doe@example.com", "[MASKED_EMAIL]")
    text = text.replace("123-456-7890", "[MASKED_PHONE]")
    text = text.replace("My address is 123 Main St", "My address is [MASKED_ADDRESS]")
    text = text.replace("my personal investment portfolio", "my [MASKED_FINANCIAL_ASSET] portfolio")
    text = text.replace("my medical history", "my [MASKED_HEALTH_INFO]")
    return text

# --- Agent Definitions ---

class LocalAgent:
    """
    Agent1: Resides on the client device.
    Acts as the user's intelligent proxy and privacy guardian.
    """
    def __init__(self, name: str, knowledge_base: list, client_config: ClientConfig):
        self.name = name
        self.knowledge_base = knowledge_base
        self.client_config = client_config
        print(f"[{self.name}]: Initialized with ClientConfig: {self.client_config}")

    def _detect_pii(self, query: str) -> bool:
        """Simulates PII detection based on keywords."""
        pii_keywords = ["john doe", "john.doe@example.com", "123-456-7890", "123 main st", "my address is",
                        "my personal investment portfolio", "my medical history"]
        return any(keyword in query.lower() for keyword in pii_keywords)

    def _assess_confidence(self, query: str) -> float:
        """Simulates confidence assessment based on internal knowledge."""
        # This is a mock; real NLU would be used
        if any(keyword in query.lower() for keyword in self.knowledge_base):
            return random.uniform(0.6, 0.9) # Can handle moderately well
        if "quantum entanglement" in query.lower() or "philosophical implications" in query.lower():
            return random.uniform(0.1, 0.3) # Very low confidence for complex topics
        return random.uniform(0.3, 0.5) # Low confidence, likely needs delegation

    def _should_delegate(self, query: str, confidence: float, features: dict) -> bool:
        """Decision logic for delegation."""
        if confidence < 0.5: # Always delegate if confidence is too low
            return True
        
        if features.get('contains_pii') and self.client_config.prompt_for_sensitive_delegation:
            # In a real UI, this would trigger a user prompt for explicit consent.
            # For this simulation, we assume the user implicitly allows if prompted.
            print(f"[{self.name}]: PII detected and user preference for sensitive delegation prompt is ON. Simulating user allowance.")
            return True
        
        if features.get('needs_niche_expertise') and self.client_config.allow_niche_expertise_sharing:
            print(f"[{self.name}]: Niche expertise needed and sharing is allowed.")
            return True
        
        return False # Default to local handling if no strong reason for delegation

    def _get_current_environment_context(self) -> dict:
        """Simulates getting real-time environmental context."""
        current_time_ist = time.strftime("%Y-%m-%d %H:%M:%S IST", time.localtime()) # Bengaluru time
        location_status = "Bengaluru, India (Presumed Home/Office)" 
        network_status = "online_wifi" 
        device_battery_level = random.randint(15, 100) # Simulate 15-100% battery

        # Introduce randomness for simulation
        if random.random() < 0.2: # 20% chance of public wifi
            location_status = "Public Wifi Zone (Simulated)"
            network_status = "online_public_wifi"
        if random.random() < 0.1: # 10% chance of cellular data
            network_status = "online_cellular"

        return {
            'network_status': network_status,
            'device_battery_level': device_battery_level,
            'local_time': current_time_ist,
            'device_location': location_status
        }
    
    def _assess_privacy_intent_and_suggest_epsilon(self, query: str, client_config: ClientConfig, environment_context: dict) -> float:
        """
        Assesses user's implicit/explicit privacy intent and suggests an epsilon value for Differential Privacy (DP).
        Lower epsilon = more privacy. Higher epsilon = less privacy/more utility.
        This dynamically overrides the default_dp_epsilon from client_config for the specific query,
        balancing user preferences with contextual risks and specific query sensitivity.
        """
        suggested_epsilon = client_config.default_dp_epsilon # Start with user's configured default

        # Rule 1: Highly sensitive topics trigger lower epsilon (more privacy)
        sensitive_keywords = ["medical history", "financial details", "personal trauma", "illegal activity", "private investment portfolio"]
        if any(keyword in query.lower() for keyword in sensitive_keywords):
            print(f"  [{self.name}]: Sensitive query content detected. Prioritizing privacy with stricter epsilon.")
            suggested_epsilon = min(suggested_epsilon, 0.5) # Cap at 0.5 (or lower) for high privacy

        # Rule 2: Explicit user consent for data sharing / low privacy concern (higher epsilon)
        if "share my data for research" in query.lower() or "I'm okay with data use" in query.lower() or client_config.privacy_preset == "Enhanced Personalization":
            print(f"  [{self.name}]: User/config explicitly indicates willingness to share or prioritize personalization. Boosting utility with higher epsilon.")
            suggested_epsilon = max(suggested_epsilon, 3.0) # Boost to at least 3.0 (or higher)

        # Rule 3: Environmental context impacting privacy needs
        if environment_context.get('network_status') == 'online_public_wifi':
            print(f"  [{self.name}]: Detected Public Wi-Fi. Applying a more conservative epsilon for caution.")
            suggested_epsilon = min(suggested_epsilon, 0.8) # Stricter privacy on public wifi

        # Rule 4: Device resource constraints (e.g., low battery) might influence trade-off
        if environment_context.get('device_battery_level') < 20:
             print(f"  [{self.name}]: Low device battery. Slightly relaxing epsilon to reduce intensive privacy processing/data transfer.")
             suggested_epsilon = max(suggested_epsilon, 1.5) # Slight utility boost to conserve resources

        # Ensure epsilon is always within a reasonable and valid range
        suggested_epsilon = max(0.01, min(10.0, suggested_epsilon)) # Epsilon typically ranges, with 0.01 being very strong privacy, 10.0 being very weak.

        return suggested_epsilon

    def generate_response(self, client_query: str):
        """Generates a response or decides to delegate based on internal logic and client config."""
        print(f"\n[{self.name}]: Received query: '{client_query}'")

        # 1. Feature Detection
        features = {}
        features['contains_pii'] = self._detect_pii(client_query)
        features['needs_niche_expertise'] = "private jet" in client_query.lower() or "cardiac surgery" in client_query.lower() or "financial advice" in client_query.lower()

        # 2. Confidence Assessment
        confidence = self._assess_confidence(client_query)
        
        # 3. Get Real-time Environment Context
        environment_context = self._get_current_environment_context()
        print(f"[{self.name}]: Current Environment: {environment_context}")

        # 4. Dynamic Epsilon Calculation (Key "Veridian" feature)
        suggested_dp_epsilon = self._assess_privacy_intent_and_suggest_epsilon(
            client_query, self.client_config, environment_context
        )
        
        # 5. Prepare Intent-Specific Parameters for Delegation
        # These parameters guide the Translation Agent's execution of specific intents
        delegation_intent_parameters = {
            TranslationIntent.APPLY_DIFFERENTIAL_PRIVACY.value: {
                'privacy_budget_epsilon': suggested_dp_epsilon
            },
            TranslationIntent.OPTIMIZE_PROBLEM_TRANSFER.value: {
                # Example: higher detail for critical medical queries
                'detail_level': 'high_precision' if "urgent" in client_query.lower() or "critical" in client_query.lower() else 'standard'
            }
            # Other intents can have their parameters defined here dynamically by LocalAgent
        }

        # 6. Decision on Delegation
        needs_delegation = self._should_delegate(client_query, confidence, features)

        # 7. Local Response (if not delegating)
        response = f"Local processing for: '{client_query}'. Confidence: {confidence:.2f}."
        reasoning = f"Detected PII: {features['contains_pii']}. Needs Niche: {features['needs_niche_expertise']}."

        return response, confidence, reasoning, features, needs_delegation, delegation_intent_parameters, environment_context


class TranslationAgent:
    """
    Agent2: Acts as a privacy and orchestration layer in the cloud.
    Receives delegated requests, applies policies, routes to Remote Agents,
    and distills learning insights. This is the "Veridian" orchestrator.
    """
    def __init__(self, name: str):
        self.name = name
        self._internal_state = {} # Stores temporary state for delegation context

    def _execute_intent_logic(self, intent: TranslationIntent, data: dict, internal_state: dict) -> dict:
        """Simulates executing the logic for a given intent based on its type."""
        print(f"  [{self.name}]: Executing Intent: {intent.name}")
        output = {}
        
        if intent == TranslationIntent.PII_MASKING:
            output['anonymized_query'] = _anonymize_text(data['original_query'])
            print(f"    - Original: '{data['original_query'][:30]}...' -> Anonymized: '{output['anonymized_query'][:30]}...'")
            
        elif intent == TranslationIntent.OPTIMIZE_PROBLEM_TRANSFER:
            detail_level = data.get('detail_level', 'standard')
            core_topic = data.get('core_topic', 'general topic') # This would be inferred by TranslationAgent
            problem_statement = f"Abstracted problem (Detail: {detail_level}) from '{data['original_query'][:40]}...'. Seeking expertise on {core_topic}."
            output['abstracted_problem'] = problem_statement
            print(f"    - Problem transferred: '{problem_statement[:50]}...'")

        elif intent == TranslationIntent.OPTIMIZE_SOLUTION_TRANSFER:
            # Here, the Translation Agent distills the Remote Agent's full response
            # into a more concise, actionable insight for the Local Agent.
            solution = data.get('remote_agent_response', 'No solution provided.')
            learning_insight = f"Distilled insight from solution: '{solution[:70]}...'. Focus on key facts and actionable steps for Local Agent to handle similar queries locally next time."
            output['distilled_solution'] = solution
            output['learning_insight_for_local_agent'] = learning_insight
            print(f"    - Solution distilled. Learning insight for Local Agent generated.")

        elif intent == TranslationIntent.APPLY_DIFFERENTIAL_PRIVACY:
            epsilon = data.get('privacy_budget_epsilon', 1.0) # Use the dynamically passed epsilon
            summary = data['raw_data_summary']
            
            # Simulate how different epsilon values affect the output of DP
            if epsilon < 0.5:
                dp_summary = f"Highly private (ε={epsilon:.2f}) trend: Very generalized. Data heavily noised for maximal privacy."
            elif epsilon < 2.0:
                dp_summary = f"Private (ε={epsilon:.2f}) aggregated insight: General patterns observable. Some noise applied."
            else:
                dp_summary = f"Lower privacy (ε={epsilon:.2f}) insight: More specific details. Minimal noise for utility."

            output['dp_summary'] = dp_summary
            print(f"    - DP applied: '{dp_summary}'")
            
        elif intent == TranslationIntent.FACILITATE_NICHE_EXPERTISE_TRANSFER:
            niche_topic = data.get('niche_topic', 'unspecified niche')
            # In a real system, this would involve routing to a specific API endpoint or model.
            output['remote_model_type'] = f"niche_expert_{niche_topic}"
            print(f"    - Routing to niche expert for '{niche_topic}'.")

        elif intent == TranslationIntent.ENHANCE_LEARNING_PRINCIPLES:
            successful_context = data['successful_context']
            # This is where the Translation Agent analyzes successful interactions
            # to extract generalizable rules or improved heuristics for the Local Agent.
            learning_principle = f"Learned principle from successful handling of '{successful_context[:50]}...': Always clarify user intent on [DOMAIN_SPECIFIC_ATTRIBUTE] for future queries."
            output['general_principle'] = learning_principle
            print(f"    - General learning principle extracted for Local Agent evolution.")

        return output

    def process_delegated_request(self, client_query: str, local_agent_context: dict, client_config: ClientConfig):
        """
        Processes a request delegated by the Local Agent, applying policies
        based on local_agent_context and the ClientConfig.
        This is the "Orchestration" layer of Veridian.
        """
        session_id = str(uuid.uuid4())
        self._internal_state[session_id] = {} # Initialize session-specific state
        
        print(f"\n[{self.name}]: Processing delegated request for '{client_query[:50]}...' (Session: {session_id[:8]})")
        print(f"  [{self.name}]: Local Agent Context Keys: {local_agent_context.keys()}")
        print(f"  [{self.name}]: Client Configuration (received by TA): {client_config}")

        original_query = client_query # The raw query before any PII masking
        processed_data = {'original_query': original_query}
        overall_success = True
        final_message = "Delegation successful. Response generated and insights distilled."

        # Extract parameters from Local Agent's delegation context
        local_agent_features = local_agent_context.get('local_agent_features', {})
        delegation_intent_parameters = local_agent_context.get('delegation_intent_parameters', {})
        environment_context = local_agent_context.get('environment_context', {})

        # --- Intent 1: PII Masking (High priority if PII is detected) ---
        query_for_remote_agent = original_query # Default to original if no PII or masking not applied
        if local_agent_features.get('contains_pii'):
            print(f"\n  [{self.name}]: PII detected by Local Agent. Applying PII_MASKING...")
            masking_output = self._execute_intent_logic(
                TranslationIntent.PII_MASKING,
                {'original_query': original_query},
                self._internal_state[session_id]
            )
            processed_data.update(masking_output)
            query_for_remote_agent = processed_data.get('anonymized_query', original_query)
        else:
            print(f"\n  [{self.name}]: No PII detected or PII masking not required for this query.")

        # --- Intent 2: Optimize Problem Transfer (Abstract the query for Remote Agent) ---
        print(f"\n  [{self.name}]: Optimizing Problem Transfer for Remote Agent...")
        problem_transfer_params = delegation_intent_parameters.get(TranslationIntent.OPTIMIZE_PROBLEM_TRANSFER.value, {})
        
        # Infer core topic for problem abstraction (mock implementation)
        core_topic = "general_query"
        if "travel" in original_query.lower(): core_topic = "travel_advice"
        elif "medical" in original_query.lower() or "health" in original_query.lower(): core_topic = "medical_advice"
        elif "financial" in original_query.lower() or "investment" in original_query.lower(): core_topic = "financial_advice"

        problem_transfer_output = self._execute_intent_logic(
            TranslationIntent.OPTIMIZE_PROBLEM_TRANSFER,
            {'original_query': query_for_remote_agent, 'core_topic': core_topic, **problem_transfer_params},
            self._internal_state[session_id]
        )
        processed_data.update(problem_transfer_output)
        abstracted_query = processed_data['abstracted_problem']

        # --- Intent 3: Facilitate Niche Expertise Transfer (Route to specialized Remote Agents) ---
        remote_model_type = "generic_llm" # Default model type
        if local_agent_features.get('needs_niche_expertise') and \
           client_config.allow_niche_expertise_sharing:
            print(f"\n  [{self.name}]: Niche expertise needed and allowed by client. Facilitating transfer...")
            niche_topic_identified = "unknown"
            if "cardiac surgery" in original_query.lower(): niche_topic_identified = "medical_specialist"
            elif "private jet" in original_query.lower(): niche_topic_identified = "luxury_travel_specialist"
            elif "financial advice" in original_query.lower() or "investment" in original_query.lower(): niche_topic_identified = "financial_expert"

            niche_transfer_output = self._execute_intent_logic(
                TranslationIntent.FACILITATE_NICHE_EXPERTISE_TRANSFER,
                {'niche_topic': niche_topic_identified},
                self._internal_state[session_id]
            )
            remote_model_type = niche_transfer_output['remote_model_type']
        else:
            print(f"\n  [{self.name}]: Niche expertise not needed or not allowed by client settings. Using generic model.")

        # --- Route to Remote Agent ---
        print(f"\n  [{self.name}]: Routing abstracted problem to Remote Agent ({remote_model_type})...")
        remote_agent_response = RemoteAgent.get_instance().process_request(abstracted_query, model_type=remote_model_type)
        processed_data['remote_agent_raw_response'] = remote_agent_response
        print(f"  [{self.name}]: Remote Agent responded: '{remote_agent_response}'")

        # --- Intent 4: Optimize Solution Transfer (Distill for Local Agent to understand) ---
        print(f"\n  [{self.name}]: Optimizing Solution Transfer for Local Agent's comprehension...")
        solution_transfer_output = self._execute_intent_logic(
            TranslationIntent.OPTIMIZE_SOLUTION_TRANSFER,
            {'remote_agent_response': remote_agent_response},
            self._internal_state[session_id]
        )
        processed_data.update(solution_transfer_output)
        
        # --- Intent 5: Apply Differential Privacy (if allowed by client_config and Local Agent hints) ---
        if client_config.allow_anonymized_data_contribution:
            print(f"\n  [{self.name}]: Client allows anonymized data contribution. Applying Differential Privacy...")
            dp_params = delegation_intent_parameters.get(TranslationIntent.APPLY_DIFFERENTIAL_PRIVACY.value, {})
            # Use Local Agent's suggested epsilon, else client_config's default
            dp_epsilon_to_use = dp_params.get('privacy_budget_epsilon', client_config.default_dp_epsilon)
            
            dp_output = self._execute_intent_logic(
                TranslationIntent.APPLY_DIFFERENTIAL_PRIVACY,
                {'raw_data_summary': f"Interaction log for: {original_query[:50]}...",
                 'privacy_budget_epsilon': dp_epsilon_to_use}, # Use the dynamically determined epsilon here
                self._internal_state[session_id]
            )
            processed_data.update(dp_output)
            print(f"  [{self.name}]: DP Epsilon used for this contribution: {dp_epsilon_to_use:.2f}")
        else:
            print(f"\n  [{self.name}]: Client DOES NOT allow anonymized data contribution. Skipping DP for this interaction.")

        # --- Intent 6: Enhance Learning Principles (if enabled by client_config) ---
        if client_config.enable_on_device_learning:
            print(f"\n  [{self.name}]: Client enables on-device learning. Enhancing principles for Local Agent...")
            learning_output = self._execute_intent_logic(
                TranslationIntent.ENHANCE_LEARNING_PRINCIPLES,
                {'successful_context': original_query}, # Simplified context of success
                self._internal_state[session_id]
            )
            processed_data.update(learning_output)
        else:
            print(f"\n  [{self.name}]: Client DOES NOT enable on-device learning. Skipping principle enhancement.")

        # Cleanup internal state for this session
        del self._internal_state[session_id]

        return processed_data, overall_success, final_message


class RemoteAgent:
    """
    Agent3: The powerful, potentially general-purpose or specialized AI model
    (e.g., a large LLM) in the cloud. It receives cleaned/abstracted queries from the Translation Agent.
    This is the "Expert" layer of Veridian.
    """
    _instance = None # Implementing a simple Singleton pattern for demonstration

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RemoteAgent, cls).__new__(cls)
            cls._instance.name = "RemoteAgent (LLM)"
            print(f"[{cls._instance.name}]: Initialized (Singleton).")
        return cls._instance

    def process_request(self, abstracted_query: str, model_type: str = "generic_llm") -> str:
        """Processes the abstracted query and returns a detailed response."""
        print(f"\n[{self.name}]: Received request from Translation Agent ({model_type}): '{abstracted_query[:70]}...'")
        # In a real system, this would be the actual call to the powerful cloud LLM or a specialized service.
        return mock_llm_call(abstracted_query, model_type=model_type, parameters={'is_remote': True, 'niche_requested': model_type})

# --- Orchestration ---

def run_scenario_delegation_with_config(client_query: str, client_config: ClientConfig):
    """
    Orchestrates the interaction between Local Agent, Translation Agent, and Remote Agent
    with the provided client-side configuration for the "Veridian" system.
    """
    print(f"\n--- Running Scenario for Client Query: '{client_query}' ---")
    
    # Initialize Agents with the specific client_config
    local_agent_kb = ["general knowledge", "basic facts", "common questions", "weather", "news",
                      "travel", "doctor", "financial", "investment"] # Expanded KB for Local Agent
    local_agent = LocalAgent("LocalAgent (Agent1)", local_agent_kb, client_config)
    translation_agent = TranslationAgent("TranslationAgent (Agent2)")
    remote_agent = RemoteAgent.get_instance() # Get the singleton instance

    # Local Agent attempts to respond, assesses confidence, and decides on delegation
    local_agent_response, local_agent_confidence, local_agent_reasoning, local_agent_features, needs_delegation, local_agent_intent_params, local_agent_environment_context = \
        local_agent.generate_response(client_query)

    print(f"\n[{local_agent.name}]: Initial Local Response: '{local_agent_response}'")
    print(f"[{local_agent.name}]: Local Confidence: {local_agent_confidence:.2f}")
    print(f"[{local_agent.name}]: Local Reasoning: {local_agent_reasoning}")

    if needs_delegation:
        print(f"\n[{local_agent.name}]: Decision: Delegating to Translation Agent based on confidence/features/config.")
        
        # Prepare the context object for the Translation Agent
        local_agent_context_for_delegation = {
            'local_agent_attempted_response': local_agent_response,
            'local_agent_confidence_score': local_agent_confidence,
            'local_agent_reasoning_trace': local_agent_reasoning,
            'client_query_with_pii': client_query, # Pass raw query for TA to perform masking if needed
            'local_agent_features': local_agent_features, # Features detected by Local Agent
            'delegation_intent_parameters': local_agent_intent_params, # Dynamic parameters for TA intents
            'environment_context': local_agent_environment_context # Real-time environment context
        }
        
        # Translation Agent processes the delegated request
        final_processed_data, overall_success, final_message = \
            translation_agent.process_delegated_request(
                client_query, # Raw query is also passed for full context to TA
                local_agent_context_for_delegation,
                client_config # The full client_config is passed to TA for policy decisions
            )
        
        print(f"\n--- Veridian Delegation Summary for Query: '{client_query}' ---")
        print(f"Overall Delegation Success: {overall_success}")
        print(f"Final Message: {final_message}")
        print(f"Key Processed Data from Translation Agent:")
        for key, value in final_processed_data.items():
            if not key.startswith('original_query'): # Hide original query for brevity in final print
                print(f"  - {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}") # Truncate long strings
            
        # Local Agent receives distilled insight and can "evolve"
        if client_config.enable_on_device_learning and 'learning_insight_for_local_agent' in final_processed_data:
            insight = final_processed_data['learning_insight_for_local_agent']
            print(f"\n[{local_agent.name}]: Received Learning Insight from Translation Agent: '{insight}'")
            # In a real system, LocalAgent would update its model/rules based on this insight.
            # For this demo, we simulate the update by adding it to its knowledge base.
            local_agent.knowledge_base.append(insight)
            print(f"[{local_agent.name}]: Local Agent's knowledge base updated (simulated).")
            
        if client_config.allow_anonymized_data_contribution and 'dp_summary' in final_processed_data:
            print(f"[{local_agent.name}]: DP Summary for collective learning: '{final_processed_data['dp_summary']}'")

    else:
        print(f"\n[{local_agent.name}]: Decision: Handled locally. No delegation needed.")
        print(f"[{local_agent.name}]: Final Local Response: {local_agent_response}")

# --- Scenarios to Demonstrate "Veridian" ---

if __name__ == "__main__":
    print("Welcome to the Veridian System Simulation!")

    # Scenario 1: Default Balanced Privacy with PII & General Query
    print("\n\n===== SCENARIO 1: BALANCED PRIVACY (PII + General Query) =====")
    user1_config = ClientConfig(user_id="user_john_doe", privacy_preset="Balanced")
    run_scenario_delegation_with_config(
        "Hi, my name is John Doe and my email is john.doe@example.com. Can you help me find a general doctor for a checkup?",
        user1_config
    )
    time.sleep(1) # Pause for readability

    # Scenario 2: Max Privacy with Highly Sensitive Query
    print("\n\n===== SCENARIO 2: MAX PRIVACY (Highly Sensitive + Niche) =====")
    user2_config = ClientConfig(user_id="user_jane_smith", privacy_preset="Max Privacy")
    run_scenario_delegation_with_config(
        "I need urgent financial advice on my personal investment portfolio. My address is 123 Main St. I also want to contribute data for research.", # Intentional conflict for demo
        user2_config
    )
    time.sleep(1)

    # Scenario 3: Enhanced Personalization with Niche Expertise
    print("\n\n===== SCENARIO 3: ENHANCED PERSONALIZATION (Niche Expertise) =====")
    user3_config = ClientConfig(user_id="user_luxury_traveler", privacy_preset="Enhanced Personalization")
    run_scenario_delegation_with_config(
        "Can you help me book a private jet to the Maldives next week? I'm okay with data use for better service.",
        user3_config
    )
    time.sleep(1)

    # Scenario 4: Dynamic Epsilon Override (Balanced preset, but sensitive query boosts privacy)
    print("\n\n===== SCENARIO 4: DYNAMIC EPSILON OVERRIDE (Sensitive Query) =====")
    user4_config = ClientConfig(user_id="user_override", privacy_preset="Balanced")
    run_scenario_delegation_with_config(
        "I have a highly sensitive medical history question regarding a cardiac surgery I had. Please be extremely private with this.",
        user4_config
    )
    time.sleep(1)

    # Scenario 5: Local Handling (High confidence, no PII/Niche, should NOT delegate)
    print("\n\n===== SCENARIO 5: LOCAL HANDLING (High Confidence, No Delegation) =====")
    user5_config = ClientConfig(user_id="user_local", privacy_preset="Balanced")
    # This query is designed to be handled locally given the expanded KB
    run_scenario_delegation_with_config(
        "What's the current weather in Bengaluru?",
        user5_config
    )
    time.sleep(1)
    
    # Scenario 6: Local Agent Learning Effect (Simulated)
    print("\n\n===== SCENARIO 6: LOCAL AGENT LEARNING EFFECT (SIMULATED) =====")
    user6_config = ClientConfig(user_id="user_learning", privacy_preset="Balanced")
    # Initial query, Local Agent may delegate due to lower confidence on 'complex problem'
    run_scenario_delegation_with_config(
        "How do I optimize my daily schedule for maximum productivity?",
        user6_config
    )
    time.sleep(1)
    # After a simulated learning cycle, Local Agent might handle similar query better
    # (The KB update is simplified; real learning would be more sophisticated)
    print("\n--- Running a similar query after simulated learning ---")
    run_scenario_delegation_with_config(
        "Can you suggest ways to improve my personal productivity?",
        user6_config
    )
