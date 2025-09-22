"""
SLM KeyDoor Agent - Simplified agent that works with KeyDoorSLMEnv
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional
import ollama
import json

class SLMKeyDoorAgent:
    """
    Simplified SLM KeyDoor agent that works directly with KeyDoorSLMEnv.
    
    This agent uses the environment's built-in spatial blueprint and navigation context
    to generate optimal navigation plans and execute them step by step.
    """
    
    def __init__(self, model_name: str = "llama3.2:3b", temperature: float = 0.1):
        """
        Initialize the SLM KeyDoor Agent.
        
        Args:
            model_name: Ollama model to use
            temperature: Generation temperature
        """
        self.model_name = model_name
        self.temperature = temperature
        self.client = ollama.Client()
        
        # Performance tracking
        self.total_requests = 0
        self.total_time = 0.0
        self.successful_requests = 0
        
        # Navigation state
        self.current_navigation_plan = None
        self.current_step_index = 0
        
        self._verify_model()
        logging.info(f"SLM KeyDoor Agent initialized with model: {model_name}")
    
    def _verify_model(self):
        """Verify that the specified model is available."""
        try:
            models = self.client.list()
            # Handle different response formats
            if hasattr(models, 'models'):
                available_models = [m.name for m in models.models]
            elif isinstance(models, dict) and 'models' in models:
                available_models = [m.get('name', str(m)) for m in models['models']]
            else:
                available_models = [str(m) for m in models]
            
            if self.model_name not in available_models:
                logging.warning(f"Model {self.model_name} not found. Available: {available_models}")
                if available_models:
                    self.model_name = available_models[0]
                    logging.info(f"Using available model: {self.model_name}")
                else:
                    raise ValueError("No Ollama models available.")
        except Exception as e:
            logging.error(f"Error verifying model: {e}")
            # Continue anyway - the model might still work
            logging.info("Continuing with specified model despite verification error")
    
    def act(self, env) -> int:
        """
        Determine the next action for the agent.
        
        Args:
            env: KeyDoorSLMEnv instance
            
        Returns:
            Action code (0-5)
        """
        # Check if we need a new navigation plan
        if self._is_navigation_complete():
            # Generate new navigation plan
            navigation_context = env.get_navigation_context()
            self.current_navigation_plan = self._plan_navigation(navigation_context)
            self.current_step_index = 0
            logging.info(f"Generated new navigation plan with {len(self.current_navigation_plan.get('steps', []))} steps")
        
        # Execute next step from current plan
        if self.current_navigation_plan and 'steps' in self.current_navigation_plan:
            steps = self.current_navigation_plan['steps']
            if self.current_step_index < len(steps):
                current_step = steps[self.current_step_index]
                action_str = current_step.get('action', 'up')
                self.current_step_index += 1
                
                # Convert action string to action code
                action_code = self._action_string_to_code(action_str)
                logging.debug(f"Executing step {self.current_step_index}: {action_str} -> {action_code}")
                return action_code
        
        # Fallback action
        return 5  # NO_OP
    
    def _is_navigation_complete(self) -> bool:
        """Check if current navigation plan is complete."""
        if not self.current_navigation_plan or 'steps' not in self.current_navigation_plan:
            return True
        
        return self.current_step_index >= len(self.current_navigation_plan['steps'])
    
    def _plan_navigation(self, navigation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan navigation path using SLM.
        
        Args:
            navigation_context: Navigation context from environment
            
        Returns:
            Navigation plan dictionary
        """
        start_time = time.time()
        
        try:
            # Create prompt for navigation planning
            prompt = self._create_navigation_prompt(navigation_context)
            
            # Generate response from SLM with optimized settings
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": self.temperature,
                    "num_predict": 500,  # Increased to ensure complete JSON
                    "num_ctx": 2048,     # Reduce context window for speed
                    "repeat_penalty": 1.1,  # Reduce repetition for faster generation
                    "stop": ["\n\n", "```"]  # Stop at natural breaks
                }
            )
            
            # Parse navigation plan
            navigation_plan = self._parse_navigation_response(response['message']['content'])
            
            # Update performance tracking
            self.total_requests += 1
            self.total_time += time.time() - start_time
            self.successful_requests += 1
            
            return navigation_plan
            
        except Exception as e:
            logging.error(f"Failed to plan navigation: {e}")
            self.total_requests += 1
            self.total_time += time.time() - start_time
            return self._fallback_navigation_plan(navigation_context)
    
    def _create_navigation_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for navigation planning."""
        spatial_blueprint = context.get('spatial_blueprint', '')
        current_task = context.get('current_task', 'unknown')
        target_position = context.get('target_position', (0, 0))
        agent_position = context.get('agent_position', (0, 0))
        
        # Calculate the exact path
        row_diff = target_position[0] - agent_position[0]
        col_diff = target_position[1] - agent_position[1]
        
        # Generate the exact sequence of moves
        moves = []
        step_num = 1
        
        # Add row moves
        for _ in range(abs(row_diff)):
            if row_diff < 0:
                moves.append(f'{{"step": {step_num}, "action": "up", "reason": "move towards target"}}')
            else:
                moves.append(f'{{"step": {step_num}, "action": "down", "reason": "move towards target"}}')
            step_num += 1
        
        # Add column moves
        for _ in range(abs(col_diff)):
            if col_diff < 0:
                moves.append(f'{{"step": {step_num}, "action": "left", "reason": "move towards target"}}')
            else:
                moves.append(f'{{"step": {step_num}, "action": "right", "reason": "move towards target"}}')
            step_num += 1
        
        # Add interact - different reason based on task type
        interact_reason = "open door" if current_task == "door" else "collect key"
        moves.append(f'{{"step": {step_num}, "action": "interact", "reason": "{interact_reason}"}}')
        
        moves_str = ",\n    ".join(moves)
        
        prompt = f"""{spatial_blueprint}

Current Task: Navigate to {current_task} at position {target_position}
Agent Current Position: {agent_position}

EXACT PATH CALCULATION:
- Agent is at {agent_position}
- Target is at {target_position}
- Row difference: {row_diff} (need {abs(row_diff)} {"UP" if row_diff < 0 else "DOWN"} moves)
- Column difference: {col_diff} (need {abs(col_diff)} {"LEFT" if col_diff < 0 else "RIGHT"} moves)
- Total moves needed: {abs(row_diff) + abs(col_diff) + 1}

Respond with ONLY this JSON format:
{{
  "steps": [
    {moves_str}
  ],
  "total_steps": {len(moves)},
  "reasoning": "Exact path calculated: {abs(row_diff)} {"UP" if row_diff < 0 else "DOWN"} moves, {abs(col_diff)} {"LEFT" if col_diff < 0 else "RIGHT"} moves, then {interact_reason}"
}}

Rules:
- Use actions: up, down, left, right, interact
- Each step must be a valid move (not into walls)
- Last step should be "interact" when at target position
- For key tasks: interact to collect the key
- For door tasks: interact to open the door (only after collecting all keys)
- Respond with ONLY the JSON, no other text

JSON:"""
        
        return prompt
    
    def _parse_navigation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse navigation response from SLM."""
        try:
            # Clean response
            response_text = response_text.strip()
            
            # Find first complete JSON object
            start_idx = response_text.find('{')
            if start_idx == -1:
                raise ValueError("No JSON found in response")
            
            # Find the end of the first JSON object
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            json_text = response_text[start_idx:end_idx]
            
            # Parse JSON
            navigation_data = json.loads(json_text)
            
            # Validate required fields
            if 'steps' not in navigation_data:
                raise ValueError("Missing 'steps' field")
            
            # Validate steps
            steps = navigation_data['steps']
            if not isinstance(steps, list):
                raise ValueError("Steps must be a list")
            
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    raise ValueError(f"Step {i} must be a dictionary")
                if 'action' not in step:
                    raise ValueError(f"Step {i} missing 'action' field")
                
                # Validate action
                valid_actions = ['up', 'down', 'left', 'right', 'interact']
                if step['action'] not in valid_actions:
                    raise ValueError(f"Step {i} has invalid action: {step['action']}")
            
            return navigation_data
            
        except Exception as e:
            logging.error(f"Failed to parse navigation response: {e}")
            logging.error(f"Raw response: {response_text}")
            return self._fallback_navigation_plan({})
    
    def _fallback_navigation_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback navigation plan when SLM fails."""
        # Simple fallback: move towards target in a straight line
        fallback_steps = [
            {"step": 1, "action": "up", "reason": "fallback move"},
            {"step": 2, "action": "right", "reason": "fallback move"},
            {"step": 3, "action": "interact", "reason": "fallback interact"}
        ]
        return {
            "steps": fallback_steps,
            "total_steps": len(fallback_steps),
            "reasoning": "Fallback plan due to parsing error"
        }
    
    def _action_string_to_code(self, action_str: str) -> int:
        """Convert action string to action code."""
        action_map = {
            "up": 0,
            "down": 1,
            "left": 2,
            "right": 3,
            "interact": 4,
            "no_op": 5
        }
        return action_map.get(action_str.lower(), 5)  # Default to NO_OP
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if self.total_requests == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0
            }
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": self.successful_requests / self.total_requests,
            "avg_response_time": self.total_time / self.total_requests
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_requests = 0
        self.total_time = 0.0
        self.successful_requests = 0
        self.current_navigation_plan = None
        self.current_step_index = 0
